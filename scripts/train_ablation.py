"""
Parameter-matched fusion ablation for MolBCAT.

Four fusion variants:
    1. Concat baseline       : no cross-attention, no gated fusion
    2. Gate only             : no cross-attention, gated fusion
    3. Attention + Concat    : cross-attention, no gated fusion
    4. Full MolBCAT          : cross-attention + gated fusion

All four variants use the same GRU encoder, GIN encoder, sequence gate,
prediction head, training schedule, and EXACTLY the same number of
trainable fusion parameters. The non-attention controls therefore include
parameter-matched MLP capacity rather than claiming an unmatched
"simple concatenation" control.

Evaluation protocol:
    5-fold outer scaffold CV
    + scaffold-disjoint inner train/validation split
    + fold-specific hyperparameter tuning
    + validation-only threshold selection for classification
    + untouched test-fold evaluation

Datasets:
    Classification: BBBP, HIV, ClinTox, Tox21_NR_AR
    Regression:     ESOL, Lipophilicity
"""

import os
import sys
import copy
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from datasets import load_dataset
from scipy.stats import ttest_rel

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    set_seed,
    get_device,
    load_vocab,
    save_json,
    load_pretrained_encoder,
)
from src.evaluation import calc_reg_metrics, make_pos_weight
from src.dataset import (
    filter_valid,
    CrossAttnDataset,
    crossattn_collate,
    scaffold_kfold_split,
    scaffold_train_val_split,
    scaffold_kfold_split_regression,
    scaffold_train_val_split_regression,
    check_scaffold_disjoint,
)
from src.models.gru import GRUEncoder
from src.models.gin import GINEncoder


PIN_MEMORY = torch.cuda.is_available()
NUM_WORKERS = 2
CODE_VERSION = "ablation_scaffold5_parammatched_v1"

VARIANTS = [
    "Concat_Baseline",
    "Gate_Only",
    "Attention_Concat",
    "Full_MolBCAT",
]



def parse_args():
    parser = argparse.ArgumentParser(description="Parameter-matched MolBCAT ablation")
    parser.add_argument("--cls_config", type=str, default="configs/classification.yaml")
    parser.add_argument("--reg_config", type=str, default="configs/regression.yaml")
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--out_dir", type=str, default="./outputs/ablation")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def cls_metrics(y_true, probs, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    pred = (probs >= threshold).astype(int)

    result = {
        "ROC_AUC": float("nan"),
        "PR_AUC": float("nan"),
        "Precision": float("nan"),
        "Recall": float("nan"),
        "F1": float("nan"),
        "MCC": float("nan"),
        "Threshold": float(threshold),
    }

    if len(np.unique(y_true)) >= 2:
        result["ROC_AUC"] = float(roc_auc_score(y_true, probs))
        result["PR_AUC"] = float(average_precision_score(y_true, probs))
        result["Precision"] = float(precision_score(y_true, pred, zero_division=0))
        result["Recall"] = float(recall_score(y_true, pred, zero_division=0))
        result["F1"] = float(f1_score(y_true, pred, zero_division=0))
        result["MCC"] = float(matthews_corrcoef(y_true, pred))

    return result


def find_best_threshold(y_true, probs):
    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in np.arange(0.05, 0.951, 0.01):
        pred = (np.asarray(probs) >= threshold).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(round(threshold, 2))

    return best_threshold



class MolBCAT_Ablation(nn.Module):
    """Four fusion variants with an exactly matched fusion parameter budget."""

    def __init__(
        self,
        vocab_size,
        emb_dim=256,
        hidden_dim=512,
        gin_hidden=128,
        node_feat_dim=9,
        nhead=4,
        dropout=0.2,
        variant="Full_MolBCAT",
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")

        self.variant = variant
        self.pad_id = 0
        self.hidden_dim = hidden_dim

        # Common components: identical across all four variants.
        self.gru_encoder = GRUEncoder(vocab_size, emb_dim, hidden_dim)
        self.gin_encoder = GINEncoder(node_feat_dim, gin_hidden)
        self.gin_proj = nn.Linear(gin_hidden, hidden_dim)
        self.seq_gate = nn.Linear(hidden_dim * 2, hidden_dim)

        # Fusion blocks are constructed so that every variant has the same
        # number of trainable fusion parameters. The matching is verified
        # programmatically before any expensive training begins.
        if variant == "Concat_Baseline":
            self.norm_gru = nn.LayerNorm(hidden_dim)
            self.norm_gin = nn.LayerNorm(hidden_dim)
            # 1024 -> 800 -> 842 -> 834 -> 512
            # Together with two LayerNorms: 2,627,072 parameters.
            self.fusion_mlp = nn.Sequential(
                nn.Linear(2 * hidden_dim, 800),
                nn.ReLU(),
                nn.Linear(800, 842),
                nn.ReLU(),
                nn.Linear(842, 834),
                nn.ReLU(),
                nn.Linear(834, hidden_dim),
            )

        elif variant == "Gate_Only":
            self.norm_gru = nn.LayerNorm(hidden_dim)
            self.norm_gin = nn.LayerNorm(hidden_dim)
            self.concat_proj = nn.Linear(2 * hidden_dim, hidden_dim)
            self.gate_fc = nn.Linear(2 * hidden_dim, hidden_dim)
            # Parameter-matched gate capacity. Both projections are active:
            # their nonlinear gate proposals are averaged before fusion.
            self.compensator = nn.Sequential(
                nn.Linear(hidden_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, hidden_dim),
            )

        elif variant == "Attention_Concat":
            self.attn_sg = nn.MultiheadAttention(
                hidden_dim, nhead, dropout=dropout, batch_first=True
            )
            self.attn_gs = nn.MultiheadAttention(
                hidden_dim, nhead, dropout=dropout, batch_first=True
            )
            self.norm = nn.LayerNorm(hidden_dim)
            self.concat_proj = nn.Linear(2 * hidden_dim, hidden_dim)

        elif variant == "Full_MolBCAT":
            self.attn_sg = nn.MultiheadAttention(
                hidden_dim, nhead, dropout=dropout, batch_first=True
            )
            self.attn_gs = nn.MultiheadAttention(
                hidden_dim, nhead, dropout=dropout, batch_first=True
            )
            self.norm = nn.LayerNorm(hidden_dim)
            self.gate_fc = nn.Linear(2 * hidden_dim, hidden_dim)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def _graph_sequence_attention(self, smiles_x, graph_data):
        """Return the two attended modality representations used by attention variants."""
        gru_out = self.gru_encoder(smiles_x)
        h_cls = gru_out[:, 0, :]
        mask = (smiles_x != self.pad_id).float()
        h_mean = (
            (gru_out * mask.unsqueeze(-1)).sum(1)
            / mask.sum(1, keepdim=True).clamp(min=1)
        )
        h_seq = torch.sigmoid(self.seq_gate(torch.cat([h_cls, h_mean], dim=1))).unsqueeze(1)

        atom_emb, bvec = self.gin_encoder(graph_data)
        atom_emb = self.gin_proj(atom_emb)

        batch_size = smiles_x.size(0)
        max_atoms = max((bvec == i).sum().item() for i in range(batch_size))
        max_atoms = max(max_atoms, 1)

        atom_pad = torch.zeros(
            batch_size, max_atoms, self.hidden_dim,
            device=smiles_x.device,
            dtype=atom_emb.dtype,
        )
        key_pad_mask = torch.ones(
            batch_size, max_atoms,
            device=smiles_x.device,
            dtype=torch.bool,
        )
        atom_counts = []

        for i in range(batch_size):
            a = atom_emb[bvec == i]
            n = a.size(0)
            atom_counts.append(n)
            atom_pad[i, :n] = a
            key_pad_mask[i, :n] = False

        atom_counts = torch.tensor(
            atom_counts, device=smiles_x.device, dtype=torch.float32
        )

        h_sg, _ = self.attn_sg(
            h_seq, atom_pad, atom_pad,
            key_padding_mask=key_pad_mask,
        )
        h_sg = self.norm(h_sg.squeeze(1))

        h_gs_all, _ = self.attn_gs(
            atom_pad, gru_out, gru_out,
            key_padding_mask=(smiles_x == self.pad_id),
        )
        h_gs_all = h_gs_all * (~key_pad_mask).unsqueeze(-1).float()
        h_gs = self.norm(
            h_gs_all.sum(1) / atom_counts.unsqueeze(1).clamp(min=1)
        )

        return h_sg, h_gs

    def forward(self, smiles_x, graph_data):
        if self.variant in {"Attention_Concat", "Full_MolBCAT"}:
            h_a, h_b = self._graph_sequence_attention(smiles_x, graph_data)
            h_cat = torch.cat([h_a, h_b], dim=1)

            if self.variant == "Attention_Concat":
                h_out = self.concat_proj(h_cat)
            else:
                gate = torch.sigmoid(self.gate_fc(h_cat))
                h_out = gate * h_a + (1.0 - gate) * h_b

            return self.head(h_out)

  
        gru_out = self.gru_encoder(smiles_x)
        h_cls = gru_out[:, 0, :]
        mask = (smiles_x != self.pad_id).float()
        h_mean = (
            (gru_out * mask.unsqueeze(-1)).sum(1)
            / mask.sum(1, keepdim=True).clamp(min=1)
        )
        h_gru = torch.sigmoid(self.seq_gate(torch.cat([h_cls, h_mean], dim=1)))

        from torch_geometric.nn import global_mean_pool
        atom_emb, bvec = self.gin_encoder(graph_data)
        h_gin = self.gin_proj(global_mean_pool(atom_emb, bvec))

        h_gru = self.norm_gru(h_gru)
        h_gin = self.norm_gin(h_gin)
        h_cat = torch.cat([h_gru, h_gin], dim=1)

        if self.variant == "Concat_Baseline":
            h_out = self.fusion_mlp(h_cat)
        else:
            gate = 0.5 * (
                torch.sigmoid(self.gate_fc(h_cat))
                + torch.sigmoid(self.concat_proj(h_cat))
            )
            h_out = gate * h_gru + (1.0 - gate) * h_gin
            h_out = self.compensator(h_out)

        return self.head(h_out)


def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def fusion_parameter_count(model):
    common = {
        "gru_encoder", "gin_encoder", "gin_proj", "seq_gate", "head"
    }
    return sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and name.split(".")[0] not in common
    )


def check_parameter_matching(models):
    counts = {name: fusion_parameter_count(model) for name, model in models.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"Fusion parameter mismatch: {counts}")

    total_counts = {name: count_parameters(model) for name, model in models.items()}
    if len(set(total_counts.values())) != 1:
        raise RuntimeError(f"Total parameter mismatch: {total_counts}")

    print(f"  Parameter-matched fusion params: {next(iter(counts.values())):,}")
    print(f"  Total trainable parameters:      {next(iter(total_counts.values())):,}")
    return counts, total_counts


def make_loaders(X_train, y_train, X_val, y_val, vocab, max_len, batch_size):
    train_ds = CrossAttnDataset(X_train, y_train, vocab, max_len)
    val_ds = CrossAttnDataset(X_val, y_val, vocab, max_len)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=crossattn_collate, num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=crossattn_collate, num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    return train_loader, val_loader


def make_test_loader(X_test, y_test, vocab, max_len, batch_size):
    test_ds = CrossAttnDataset(X_test, y_test, vocab, max_len)
    return DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=crossattn_collate, num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


def infer_model(model, loader, task, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for smiles_x, graph_data, labs in loader:
            out = model(smiles_x.to(device), graph_data.to(device)).view(-1)
            if task == "cls":
                out = torch.sigmoid(out)
            preds.extend(out.detach().cpu().numpy().tolist())
            trues.extend(labs.detach().cpu().numpy().tolist())
    return np.asarray(preds), np.asarray(trues)


def train_variant(
    model,
    train_loader,
    val_loader,
    pos_weight,
    lr_encoder,
    lr_head,
    epochs,
    early_stop,
    task,
    device,
):
    encoder_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("head.")
    ]
    head_params = list(model.head.parameters())

    optimizer = torch.optim.Adam([
        {"params": encoder_params, "lr": lr_encoder},
        {"params": head_params, "lr": lr_head},
    ])

    loss_fn = (
        nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if task == "cls" else nn.MSELoss()
    )

    best_metric = -float("inf") if task == "cls" else float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience = 0

    for epoch in range(epochs):
        req_grad = epoch >= 3
        for p in model.gru_encoder.parameters():
            p.requires_grad = req_grad

        model.train()
        for smiles_x, graph_data, labs in train_loader:
            smiles_x = smiles_x.to(device)
            graph_data = graph_data.to(device)
            labs = labs.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(smiles_x, graph_data).view(-1), labs)
            loss.backward()
            optimizer.step()

        val_pred, val_true = infer_model(model, val_loader, task, device)
        if task == "cls":
            if len(np.unique(val_true)) < 2:
                continue
            metric = roc_auc_score(val_true, val_pred)
            improved = metric > best_metric + 1e-4
        else:
            metric = calc_reg_metrics(val_true, val_pred)["RMSE"]
            improved = metric < best_metric - 1e-4

        if improved:
            best_metric = float(metric)
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= early_stop:
                break

    model.load_state_dict(best_state)
    return model, best_metric


def build_model(cfg, vocab, variant, device):
    model = MolBCAT_Ablation(
        len(vocab),
        cfg["model"]["emb_dim"],
        cfg["model"]["hidden_dim"],
        cfg["model"]["gin_hidden_dim"],
        cfg["model"]["node_feat_dim"],
        cfg["model"]["nhead"],
        variant=variant,
    ).to(device)
    return model


def tune_variant(cfg, X_train, y_train, X_val, y_val, vocab, device, task, variant, fold):
    grid = cfg["grids"]["molbcat"]
    pos_weight = make_pos_weight(y_train, device) if task == "cls" else None
    tl, vl = make_loaders(
        X_train, y_train, X_val, y_val, vocab,
        cfg["model"]["max_len"], cfg["training"]["batch_size"],
    )

    best_hp = None
    best_val = -float("inf") if task == "cls" else float("inf")

    for trial_id, trial in enumerate(grid):
        set_seed(cfg["split"]["random_seed"] + fold * 1000 + trial_id)
        model = build_model(cfg, vocab, variant, device)
        load_pretrained_encoder(
            model.gru_encoder,
            cfg["weights"]["pretrained_encoder"],
            device,
        )
        model, val_metric = train_variant(
            model, tl, vl, pos_weight,
            trial["lr_encoder"], trial["lr_head"],
            cfg["training"].get("molbcat_epochs", cfg["training"]["epochs"]),
            cfg["training"]["early_stop"], task, device,
        )

        improved = (
            val_metric > best_val if task == "cls" else val_metric < best_val
        )
        if improved:
            best_val = val_metric
            best_hp = copy.deepcopy(trial)

    if best_hp is None:
        raise RuntimeError(f"No valid HP found for {variant}, fold {fold}")
    return best_hp


def run_fold(cfg, dataset_name, smiles, labels, train_idx, val_idx, test_idx,
             vocab, device, task, fold):
    X_train = [smiles[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    X_val = [smiles[i] for i in val_idx]
    y_val = [labels[i] for i in val_idx]
    X_test = [smiles[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]

    print(
        f"  Fold {fold}: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )

    # Class guards for classification.
    if task == "cls":
        for part_name, part_y in [("train", y_train), ("val", y_val), ("test", y_test)]:
            if len(np.unique(part_y)) < 2:
                raise RuntimeError(
                    f"{dataset_name} Fold {fold}: {part_name} contains only one class."
                )

    hps = {}
    for variant in VARIANTS:
        hps[variant] = tune_variant(
            cfg, X_train, y_train, X_val, y_val,
            vocab, device, task, variant, fold,
        )

    fold_results = {
        "protocol_version": CODE_VERSION,
        "fold": fold,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "hyperparameters": hps,
        "results": {},
    }

    for variant in VARIANTS:
        set_seed(cfg["split"]["random_seed"] + fold * 1000)
        model = build_model(cfg, vocab, variant, device)
        load_pretrained_encoder(
            model.gru_encoder,
            cfg["weights"]["pretrained_encoder"],
            device,
        )

        tl, vl = make_loaders(
            X_train, y_train, X_val, y_val, vocab,
            cfg["model"]["max_len"], cfg["training"]["batch_size"],
        )
        test_loader = make_test_loader(
            X_test, y_test, vocab,
            cfg["model"]["max_len"], cfg["training"]["batch_size"],
        )
        pos_weight = make_pos_weight(y_train, device) if task == "cls" else None
        hp = hps[variant]

        model, _ = train_variant(
            model, tl, vl, pos_weight,
            hp["lr_encoder"], hp["lr_head"],
            cfg["training"].get("molbcat_epochs", cfg["training"]["epochs"]),
            cfg["training"]["early_stop"], task, device,
        )

        val_pred, val_true = infer_model(model, vl, task, device)
        test_pred, test_true = infer_model(model, test_loader, task, device)

        if task == "cls":
            # Threshold is selected only from inner validation predictions.
            threshold = find_best_threshold(val_true, val_pred)
            metrics = cls_metrics(test_true, test_pred, threshold)
        else:
            threshold = None
            metrics = calc_reg_metrics(test_true, test_pred)

        metrics["threshold"] = threshold
        fold_results["results"][variant] = metrics

        print(
            f"    {variant:<20} "
            + (
                f"ROC-AUC={metrics['ROC_AUC']:.4f} F1={metrics['F1']:.4f} "
                f"thr={threshold:.2f}"
                if task == "cls"
                else f"RMSE={metrics['RMSE']:.4f} MAE={metrics['MAE']:.4f}"
            )
        )

    return fold_results


def summarize(fold_results, dataset_name, task):
    metrics = (
        ["ROC_AUC", "PR_AUC", "Precision", "Recall", "F1", "MCC"]
        if task == "cls"
        else ["RMSE", "MAE", "R2"]
    )
    rows = []

    for variant in VARIANTS:
        row = {"Dataset": dataset_name, "Variant": variant}
        for metric in metrics:
            vals = [
                fold["results"][variant].get(metric, np.nan)
                for fold in fold_results
            ]
            vals = [float(v) for v in vals if np.isfinite(v)]
            row[f"{metric}_mean"] = float(np.mean(vals)) if vals else np.nan
            row[f"{metric}_std"] = (
                float(np.std(vals, ddof=1)) if len(vals) >= 2 else np.nan
            )
        rows.append(row)

    return rows


def holm_adjust(p_values):
    """Return Holm-adjusted p-values in the original test order."""
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p_values[idx])
        running_max = max(running_max, value)
        adjusted[idx] = running_max
    return adjusted.tolist()


def paired_ablation_stats(fold_results, dataset_name, task):
    """Paired outer-fold tests; Holm correction is applied within each dataset."""
    metrics = (
        ["ROC_AUC", "PR_AUC", "Precision", "Recall", "F1", "MCC"]
        if task == "cls" else ["RMSE", "MAE", "R2"]
    )
    comparisons = []
    raw_p = []

    for variant in VARIANTS:
        if variant == "Full_MolBCAT":
            continue
        for metric in metrics:
            full = np.asarray(
                [f["results"]["Full_MolBCAT"][metric] for f in fold_results],
                dtype=float,
            )
            other = np.asarray(
                [f["results"][variant][metric] for f in fold_results],
                dtype=float,
            )
            mask = np.isfinite(full) & np.isfinite(other)
            if mask.sum() >= 2:
                stat, p = ttest_rel(full[mask], other[mask])
                stat = float(stat)
                p = float(p)
            else:
                stat, p = float("nan"), float("nan")
            comparisons.append({
                "Dataset": dataset_name,
                "Task": task,
                "Comparison": f"Full_MolBCAT_vs_{variant}",
                "Metric": metric,
                "n_folds": int(mask.sum()),
                "t_stat": stat,
                "p_raw": p,
            })
            raw_p.append(p)

    finite = np.isfinite(raw_p)
    if finite.any():
        finite_adj = holm_adjust(np.asarray(raw_p)[finite])
        j = 0
        for i, is_finite in enumerate(finite):
            if is_finite:
                comparisons[i]["p_holm"] = float(finite_adj[j])
                j += 1
            else:
                comparisons[i]["p_holm"] = float("nan")
    else:
        for row in comparisons:
            row["p_holm"] = float("nan")

    return comparisons


def run_dataset(dataset_cfg, cfg, vocab, device, task, out_dir, resume=False):
    name = dataset_cfg["name"]
    print(f"\n{'=' * 70}\nABLATION — {name}\n{'=' * 70}")

    ds = load_dataset(dataset_cfg["hf"])["train"]
    smiles, labels = filter_valid(
        ds["SMILES"], ds[dataset_cfg["col"]], task=task
    )

    if task == "cls":
        folds = scaffold_kfold_split(
            smiles,
            labels=labels,
            n_folds=cfg["split"]["n_folds"],
            seed=cfg["split"]["random_seed"],
        )
    else:
        folds = scaffold_kfold_split_regression(
            smiles,
            labels=labels,
            n_folds=cfg["split"]["n_folds"],
            seed=cfg["split"]["random_seed"],
        )

    result_dir = os.path.join(out_dir, name)
    os.makedirs(result_dir, exist_ok=True)
    fold_results = []

    # Parameter check is performed once before expensive training.
    models_for_check = {
        variant: build_model(cfg, vocab, variant, device)
        for variant in VARIANTS
    }
    check_parameter_matching(models_for_check)
    del models_for_check

    for fold_id, test_idx in enumerate(folds):
        fold = fold_id + 1
        fold_path = os.path.join(result_dir, f"fold_{fold}.json")

        if resume and os.path.exists(fold_path):
            import json
            with open(fold_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if (
                cached.get("protocol_version") == CODE_VERSION
                and set(cached.get("results", {}).keys()) == set(VARIANTS)
            ):
                fold_results.append(cached)
                print(f"  [RESUME] Fold {fold} loaded.")
                continue
            print(
                f"  [RESUME] Fold {fold} checkpoint is from an incompatible "
                f"protocol/version; rerunning this fold."
            )

        test_set = set(test_idx)
        development_idx = [
            i for i in range(len(smiles)) if i not in test_set
        ]

        if task == "cls":
            train_idx, val_idx = scaffold_train_val_split(
                smiles,
                development_idx,
                labels=labels,
                val_ratio=cfg["split"]["inner_val_ratio"],
                seed=cfg["split"]["random_seed"] + fold_id,
            )
        else:
            train_idx, val_idx = scaffold_train_val_split_regression(
                smiles,
                development_idx,
                val_ratio=cfg["split"]["inner_val_ratio"],
                seed=cfg["split"]["random_seed"] + fold_id,
            )

        overlap = check_scaffold_disjoint(
            smiles, train_idx, val_idx, test_idx
        )
        if any(v != 0 for v in overlap.values()):
            raise RuntimeError(
                f"Scaffold leakage in {name}, Fold {fold}: {overlap}"
            )

        result = run_fold(
            cfg, name, smiles, labels,
            train_idx, val_idx, test_idx,
            vocab, device, task, fold,
        )
        fold_results.append(result)
        save_json(fold_path, result)

    if len(fold_results) != cfg["split"]["n_folds"]:
        raise RuntimeError(
            f"Expected {cfg['split']['n_folds']} completed folds, "
            f"but found {len(fold_results)}."
        )

    save_json(
        os.path.join(result_dir, "fold_results.json"),
        fold_results,
    )

    rows = summarize(fold_results, name, task)
    save_json(
        os.path.join(result_dir, "summary.json"),
        rows,
    )

    stats = paired_ablation_stats(fold_results, name, task)
    save_json(
        os.path.join(result_dir, "paired_stats.json"),
        stats,
    )
    return rows

def main(args=None):
    if args is None:
        args = parse_args()

    with open(args.cls_config, "r", encoding="utf-8") as f:
        cls_cfg = yaml.safe_load(f)
    with open(args.reg_config, "r", encoding="utf-8") as f:
        reg_cfg = yaml.safe_load(f)

    device = get_device()
    vocab = load_vocab(cls_cfg["weights"]["vocab"])
    os.makedirs(args.out_dir, exist_ok=True)

    cls_names = ["BBBP", "HIV"]
    reg_names = ["ESOL"]
    all_rows = []

    if args.dataset == "all":
        selected = [("cls", n) for n in cls_names] + [("reg", n) for n in reg_names]
    elif args.dataset in cls_names:
        selected = [("cls", args.dataset)]
    elif args.dataset in reg_names:
        selected = [("reg", args.dataset)]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    for task, name in selected:
        cfg = cls_cfg if task == "cls" else reg_cfg
        ds_cfg = next(
            d for d in cfg["data"]["datasets"] if d["name"] == name
        )
        rows = run_dataset(
            ds_cfg, cfg, vocab, device, task,
            args.out_dir, args.resume,
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(args.out_dir, "ablation_5fold_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved ablation summary to: {out_csv}")


if __name__ == "__main__":
    main()
