"""
Concatenation ablation study.
Compares MolBCAT (cross-modal attention) vs MolBCAT_Concat
(simple concatenation fusion) on BBBP, ClinTox, and Lipophilicity.

Usage:
    python scripts/train_ablation.py --cls_config configs/classification.yaml \
                                     --reg_config configs/regression.yaml
"""
import os
import sys
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import yaml
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeoLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_seed, get_device, load_vocab, load_json, save_json, load_pretrained_encoder
from src.evaluation import calc_cls_metrics, calc_reg_metrics, make_pos_weight, safe_auc
from src.dataset import filter_valid, scaffold_split, CrossAttnDataset, crossattn_collate
from src.models import GRUModel, MolBCAT
from src.trainer import train_molbcat, infer_molbcat


def parse_args():
    parser = argparse.ArgumentParser(description='Concatenation ablation study')
    parser.add_argument('--cls_config', type=str, default='configs/classification.yaml')
    parser.add_argument('--reg_config', type=str, default='configs/regression.yaml')
    parser.add_argument('--out_dir',    type=str, default='./outputs/ablation')
    return parser.parse_args()


class MolBCAT_Concat(nn.Module):
    """
    Ablation model: replaces bidirectional cross-modal attention
    with simple feature concatenation followed by a projection layer.

    Fair comparison — same parameter count and training schedule as MolBCAT.
    """

    def __init__(self, vocab_size: int,
                 emb_dim:       int = 256,
                 hidden_dim:    int = 512,
                 gin_hidden:    int = 128,
                 node_feat_dim: int = 9,
                 dropout:       float = 0.2):
        super().__init__()
        from src.models.gru import GRUEncoder
        from src.models.gin import GINEncoder

        self.pad_id      = 0
        self.gru_encoder = GRUEncoder(vocab_size, emb_dim, hidden_dim)
        self.gin_encoder = GINEncoder(node_feat_dim, gin_hidden)
        self.gin_proj    = nn.Linear(gin_hidden, hidden_dim)
        self.seq_gate    = nn.Linear(hidden_dim * 2, hidden_dim)

        # LayerNorm to equalise scale before concat (fairness)
        self.norm_gru    = nn.LayerNorm(hidden_dim)
        self.norm_gin    = nn.LayerNorm(hidden_dim)

        # Project concat(H, H) → H so output dim == MolBCAT
        self.proj        = nn.Linear(hidden_dim * 2, hidden_dim)

        # Same head architecture as MolBCAT
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_dim, 256),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 1),
        )

    def forward(self, smiles_x: torch.Tensor, graph_data) -> torch.Tensor:
        from torch_geometric.nn import global_mean_pool

        # Sequence branch
        gru_out = self.gru_encoder(smiles_x)
        h_cls   = gru_out[:, 0, :]
        mask    = (smiles_x != self.pad_id).float()
        h_mean  = (
            (gru_out * mask.unsqueeze(-1)).sum(1)
            / mask.sum(1, keepdim=True).clamp(min=1)
        )
        h_gru = torch.sigmoid(self.seq_gate(torch.cat([h_cls, h_mean], dim=1)))

        # Graph branch
        atom_emb, bvec = self.gin_encoder(graph_data)
        h_gin = self.gin_proj(global_mean_pool(atom_emb, bvec))

        # Normalize + concat + project (no cross-attention)
        h_gru = self.norm_gru(h_gru)
        h_gin = self.norm_gin(h_gin)
        return self.head(self.proj(torch.cat([h_gru, h_gin], dim=1)))


def _train_concat(model, train_loader, val_loader,
                   pos_weight, lr_encoder, lr_head,
                   epochs, early_stop, task, device) -> tuple:
    """
    Train MolBCAT_Concat with same schedule as MolBCAT:
    - Separate LR for encoder vs head
    - Freeze GRU encoder for first 3 epochs
    """
    encoder_params = (
        list(model.gru_encoder.parameters()) +
        list(model.gin_encoder.parameters()) +
        list(model.gin_proj.parameters()) +
        list(model.seq_gate.parameters()) +
        list(model.norm_gru.parameters()) +
        list(model.norm_gin.parameters()) +
        list(model.proj.parameters())
    )
    head_params = list(model.head.parameters())

    opt = torch.optim.Adam([
        {'params': encoder_params, 'lr': lr_encoder},
        {'params': head_params,    'lr': lr_head},
    ])
    loss_fn = (
        nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if task == 'cls' else nn.MSELoss()
    )

    best_metric = 0.0 if task == 'cls' else float('inf')
    best_state  = copy.deepcopy(model.state_dict())
    patience    = 0

    for epoch in range(epochs):
        req_grad = epoch >= 3
        for p in model.gru_encoder.parameters():
            p.requires_grad = req_grad

        model.train()
        for smiles_x, graph_data, labs in train_loader:
            smiles_x   = smiles_x.to(device)
            graph_data = graph_data.to(device)
            labs       = labs.to(device)
            opt.zero_grad()
            loss_fn(model(smiles_x, graph_data).view(-1), labs).backward()
            opt.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for smiles_x, graph_data, labs in val_loader:
                out = model(smiles_x.to(device), graph_data.to(device)).view(-1)
                preds.extend((torch.sigmoid(out) if task == 'cls'
                               else out).cpu().numpy())
                trues.extend(labs.numpy())

        if task == 'cls':
            metric   = safe_auc(trues, preds)
            if np.isnan(metric):
                continue
            improved = metric > best_metric + 1e-4
        else:
            metric = calc_reg_metrics(trues, preds)['RMSE']
            improved = metric < best_metric - 1e-4

        if improved:
            best_metric = metric
            best_state  = copy.deepcopy(model.state_dict())
            patience    = 0
        else:
            patience += 1
            if patience >= early_stop:
                print(f"    Early stop at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    return model, best_metric


def run_ablation(dataset_cfg: dict, cfg: dict,
                 vocab: dict, device: str,
                 task: str, out_dir: str) -> dict:
    """Run Phase 1 + Phase 2 for MolBCAT_Concat on one dataset."""
    name = dataset_cfg['name']
    print(f"\n{'='*55}\nABLATION — {name}\n{'='*55}")

    os.makedirs(out_dir, exist_ok=True)
    p1_path  = os.path.join(out_dir, f'{name}_phase1.json')
    p2_path  = os.path.join(out_dir, f'{name}_phase2.json')
    best_hps = load_json(p1_path)
    p2       = load_json(p2_path)

    ds = load_dataset(dataset_cfg['hf'])['train']
    smiles, labels = filter_valid(ds['SMILES'], ds[dataset_cfg['col']], task=task)
    vocab_size     = len(vocab)

    grid = cfg['grids']['molbcat']

    # Phase 1 — HP tuning
    if 'lr_encoder' not in best_hps:
        print("  Phase 1: HP tuning...")
        set_seed(cfg['split']['phase1_seed'])
        train_idx, _ = scaffold_split(smiles, cfg['split']['phase1_seed'],
                                       cfg['split']['test_ratio'])
        X_tr_all = [smiles[i] for i in train_idx]
        y_tr_all = [labels[i] for i in train_idx]
        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_tr_all, y_tr_all, test_size=cfg['split']['val_ratio'],
                stratify=y_tr_all if task == 'cls' else None,
                random_state=cfg['split']['phase1_seed'])
        except ValueError:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_tr_all, y_tr_all, test_size=cfg['split']['val_ratio'],
                random_state=cfg['split']['phase1_seed'])

        pos_weight = make_pos_weight(y_tr, device) if task == 'cls' else None
        tl = DataLoader(CrossAttnDataset(X_tr,  y_tr,  vocab, cfg['model']['max_len']),
                        batch_size=cfg['training']['batch_size'],
                        shuffle=True, collate_fn=crossattn_collate)
        vl = DataLoader(CrossAttnDataset(X_val, y_val, vocab, cfg['model']['max_len']),
                        batch_size=cfg['training']['batch_size'],
                        collate_fn=crossattn_collate)

        best_trial = None
        best_val   = 0.0 if task == 'cls' else float('inf')
        for trial in grid:
            set_seed(cfg['split']['phase1_seed'])
            try:
                m = MolBCAT_Concat(vocab_size, cfg['model']['emb_dim'],
                                    cfg['model']['hidden_dim'],
                                    cfg['model']['gin_hidden_dim'],
                                    cfg['model']['node_feat_dim']).to(device)
                m.pad_id = vocab.get('<PAD>', 0)
                load_pretrained_encoder(m.gru_encoder,
                                         cfg['weights']['pretrained_encoder'], device)
                _, val_m = _train_concat(
                    m, tl, vl, pos_weight,
                    trial['lr_encoder'], trial['lr_head'],
                    50, cfg['training']['early_stop'],
                    task, device)
                print(f"    lr_enc={trial['lr_encoder']} lr_head={trial['lr_head']} → {val_m:.4f}")
                improved = (val_m > best_val) if task == 'cls' else (val_m < best_val)
                if improved:
                    best_val, best_trial = val_m, copy.deepcopy(trial)
            except Exception as e:
                print(f"    Trial failed: {e}")

        best_hps = best_trial or {'lr_encoder': 1e-5, 'lr_head': 1e-3}
        save_json(p1_path, best_hps)
        print(f"  Best HPs: {best_hps}")

    # Phase 2 — Multi-seed evaluation
    seeds     = cfg['split']['phase2_seeds']
    remaining = [s for s in seeds if str(s) not in p2]
    print(f"  Phase 2: remaining seeds = {remaining}")

    if remaining:
        for seed in remaining:
            print(f"\n  Seed {seed}")
            set_seed(seed)
            train_idx, test_idx = scaffold_split(smiles, seed, cfg['split']['test_ratio'])
            X_train = [smiles[i] for i in train_idx]
            y_train = [labels[i] for i in train_idx]
            X_test  = [smiles[i] for i in test_idx]
            y_test  = [labels[i] for i in test_idx]
            try:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train, y_train, test_size=cfg['split']['val_ratio'],
                    stratify=y_train if task == 'cls' else None,
                    random_state=seed)
            except ValueError:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train, y_train, test_size=cfg['split']['val_ratio'],
                    random_state=seed)

            pos_weight = make_pos_weight(y_tr, device) if task == 'cls' else None
            train_ds   = CrossAttnDataset(X_tr,   y_tr,   vocab, cfg['model']['max_len'])
            val_ds     = CrossAttnDataset(X_val,  y_val,  vocab, cfg['model']['max_len'])
            test_ds    = CrossAttnDataset(X_test, y_test, vocab, cfg['model']['max_len'])

            try:
                tl      = DataLoader(train_ds, batch_size=cfg['training']['batch_size'],
                                     shuffle=True, collate_fn=crossattn_collate)
                vl      = DataLoader(val_ds,   batch_size=cfg['training']['batch_size'],
                                     collate_fn=crossattn_collate)
                tl_test = DataLoader(test_ds,  batch_size=cfg['training']['batch_size'],
                                     collate_fn=crossattn_collate)

                set_seed(seed)
                m = MolBCAT_Concat(vocab_size, cfg['model']['emb_dim'],
                                    cfg['model']['hidden_dim'],
                                    cfg['model']['gin_hidden_dim'],
                                    cfg['model']['node_feat_dim']).to(device)
                m.pad_id = vocab.get('<PAD>', 0)
                load_pretrained_encoder(m.gru_encoder,
                                         cfg['weights']['pretrained_encoder'], device)
                m, _ = _train_concat(
                    m, tl, vl, pos_weight,
                    best_hps['lr_encoder'], best_hps['lr_head'],
                    50, cfg['training']['early_stop'],
                    task, device)

                preds, trues = infer_molbcat(m, tl_test, task, device)
                metrics = (calc_cls_metrics(trues, preds) if task == 'cls'
                           else calc_reg_metrics(trues, preds))
                p2[str(seed)] = metrics
                save_json(p2_path, p2)

                key_metric = 'ROC_AUC' if task == 'cls' else 'RMSE'
                print(f"    {key_metric}={metrics[key_metric]:.4f}")

            except Exception as e:
                print(f"    Seed {seed} failed: {e}")

    return load_json(p2_path)


def summarize(p2: dict, dataset_name: str, seeds: list, task: str) -> dict:
    metrics = ['ROC_AUC', 'PR_AUC', 'Precision', 'Recall', 'F1'] if task == 'cls' \
              else ['RMSE', 'MAE', 'R2']
    result = {'Dataset': dataset_name, 'Model': 'MolBCAT_Concat'}
    for m in metrics:
        vals = [p2[str(s)][m] for s in seeds
                if str(s) in p2 and not np.isnan(p2[str(s)][m])]
        result[f'{m}_mean'] = float(np.mean(vals)) if vals else float('nan')
        result[f'{m}_std']  = float(np.std(vals))  if vals else float('nan')
    return result


def main(args=None):
    if args is None:
        args = parse_args()

    with open(args.cls_config) as f:
        cls_cfg = yaml.safe_load(f)
    with open(args.reg_config) as f:
        reg_cfg = yaml.safe_load(f)

    device = get_device()
    vocab  = load_vocab(cls_cfg['weights']['vocab'])

    os.makedirs(args.out_dir, exist_ok=True)
    all_results = []

    # Classification: BBBP, ClinTox
    for ds_name in ['BBBP', 'ClinTox']:
        ds_cfg = next((d for d in cls_cfg['data']['datasets']
                       if d['name'] == ds_name), None)
        if ds_cfg is None:
            continue
        try:
            p2     = run_ablation(ds_cfg, cls_cfg, vocab, device,
                                   task='cls', out_dir=args.out_dir)
            result = summarize(p2, ds_name, cls_cfg['split']['phase2_seeds'], 'cls')
            all_results.append(result)
        except Exception as e:
            print(f"ERROR on {ds_name}: {e}")

    # Regression: Lipophilicity
    lipo_cfg = next((d for d in reg_cfg['data']['datasets']
                     if d['name'] == 'Lipophilicity'), None)
    if lipo_cfg:
        try:
            p2     = run_ablation(lipo_cfg, reg_cfg, vocab, device,
                                   task='reg', out_dir=args.out_dir)
            result = summarize(p2, 'Lipophilicity',
                                reg_cfg['split']['phase2_seeds'], 'reg')
            all_results.append(result)
        except Exception as e:
            print(f"ERROR on Lipophilicity: {e}")

    if not all_results:
        print("No results to summarize.")
        return

    # Print and save
    print(f"\n{'='*60}\nABLATION SUMMARY — MolBCAT_Concat\n{'='*60}")
    df = pd.DataFrame(all_results)
    print(df.to_string(index=False))

    out_csv = os.path.join(args.out_dir, 'ablation_summary.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nSaved to {out_csv}")


if __name__ == '__main__':
    main()
