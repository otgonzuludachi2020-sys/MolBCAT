"""
MolBCAT data-efficiency experiment — scaffold-disjoint 5-fold version.

"""

import os
import sys
import json
import copy
import argparse
import random

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Run from the repository root; this script should live in scripts/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils import set_seed, get_device, load_vocab, save_json
from src.dataset import (
    filter_valid,
    scaffold_kfold_split,
    scaffold_train_val_split,
    scaffold_kfold_split_regression,
    scaffold_train_val_split_regression,
    check_scaffold_disjoint,
)

from scripts import train_cls as cls_runner
from scripts import train_reg as reg_runner


RATIOS = [0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
REPORT_MODELS = ["GRU_Random", "GRU_Finetune", "MolBCAT"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Data-efficiency experiment using 5-fold scaffold CV"
    )
    p.add_argument("--cls_config", default="configs/classification.yaml")
    p.add_argument("--reg_config", default="configs/regression.yaml")
    p.add_argument("--out_dir", default="outputs/data_efficiency_scaffold5fold")
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


def _save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=True)


def _stratified_subset(indices, labels, ratio, seed, task):
    """Select a reproducible molecule-level subset from the outer train pool."""
    indices = np.asarray(indices, dtype=int)
    n_total = len(indices)
    n_take = n_total if ratio >= 1.0 else max(16, int(round(n_total * ratio)))
    n_take = min(n_take, n_total)

    if n_take == n_total:
        return indices.tolist()

    y = np.asarray([labels[i] for i in indices])

    if task == "cls":
        # Stratification requires at least two examples per class in the
        # selected subset and in the remaining pool.
        class_counts = np.unique(y, return_counts=True)[1]
        if n_take < len(class_counts) * 2 or np.min(class_counts) < 2:
            raise ValueError(
                "The requested classification training fraction is too small "
                "to preserve both classes by stratified sampling."
            )
        # The subset remains inside the scaffold-separated training pool.
        chosen, _ = train_test_split(
            indices,
            train_size=n_take,
            random_state=seed,
            shuffle=True,
            stratify=y,
        )
        return sorted(chosen.tolist())

    rng = np.random.default_rng(seed)
    return sorted(rng.choice(indices, size=n_take, replace=False).tolist())


def _check_splits(smiles, train_idx, val_idx, test_idx, dataset, fold, ratio):
    overlap = check_scaffold_disjoint(smiles, train_idx, val_idx, test_idx)
    if any(v != 0 for v in overlap.values()):
        raise RuntimeError(
            f"Scaffold leakage: {dataset}, fold={fold}, ratio={ratio}: {overlap}"
        )
    return overlap


def _classification_one_setting(
    dataset_cfg, cfg, vocab, device, smiles, labels,
    train_idx, val_idx, test_idx, fold_id
):

    X_tr = [smiles[i] for i in train_idx]
    y_tr = [labels[i] for i in train_idx]
    X_val = [smiles[i] for i in val_idx]
    y_val = [labels[i] for i in val_idx]
    X_test = [smiles[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
        raise ValueError("BBBP training and validation subsets must contain both classes.")

    cr = cls_runner
    pos_weight = cr.make_pos_weight(y_tr, device)
    if pos_weight is None:
        raise ValueError("BBBP training labels are degenerate.")

    seed = cfg["split"]["random_seed"] + fold_id
    best_hps = {}
    train_loader, val_loader = cr.make_gru_loaders(
        X_tr, y_tr, X_val, y_val, vocab,
        cfg["model"]["max_len"], cfg["training"]["batch_size"]
    )

    # Tune only the two requested GRU variants.
    for model_name, mode, grid_key in [
        ("GRU_Random", "random", "gru_random"),
        ("GRU_Finetune", "finetune", "gru_finetune"),
    ]:
        best_trial, best_auc = None, -np.inf
        for trial in cfg["grids"][grid_key]:
            cr.set_seed(seed)
            model = cr._build_gru(
                cfg, len(vocab), trial.get("hidden", cfg["model"]["hidden_dim"]), device, mode
            )
            _, val_auc = cr.train_gru(
                model, train_loader, val_loader, pos_weight, trial["lr"],
                cfg["training"]["epochs"], cfg["training"]["early_stop"], "cls", device
            )
            print(f"  {model_name}: lr={trial['lr']} -> val ROC-AUC={val_auc:.4f}")
            if np.isfinite(val_auc) and val_auc > best_auc:
                best_auc, best_trial = val_auc, copy.deepcopy(trial)
        if best_trial is None:
            raise RuntimeError(f"No successful hyperparameter trial for {model_name}")
        best_hps[model_name] = best_trial

    # Tune MolBCAT only.
    train_ds = cr.CrossAttnDataset(X_tr, y_tr, vocab, cfg["model"]["max_len"])
    val_ds = cr.CrossAttnDataset(X_val, y_val, vocab, cfg["model"]["max_len"])
    train_ca = cr.DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True,
        collate_fn=cr.crossattn_collate, num_workers=cr._NW, pin_memory=cr._PIN
    )
    val_ca = cr.DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"],
        collate_fn=cr.crossattn_collate, num_workers=cr._NW, pin_memory=cr._PIN
    )
    best_trial, best_auc = None, -np.inf
    for trial in cfg["grids"]["molbcat"]:
        cr.set_seed(seed)
        model = cr.MolBCAT(
            len(vocab), cfg["model"]["emb_dim"], cfg["model"]["hidden_dim"],
            cfg["model"]["gin_hidden_dim"], cfg["model"]["node_feat_dim"],
            cfg["model"]["nhead"]
        ).to(device)
        model.pad_id = vocab.get("<PAD>", 0)
        cr.load_pretrained_encoder(model.gru_encoder, cfg["weights"]["pretrained_encoder"], device)
        _, val_auc, _, _ = cr.train_molbcat(
            model, train_ca, val_ca, pos_weight, trial["lr_encoder"], trial["lr_head"],
            cfg["training"].get("molbcat_epochs", cfg["training"]["epochs"]),
            cfg["training"]["early_stop"], "cls", device
        )
        print(f"  MolBCAT: lr_encoder={trial['lr_encoder']} lr_head={trial['lr_head']} -> val ROC-AUC={val_auc:.4f}")
        if np.isfinite(val_auc) and val_auc > best_auc:
            best_auc, best_trial = val_auc, copy.deepcopy(trial)
    if best_trial is None:
        raise RuntimeError("No successful hyperparameter trial for MolBCAT")
    best_hps["MolBCAT"] = best_trial

    # Fit selected models; select thresholds on validation only; test once.
    X_val_tensor = torch.tensor([cr.encode_smiles(x, vocab, cfg["model"]["max_len"]) for x in X_val])
    X_test_tensor = torch.tensor([cr.encode_smiles(x, vocab, cfg["model"]["max_len"]) for x in X_test])
    results = {}
    for model_name, mode in [("GRU_Random", "random"), ("GRU_Finetune", "finetune")]:
        hp = best_hps[model_name]
        cr.set_seed(seed)
        model = cr._build_gru(
            cfg, len(vocab), hp.get("hidden", cfg["model"]["hidden_dim"]), device, mode
        )
        model, _ = cr.train_gru(
            model, train_loader, val_loader, pos_weight, hp["lr"],
            cfg["training"]["epochs"], cfg["training"]["early_stop"], "cls", device
        )
        val_prob = cr.infer_gru(model, X_val_tensor, "cls", 256, device)
        threshold, val_f1 = cr.find_best_threshold(y_val, val_prob)
        test_prob = cr.infer_gru(model, X_test_tensor, "cls", 256, device)
        metrics = cr.classification_metrics(y_test, test_prob, threshold)
        results[model_name] = {**metrics, "threshold": threshold, "val_F1_at_threshold": val_f1}
        print(f"  {model_name}: threshold={threshold:.2f}, val_F1={val_f1:.4f}, test_F1={metrics['F1']:.4f}")

    # Rebuild MolBCAT loaders including untouched test set.
    test_ds = cr.CrossAttnDataset(X_test, y_test, vocab, cfg["model"]["max_len"])
    test_ca = cr.DataLoader(
        test_ds, batch_size=cfg["training"]["batch_size"],
        collate_fn=cr.crossattn_collate, num_workers=cr._NW, pin_memory=cr._PIN
    )
    hp = best_hps["MolBCAT"]
    cr.set_seed(seed)
    model = cr.MolBCAT(
        len(vocab), cfg["model"]["emb_dim"], cfg["model"]["hidden_dim"],
        cfg["model"]["gin_hidden_dim"], cfg["model"]["node_feat_dim"], cfg["model"]["nhead"]
    ).to(device)
    model.pad_id = vocab.get("<PAD>", 0)
    cr.load_pretrained_encoder(model.gru_encoder, cfg["weights"]["pretrained_encoder"], device)
    model, _, _, _ = cr.train_molbcat(
        model, train_ca, val_ca, pos_weight, hp["lr_encoder"], hp["lr_head"],
        cfg["training"].get("molbcat_epochs", cfg["training"]["epochs"]),
        cfg["training"]["early_stop"], "cls", device
    )
    val_prob, _ = cr.infer_molbcat(model, val_ca, "cls", device)
    threshold, val_f1 = cr.find_best_threshold(y_val, val_prob)
    test_prob, test_true = cr.infer_molbcat(model, test_ca, "cls", device)
    metrics = cr.classification_metrics(test_true, test_prob, threshold)
    results["MolBCAT"] = {**metrics, "threshold": threshold, "val_F1_at_threshold": val_f1}
    print(f"  MolBCAT: threshold={threshold:.2f}, val_F1={val_f1:.4f}, test_F1={metrics['F1']:.4f}")

    assert set(results) == set(REPORT_MODELS), f"Unexpected classification models: {list(results)}"
    return results


def _regression_one_setting(
    cfg, dataset_name, vocab, device, smiles, labels,
    train_idx, val_idx, test_idx, fold_number, ratio_out_dir
):

    rr = reg_runner
    X_train = [smiles[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    X_val = [smiles[i] for i in val_idx]
    y_val = [labels[i] for i in val_idx]
    X_test = [smiles[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]
    seed = cfg["split"]["random_seed"] + fold_number * 100

    # Tune only GRU_Random, GRU_Finetune and MolBCAT.
    best_hps = {}
    best_hps["GRU_Random"] = rr.tune_gru(
        cfg, X_train, y_train, X_val, y_val, vocab, device,
        "random", "gru_random", seed
    )
    best_hps["GRU_Finetune"] = rr.tune_gru(
        cfg, X_train, y_train, X_val, y_val, vocab, device,
        "finetune", "gru_finetune", seed + 20
    )
    best_hps["MolBCAT"] = rr.tune_molbcat(
        cfg, X_train, y_train, X_val, y_val, vocab, device, seed + 40
    )

    cfg_run = copy.deepcopy(cfg)
    cfg_run.setdefault("output", {})
    cfg_run["output"]["save_dir"] = os.path.join(ratio_out_dir, "model_artifacts")
    results = {}
    train_loader, val_loader = rr.make_gru_loaders(
        X_train, y_train, X_val, y_val, vocab,
        cfg["model"]["max_len"], cfg["training"]["batch_size"]
    )
    X_test_tensor = torch.tensor([
        rr.encode_smiles(x, vocab, cfg["model"]["max_len"]) for x in X_test
    ])
    for model_name, mode in [("GRU_Random", "random"), ("GRU_Finetune", "finetune")]:
        hp = best_hps[model_name]
        rr.set_seed(seed)
        model = rr.build_gru(
            cfg, len(vocab), hp.get("hidden", cfg["model"]["hidden_dim"]), device, mode
        )
        model, _ = rr.train_gru(
            model, train_loader, val_loader, None, hp["lr"],
            cfg["training"]["epochs"], cfg["training"]["early_stop"], "reg", device
        )
        pred = rr.infer_gru(model, X_test_tensor, "reg", 256, device)
        results[model_name] = rr.calc_reg_metrics(y_test, pred)

    # MolBCAT final fit and test evaluation.
    train_ds = rr.CrossAttnDataset(X_train, y_train, vocab, cfg["model"]["max_len"])
    val_ds = rr.CrossAttnDataset(X_val, y_val, vocab, cfg["model"]["max_len"])
    test_ds = rr.CrossAttnDataset(X_test, y_test, vocab, cfg["model"]["max_len"])
    def _loader(ds, shuffle=False):
        return rr.DataLoader(
            ds, batch_size=cfg["training"]["batch_size"], shuffle=shuffle,
            collate_fn=rr.crossattn_collate, num_workers=rr.NUM_WORKERS,
            pin_memory=rr.PIN_MEMORY
        )
    train_ca, val_ca, test_ca = _loader(train_ds, True), _loader(val_ds), _loader(test_ds)
    hp = best_hps["MolBCAT"]
    rr.set_seed(seed)
    model = rr.MolBCAT(
        len(vocab), cfg["model"]["emb_dim"], cfg["model"]["hidden_dim"],
        cfg["model"]["gin_hidden_dim"], cfg["model"]["node_feat_dim"], cfg["model"]["nhead"]
    ).to(device)
    model.pad_id = vocab.get("<PAD>", 0)
    rr.load_pretrained_encoder(model.gru_encoder, cfg["weights"]["pretrained_encoder"], device)
    model, _, train_curve, val_curve, train_r2_curve, val_r2_curve = rr.train_molbcat(
        model, train_ca, val_ca, None, hp["lr_encoder"], hp["lr_head"],
        cfg["training"]["epochs"], cfg["training"]["early_stop"], "reg", device
    )
    weight_cfg = copy.deepcopy(cfg_run)
    rr.save_molbcat_weights(model, weight_cfg, dataset_name, fold_number, hp)
    pred, true = rr.infer_molbcat(model, test_ca, "reg", device)
    results["MolBCAT"] = rr.calc_reg_metrics(true, pred)
    results["MolBCAT"].update({
        "train_rmse_curve": train_curve, "val_rmse_curve": val_curve,
        "train_r2_curve": train_r2_curve, "val_r2_curve": val_r2_curve,
        "test_y_true": np.asarray(true).tolist(), "test_y_pred": np.asarray(pred).tolist(),
    })
    assert set(results) == set(REPORT_MODELS), f"Unexpected regression models: {list(results)}"
    return results, best_hps

def _aggregate(fold_results, task):
    summary = {}
    metric = "ROC_AUC" if task == "cls" else "RMSE"
    for ratio in RATIOS:
        key = f"{int(round(ratio * 100))}%"
        summary[key] = {}
        for model in REPORT_MODELS:
            values = []
            for fold in range(1, 6):
                item = fold_results.get(str(fold), {}).get(key, {}).get(model, {})
                value = item.get(metric, float("nan"))
                if value is not None and np.isfinite(value):
                    values.append(float(value))
            summary[key][f"{model}_mean"] = (
                float(np.mean(values)) if values else float("nan")
            )
            summary[key][f"{model}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
            )
            summary[key][f"{model}_n_folds"] = len(values)
    return summary


def _plot_summaries(cls_summary, reg_summary, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = [int(round(r * 100)) for r in RATIOS]
    styles = {
        "GRU_Random": ("o--", "#4472C4"),
        "GRU_Finetune": ("s--", "#ED7D31"),
        "MolBCAT": ("D-", "#548235"),
    }

    for ax, summary, title, metric, lower_is_better in [
        (axes[0], cls_summary, "BBBP (higher is better)", "ROC_AUC", False),
        (axes[1], reg_summary, "Lipophilicity (lower is better)", "RMSE", True),
    ]:
        for model in REPORT_MODELS:
            means = [summary[f"{p}%"][f"{model}_mean"] for p in x]
            stds = [summary[f"{p}%"][f"{model}_std"] for p in x]
            stds = [0.0 if not np.isfinite(s) else s for s in stds]
            marker, color = styles[model]
            ax.plot(x, means, marker, color=color, label=model, linewidth=2)
            ax.fill_between(
                x,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                alpha=0.15,
                color=color,
            )
        ax.set_title(title)
        ax.set_xlabel("Training Data Size (%)")
        ax.set_ylabel("ROC-AUC" if metric == "ROC_AUC" else "RMSE")
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def _run_classification(cfg, vocab, device, out_dir, resume):
    all_cfgs = cfg["data"]["datasets"]
    dataset_cfg = next(d for d in all_cfgs if d["name"] == "BBBP")
    ds = load_dataset(dataset_cfg["hf"])["train"]
    smiles, labels = filter_valid(
        ds["SMILES"], ds[dataset_cfg["col"]], task="cls"
    )

    folds = scaffold_kfold_split(
        smiles,
        labels=labels,
        n_folds=cfg["split"]["n_folds"],
        seed=cfg["split"]["random_seed"],
    )
    result_path = os.path.join(out_dir, "BBBP_fold_results.json")
    fold_results = {}
    if resume and os.path.exists(result_path):
        with open(result_path, encoding="utf-8") as f:
            fold_results = json.load(f)

    for fold_id, test_idx in enumerate(folds):
        fold_number = fold_id + 1
        fold_key = str(fold_number)
        if resume and fold_key in fold_results and all(
            f"{int(r*100)}%" in fold_results[fold_key] for r in RATIOS
        ):
            print(f"[RESUME] BBBP fold {fold_number} already complete.")
            continue

        dev_idx = [i for j, fold in enumerate(folds) if j != fold_id for i in fold]
        train_idx, val_idx = scaffold_train_val_split(
            smiles,
            dev_idx,
            labels=labels,
            val_ratio=cfg["split"]["inner_val_ratio"],
            seed=cfg["split"]["random_seed"] + fold_id,
        )
        fold_results.setdefault(fold_key, {})

        for ratio in RATIOS:
            ratio_key = f"{int(round(ratio * 100))}%"
            if resume and ratio_key in fold_results[fold_key]:
                continue
            subset_idx = _stratified_subset(
                train_idx, labels, ratio,
                cfg["split"]["random_seed"] + fold_id * 1000 + int(ratio * 100),
                "cls",
            )
            overlap = _check_splits(
                smiles, subset_idx, val_idx, test_idx,
                "BBBP", fold_number, ratio_key
            )
            print(
                f"\nBBBP fold={fold_number}, ratio={ratio_key}: "
                f"train={len(subset_idx)}, val={len(val_idx)}, test={len(test_idx)}; "
                f"scaffold overlap={overlap}"
            )
            metrics = _classification_one_setting(
                dataset_cfg, cfg, vocab, device, smiles, labels,
                subset_idx, val_idx, test_idx, fold_id
            )
            fold_results[fold_key][ratio_key] = metrics
            _save_json(result_path, fold_results)

    summary = _aggregate(fold_results, "cls")
    _save_json(os.path.join(out_dir, "BBBP_summary.json"), summary)
    return fold_results, summary


def _run_regression(cfg, vocab, device, out_dir, resume):
    dataset_cfg = next(
        d for d in cfg["data"]["datasets"] if d["name"] == "Lipophilicity"
    )
    ds = load_dataset(dataset_cfg["hf"])["train"]
    smiles, labels = filter_valid(
        ds["SMILES"], ds[dataset_cfg["col"]], task="reg"
    )

    folds = scaffold_kfold_split_regression(
        smiles,
        labels=labels,
        n_folds=cfg["split"]["n_folds"],
        seed=cfg["split"]["random_seed"],
    )
    result_path = os.path.join(out_dir, "Lipophilicity_fold_results.json")
    fold_results = {}
    if resume and os.path.exists(result_path):
        with open(result_path, encoding="utf-8") as f:
            fold_results = json.load(f)

    for fold_id, test_idx in enumerate(folds):
        fold_number = fold_id + 1
        fold_key = str(fold_number)
        if resume and fold_key in fold_results and all(
            f"{int(r*100)}%" in fold_results[fold_key] for r in RATIOS
        ):
            print(f"[RESUME] Lipophilicity fold {fold_number} already complete.")
            continue

        dev_idx = [i for j, fold in enumerate(folds) if j != fold_id for i in fold]
        train_idx, val_idx = scaffold_train_val_split_regression(
            smiles,
            dev_idx,
            val_ratio=cfg["split"]["inner_val_ratio"],
            seed=cfg["split"]["random_seed"] + fold_id,
        )
        fold_results.setdefault(fold_key, {})

        for ratio in RATIOS:
            ratio_key = f"{int(round(ratio * 100))}%"
            if resume and ratio_key in fold_results[fold_key]:
                continue
            subset_idx = _stratified_subset(
                train_idx, labels, ratio,
                cfg["split"]["random_seed"] + fold_id * 1000 + int(ratio * 100),
                "reg",
            )
            overlap = _check_splits(
                smiles, subset_idx, val_idx, test_idx,
                "Lipophilicity", fold_number, ratio_key
            )
            print(
                f"\nLipophilicity fold={fold_number}, ratio={ratio_key}: "
                f"train={len(subset_idx)}, val={len(val_idx)}, test={len(test_idx)}; "
                f"scaffold overlap={overlap}"
            )
            ratio_dir = os.path.join(out_dir, "Lipophilicity", f"fold_{fold_number}", ratio_key)
            metrics, hps = _regression_one_setting(
                cfg, "Lipophilicity", vocab, device, smiles, labels,
                subset_idx, val_idx, test_idx, fold_number, ratio_dir
            )
            fold_results[fold_key][ratio_key] = {
                "metrics": metrics,
                "hyperparameters": hps,
            }
            _save_json(result_path, fold_results)

    # Normalize regression result structure to match aggregate helper.
    normalized = {}
    for fold, ratios in fold_results.items():
        normalized[fold] = {
            ratio: (
                value["metrics"] if "metrics" in value else value
            )
            for ratio, value in ratios.items()
        }
    summary = _aggregate(normalized, "reg")
    _save_json(os.path.join(out_dir, "Lipophilicity_summary.json"), summary)
    return fold_results, summary


def _summary_csv(cls_summary, reg_summary, out_dir):
    rows = []
    for ratio in [f"{int(round(r * 100))}%" for r in RATIOS]:
        row = {"training_fraction": ratio}
        for model in REPORT_MODELS:
            for dataset, summary in [("BBBP", cls_summary), ("Lipophilicity", reg_summary)]:
                row[f"{dataset}_{model}_mean"] = summary[ratio][f"{model}_mean"]
                row[f"{dataset}_{model}_sd"] = summary[ratio][f"{model}_std"]
                row[f"{dataset}_{model}_n_folds"] = summary[ratio][f"{model}_n_folds"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "data_efficiency_5fold_summary.csv"), index=False
    )


def main(args=None):
    if args is None:
        args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.cls_config, encoding="utf-8") as f:
        cls_cfg = yaml.safe_load(f)
    with open(args.reg_config, encoding="utf-8") as f:
        reg_cfg = yaml.safe_load(f)

    # Apply the dedicated schedule defined in each YAML data_efficiency block
    # to both hyperparameter tuning and final model fitting.
    for cfg in (cls_cfg, reg_cfg):
        de_cfg = cfg.get("data_efficiency", {})
        cfg.setdefault("training", {})
        if "epochs" in de_cfg:
            cfg["training"]["epochs"] = de_cfg["epochs"]
        if "early_stop" in de_cfg:
            cfg["training"]["early_stop"] = de_cfg["early_stop"]

    device = get_device()
    vocab = load_vocab(cls_cfg["weights"]["vocab"])

    if cls_cfg["split"]["n_folds"] != 5 or reg_cfg["split"]["n_folds"] != 5:
        raise ValueError("This reviewer-aligned script expects n_folds: 5.")

    cls_results, cls_summary = _run_classification(
        cls_cfg, vocab, device, args.out_dir, args.resume
    )
    reg_results, reg_summary = _run_regression(
        reg_cfg, vocab, device, args.out_dir, args.resume
    )

    _summary_csv(cls_summary, reg_summary, args.out_dir)
    _plot_summaries(
        cls_summary,
        reg_summary,
        os.path.join(args.out_dir, "Figure_S3_data_efficiency_5fold.png"),
    )
    print("\nCompleted scaffold-disjoint 5-fold data-efficiency analysis.")
    print(f"Outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
