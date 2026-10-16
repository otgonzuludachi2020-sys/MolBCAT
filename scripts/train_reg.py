"""
Regression experiment runner for MolBCAT.

Protocol
--------
5-fold outer scaffold cross-validation
+ scaffold-disjoint inner train/validation split
+ fold-specific hyperparameter tuning
+ untouched test fold evaluation

Models
------
GRU_Random
GRU_Frozen
GRU_Finetune
GIN
MolBCAT
ChemBERTa

Metrics
-------
RMSE, MAE, R2

Usage
-----
python scripts/train_reg.py --config configs/regression.yaml
python scripts/train_reg.py --config configs/regression.yaml --dataset ESOL
"""

import os
import sys
import copy
import argparse

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt

from src.checkpoint import fold_checkpoint_path, load_checkpoint, save_checkpoint

from datasets import load_dataset
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeoLoader

from scipy.stats import ttest_rel


sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


from src.utils import (
    set_seed,
    get_device,
    load_vocab,
    save_json,
    load_pretrained_encoder,
)

from src.evaluation import (
    calc_reg_metrics,
    cohens_d_paired,
)

from src.dataset import (
    filter_valid,
    encode_smiles,
    make_gru_loaders,
    scaffold_kfold_split_regression,
    scaffold_train_val_split_regression,
    check_scaffold_disjoint,
    build_graph_list,
    CrossAttnDataset,
    crossattn_collate,
    ChemBERTaDataset,
    make_chemberta_loaders,
)

from src.models import (
    GRUModel,
    GINModel,
    MolBCAT,
    ChemBERTaModel,
)

from src.trainer import (
    train_gru,
    train_molbcat,
    train_chemberta,
    infer_gru,
    infer_gin,
    infer_molbcat,
    infer_chemberta,
)


PIN_MEMORY = torch.cuda.is_available()
NUM_WORKERS = 2


def save_molbcat_weights(model, cfg, dataset_name, fold, hyperparameters):
    """Save the best-validation MolBCAT weights and reproducibility metadata."""
    weight_dir = os.path.join(
        cfg["output"]["save_dir"],
        "weights",
        dataset_name,
    )
    os.makedirs(weight_dir, exist_ok=True)

    path = os.path.join(
        weight_dir,
        f"fold_{fold}_MolBCAT.pt",
    )

    torch.save(
        {
            "dataset": dataset_name,
            "fold": fold,
            "model_state_dict": model.state_dict(),
            "hyperparameters": hyperparameters,
        },
        path,
    )
    return path


def parse_args():

    parser = argparse.ArgumentParser(
        description="5-fold scaffold CV regression"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/regression.yaml",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help='Dataset name or "all"',
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed outer folds from fold checkpoints.",
    )

    return parser.parse_args()


def build_gru(
    cfg,
    vocab_size,
    hidden_dim,
    device,
    mode,
):

    model = GRUModel(
        vocab_size,
        cfg["model"]["emb_dim"],
        hidden_dim,
    ).to(device)

    if mode in ["frozen", "finetune"]:

        load_pretrained_encoder(
            model,
            cfg["weights"]["pretrained_encoder"],
            device,
        )

    if mode == "frozen":

        for p in model.emb.parameters():
            p.requires_grad = False

        for p in model.gru.parameters():
            p.requires_grad = False

    return model


def tune_gru(
    cfg,
    X_train,
    y_train,
    X_val,
    y_val,
    vocab,
    device,
    mode,
    grid_key,
    seed,
):

    train_loader, val_loader = make_gru_loaders(
        X_train,
        y_train,
        X_val,
        y_val,
        vocab,
        cfg["model"]["max_len"],
        cfg["training"]["batch_size"],
    )

    best_hp = None
    best_rmse = float("inf")

    for i, trial in enumerate(cfg["grids"][grid_key]):

        try:

            set_seed(seed + i)

            model = build_gru(
                cfg,
                len(vocab),
                trial.get(
                    "hidden",
                    cfg["model"]["hidden_dim"],
                ),
                device,
                mode,
            )

            _, val_rmse = train_gru(
                model,
                train_loader,
                val_loader,
                None,
                trial["lr"],
                cfg["training"]["epochs"],
                cfg["training"]["early_stop"],
                "reg",
                device,
            )

            print(
                f"      {mode:<10} "
                f"lr={trial['lr']} "
                f"RMSE={val_rmse:.5f}"
            )

            if np.isfinite(val_rmse) and val_rmse < best_rmse:

                best_rmse = val_rmse
                best_hp = copy.deepcopy(trial)

        except Exception as e:

            print(
                f"      {mode} trial failed: {e}"
            )

    if best_hp is None:

        raise RuntimeError(
            f"No valid HP found for {mode}"
        )

    return best_hp


def tune_gin(
    cfg,
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    seed,
):

    train_graphs = build_graph_list(
        X_train,
        y_train,
    )

    val_graphs = build_graph_list(
        X_val,
        y_val,
    )

    train_loader = GeoLoader(
        train_graphs,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
    )

    val_loader = GeoLoader(
        val_graphs,
        batch_size=cfg["training"]["batch_size"],
    )

    best_hp = None
    best_rmse = float("inf")

    for i, trial in enumerate(
        cfg["grids"]["gin"]
    ):

        try:

            set_seed(seed + i)

            model = GINModel(
                cfg["model"]["node_feat_dim"],
                cfg["model"]["gin_hidden_dim"],
            ).to(device)

            _, val_rmse = train_gru(
                model,
                train_loader,
                val_loader,
                None,
                trial["lr"],
                cfg["training"]["epochs"],
                cfg["training"]["early_stop"],
                "reg",
                device,
            )

            print(
                f"      GIN        "
                f"lr={trial['lr']} "
                f"RMSE={val_rmse:.5f}"
            )

            if np.isfinite(val_rmse) and val_rmse < best_rmse:

                best_rmse = val_rmse
                best_hp = copy.deepcopy(trial)

        except Exception as e:

            print(
                f"      GIN trial failed: {e}"
            )

    if best_hp is None:
        raise RuntimeError(
            "No valid HP found for GIN"
        )

    return best_hp

def tune_molbcat(
    cfg,
    X_train,
    y_train,
    X_val,
    y_val,
    vocab,
    device,
    seed,
):

    train_ds = CrossAttnDataset(
        X_train,
        y_train,
        vocab,
        cfg["model"]["max_len"],
    )

    val_ds = CrossAttnDataset(
        X_val,
        y_val,
        vocab,
        cfg["model"]["max_len"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=crossattn_collate,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        collate_fn=crossattn_collate,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    best_hp = None
    best_rmse = float("inf")

    for i, trial in enumerate(
        cfg["grids"]["molbcat"]
    ):

        try:

            set_seed(seed + i)

            model = MolBCAT(
                len(vocab),
                cfg["model"]["emb_dim"],
                cfg["model"]["hidden_dim"],
                cfg["model"]["gin_hidden_dim"],
                cfg["model"]["node_feat_dim"],
                cfg["model"]["nhead"],
            ).to(device)

            model.pad_id = vocab.get(
                "<PAD>",
                0,
            )

            load_pretrained_encoder(
                model.gru_encoder,
                cfg["weights"]["pretrained_encoder"],
                device,
            )

            tune_out = train_molbcat(
                model,
                train_loader,
                val_loader,
                None,
                trial["lr_encoder"],
                trial["lr_head"],
                cfg["training"]["epochs"],
                cfg["training"]["early_stop"],
                "reg",
                device,
            )
            val_rmse = tune_out[1]

            print(
                f"      MolBCAT    "
                f"lr_enc={trial['lr_encoder']} "
                f"lr_head={trial['lr_head']} "
                f"RMSE={val_rmse:.5f}"
            )

            if np.isfinite(val_rmse) and val_rmse < best_rmse:

                best_rmse = val_rmse
                best_hp = copy.deepcopy(trial)

        except Exception as e:

            print(
                f"      MolBCAT trial failed: {e}"
            )

    if best_hp is None:
        raise RuntimeError(
            "No valid HP found for MolBCAT"
        )

    return best_hp

def tune_chemberta(
    cfg,
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    seed,
):

    train_loader, val_loader = make_chemberta_loaders(
        X_train,
        y_train,
        X_val,
        y_val,
        cfg["training"]["batch_size"],
    )

    best_hp = None
    best_rmse = float("inf")

    for i, trial in enumerate(
        cfg["grids"]["chemberta"]
    ):

        try:

            set_seed(seed + i)

            model = ChemBERTaModel().to(device)

            lr = trial["lr"]

            _, val_rmse, _, _ = train_chemberta(
                model,
                train_loader,
                val_loader,
                None,
                lr,
                lr * 10,
                cfg["training"].get(
                    "chemberta_epochs",
                    cfg["training"]["epochs"],
                ),
                cfg["training"]["early_stop"],
                "reg",
                device,
            )

            print(
                f"      ChemBERTa  "
                f"lr={lr} "
                f"RMSE={val_rmse:.5f}"
            )

            if np.isfinite(val_rmse) and val_rmse < best_rmse:

                best_rmse = val_rmse
                best_hp = copy.deepcopy(trial)

        except Exception as e:

            print(
                f"      ChemBERTa trial failed: {e}"
            )

    if best_hp is None:
        raise RuntimeError(
            "No valid HP found for ChemBERTa"
        )

    return best_hp


def tune_all_models(
    cfg,
    X_train,
    y_train,
    X_val,
    y_val,
    vocab,
    device,
    fold,
):

    base_seed = (
        cfg["split"]["random_seed"]
        + fold * 100
    )

    print(
        f"\n    Hyperparameter tuning — Fold {fold}"
    )

    hps = {}

    hps["GRU_Random"] = tune_gru(
        cfg,
        X_train,
        y_train,
        X_val,
        y_val,
        vocab,
        device,
        "random",
        "gru_random",
        base_seed,
    )

    hps["GRU_Frozen"] = tune_gru(
        cfg,
        X_train,
        y_train,
        X_val,
        y_val,
        vocab,
        device,
        "frozen",
        "gru_frozen",
        base_seed + 10,
    )

    hps["GRU_Finetune"] = tune_gru(
        cfg,
        X_train,
        y_train,
        X_val,
        y_val,
        vocab,
        device,
        "finetune",
        "gru_finetune",
        base_seed + 20,
    )

    hps["GIN"] = tune_gin(
        cfg,
        X_train,
        y_train,
        X_val,
        y_val,
        device,
        base_seed + 30,
    )

    hps["MolBCAT"] = tune_molbcat(
        cfg,
        X_train,
        y_train,
        X_val,
        y_val,
        vocab,
        device,
        base_seed + 40,
    )

    hps["ChemBERTa"] = tune_chemberta(
        cfg,
        X_train,
        y_train,
        X_val,
        y_val,
        device,
        base_seed + 50,
    )

    return hps


def run_fold(
    cfg,
    dataset_name,
    vocab,
    device,
    smiles,
    labels,
    train_idx,
    val_idx,
    test_idx,
    fold,
):

    X_train = [smiles[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]

    X_val = [smiles[i] for i in val_idx]
    y_val = [labels[i] for i in val_idx]

    X_test = [smiles[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]

    print(
        f"\n{'=' * 70}"
    )

    print(
        f"Fold {fold} — "
        f"Train={len(X_train)}, "
        f"Val={len(X_val)}, "
        f"Test={len(X_test)}"
    )

    print(
        f"{'=' * 70}"
    )


    hps = tune_all_models(
        cfg,
        X_train,
        y_train,
        X_val,
        y_val,
        vocab,
        device,
        fold,
    )


    results = {}

    for model_name, mode in [
        ("GRU_Random", "random"),
        ("GRU_Frozen", "frozen"),
        ("GRU_Finetune", "finetune"),
    ]:

        try:

            set_seed(
                cfg["split"]["random_seed"]
                + fold * 100
            )

            hp = hps[model_name]

            train_loader, val_loader = make_gru_loaders(
                X_train,
                y_train,
                X_val,
                y_val,
                vocab,
                cfg["model"]["max_len"],
                cfg["training"]["batch_size"],
            )

            model = build_gru(
                cfg,
                len(vocab),
                hp.get(
                    "hidden",
                    cfg["model"]["hidden_dim"],
                ),
                device,
                mode,
            )

            model, _ = train_gru(
                model,
                train_loader,
                val_loader,
                None,
                hp["lr"],
                cfg["training"]["epochs"],
                cfg["training"]["early_stop"],
                "reg",
                device,
            )

            test_x = torch.tensor(
                [
                    encode_smiles(
                        s,
                        vocab,
                        cfg["model"]["max_len"],
                    )
                    for s in X_test
                ]
            )

            pred = infer_gru(
                model,
                test_x,
                "reg",
                256,
                device,
            )

            results[model_name] = calc_reg_metrics(
                y_test,
                pred,
            )

        except Exception as e:

            print(
                f"    {model_name} failed: {e}"
            )

            results[model_name] = {
                "RMSE": np.nan,
                "MAE": np.nan,
                "R2": np.nan,
            }


    try:

        set_seed(
            cfg["split"]["random_seed"]
            + fold * 100
        )

        train_g = build_graph_list(
            X_train,
            y_train,
        )

        val_g = build_graph_list(
            X_val,
            y_val,
        )

        test_g = build_graph_list(
            X_test,
            y_test,
        )

        train_loader = GeoLoader(
            train_g,
            batch_size=cfg["training"]["batch_size"],
            shuffle=True,
        )

        val_loader = GeoLoader(
            val_g,
            batch_size=cfg["training"]["batch_size"],
        )

        model = GINModel(
            cfg["model"]["node_feat_dim"],
            cfg["model"]["gin_hidden_dim"],
        ).to(device)

        model, _ = train_gru(
            model,
            train_loader,
            val_loader,
            None,
            hps["GIN"]["lr"],
            cfg["training"]["epochs"],
            cfg["training"]["early_stop"],
            "reg",
            device,
        )

        pred, true = infer_gin(
            model,
            test_g,
            "reg",
            32,
            device,
        )

        results["GIN"] = calc_reg_metrics(
            true,
            pred,
        )

    except Exception as e:

        print(
            f"    GIN failed: {e}"
        )

        results["GIN"] = {
            "RMSE": np.nan,
            "MAE": np.nan,
            "R2": np.nan,
        }


    try:

        set_seed(
            cfg["split"]["random_seed"]
            + fold * 100
        )

        train_ds = CrossAttnDataset(
            X_train,
            y_train,
            vocab,
            cfg["model"]["max_len"],
        )

        val_ds = CrossAttnDataset(
            X_val,
            y_val,
            vocab,
            cfg["model"]["max_len"],
        )

        test_ds = CrossAttnDataset(
            X_test,
            y_test,
            vocab,
            cfg["model"]["max_len"],
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=True,
            collate_fn=crossattn_collate,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["training"]["batch_size"],
            collate_fn=crossattn_collate,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=cfg["training"]["batch_size"],
            collate_fn=crossattn_collate,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

        model = MolBCAT(
            len(vocab),
            cfg["model"]["emb_dim"],
            cfg["model"]["hidden_dim"],
            cfg["model"]["gin_hidden_dim"],
            cfg["model"]["node_feat_dim"],
            cfg["model"]["nhead"],
        ).to(device)

        model.pad_id = vocab.get(
            "<PAD>",
            0,
        )

        load_pretrained_encoder(
            model.gru_encoder,
            cfg["weights"]["pretrained_encoder"],
            device,
        )

        (
            model,
            _,
            train_curve,
            val_curve,
            train_r2_curve,
            val_r2_curve,
        ) = train_molbcat(
            model,
            train_loader,
            val_loader,
            None,
            hps["MolBCAT"]["lr_encoder"],
            hps["MolBCAT"]["lr_head"],
            cfg["training"]["epochs"],
            cfg["training"]["early_stop"],
            "reg",
            device,
        )

        molbcat_weight_path = save_molbcat_weights(
            model,
            cfg,
            dataset_name,
            fold,
            hps["MolBCAT"],
        )

        pred, true = infer_molbcat(
            model,
            test_loader,
            "reg",
            device,
        )

        results["MolBCAT"] = calc_reg_metrics(
            true,
            pred,
        )

        # Save MolBCAT training/validation histories and untouched outer-test
        # predictions for reproducible plots.
        results["MolBCAT"]["train_rmse_curve"] = train_curve
        results["MolBCAT"]["val_rmse_curve"] = val_curve
        results["MolBCAT"]["train_r2_curve"] = train_r2_curve
        results["MolBCAT"]["val_r2_curve"] = val_r2_curve
        results["MolBCAT"]["test_y_true"] = np.asarray(true).tolist()
        results["MolBCAT"]["test_y_pred"] = np.asarray(pred).tolist()
        results["MolBCAT"]["weights_path"] = molbcat_weight_path

    except Exception as e:

        print(
            f"    MolBCAT failed: {e}"
        )

        results["MolBCAT"] = {
            "RMSE": np.nan,
            "MAE": np.nan,
            "R2": np.nan,
        }



    try:

        set_seed(
            cfg["split"]["random_seed"]
            + fold * 100
        )

        train_loader, val_loader = make_chemberta_loaders(
            X_train,
            y_train,
            X_val,
            y_val,
            cfg["training"]["batch_size"],
        )

        test_loader = DataLoader(
            ChemBERTaDataset(
                X_test,
                y_test,
            ),
            batch_size=cfg["training"]["batch_size"],
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

        model = ChemBERTaModel().to(device)

        lr = hps["ChemBERTa"]["lr"]

        model, _, _, _ = train_chemberta(
            model,
            train_loader,
            val_loader,
            None,
            lr,
            lr * 10,
            cfg["training"].get(
                "chemberta_epochs",
                cfg["training"]["epochs"],
            ),
            cfg["training"]["early_stop"],
            "reg",
            device,
        )

        pred, true = infer_chemberta(
            model,
            test_loader,
            "reg",
            device,
        )

        results["ChemBERTa"] = calc_reg_metrics(
            true,
            pred,
        )

    except Exception as e:

        print(
            f"    ChemBERTa failed: {e}"
        )

        results["ChemBERTa"] = {
            "RMSE": np.nan,
            "MAE": np.nan,
            "R2": np.nan,
        }


    for model_name, metrics in results.items():

        print(
            f"    {model_name:<15} "
            f"RMSE={metrics['RMSE']:.4f} "
            f"MAE={metrics['MAE']:.4f} "
            f"R2={metrics['R2']:.4f}"
        )

    return {
        "fold": fold,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "hyperparameters": hps,
        "results": results,
    }



def _pad_curves(curves):
    """Pad variable-length epoch curves with NaN for fold-wise averaging."""
    if not curves:
        return np.empty((0, 0))
    max_len = max(len(c) for c in curves)
    arr = np.full((len(curves), max_len), np.nan, dtype=float)
    for i, curve in enumerate(curves):
        if curve:
            arr[i, :len(curve)] = np.asarray(curve, dtype=float)
    return arr


def plot_molbcat_training_curves(fold_results, dataset_name, output_dir):
    """Plot mean ± sample SD training/validation RMSE and R² across outer folds."""
    curves = [
        fold.get("results", {}).get("MolBCAT", {})
        for fold in fold_results
    ]

    rmse_train = [c.get("train_rmse_curve", []) for c in curves]
    rmse_val = [c.get("val_rmse_curve", []) for c in curves]
    r2_train = [c.get("train_r2_curve", []) for c in curves]
    r2_val = [c.get("val_r2_curve", []) for c in curves]

    def _plot(train_curves, val_curves, ylabel, filename):
        tr = _pad_curves(train_curves)
        va = _pad_curves(val_curves)
        if tr.size == 0 or va.size == 0:
            return

        epochs = np.arange(1, max(tr.shape[1], va.shape[1]) + 1)

        tr_mean = np.nanmean(tr, axis=0)
        va_mean = np.nanmean(va, axis=0)
        tr_sd = np.nanstd(tr, axis=0, ddof=1)
        va_sd = np.nanstd(va, axis=0, ddof=1)

        plt.figure(figsize=(7, 5))
        x_tr = np.arange(1, tr.shape[1] + 1)
        x_va = np.arange(1, va.shape[1] + 1)

        plt.plot(x_tr, tr_mean, label="Train")
        plt.fill_between(x_tr, tr_mean - tr_sd, tr_mean + tr_sd, alpha=0.2)

        plt.plot(x_va, va_mean, label="Validation")
        plt.fill_between(x_va, va_mean - va_sd, va_mean + va_sd, alpha=0.2)

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"MolBCAT {ylabel} — {dataset_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()

    _plot(
        rmse_train,
        rmse_val,
        "RMSE",
        f"{dataset_name}_MolBCAT_RMSE_curve.png",
    )
    _plot(
        r2_train,
        r2_val,
        "R²",
        f"{dataset_name}_MolBCAT_R2_curve.png",
    )


def plot_molbcat_predicted_vs_measured(fold_results, dataset_name, output_dir):
    """Pool untouched outer-test predictions from all five folds."""
    y_true = []
    y_pred = []

    for fold in fold_results:
        molbcat = fold.get("results", {}).get("MolBCAT", {})
        y_true.extend(molbcat.get("test_y_true", []))
        y_pred.extend(molbcat.get("test_y_pred", []))

    if not y_true:
        return

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())

    plt.figure(figsize=(6, 6))
    plt.scatter(y_pred, y_true, alpha=0.55)
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)

    plt.xlabel("Predicted")
    plt.ylabel("Measured")
    plt.title(f"MolBCAT Predicted vs Measured — {dataset_name}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            f"{dataset_name}_MolBCAT_predicted_vs_measured.png",
        ),
        dpi=300,
    )
    plt.close()


def summarize(
    fold_results,
    dataset_name,
):

    models = [
        "GRU_Random",
        "GRU_Frozen",
        "GRU_Finetune",
        "GIN",
        "MolBCAT",
        "ChemBERTa",
    ]

    metrics = [
        "RMSE",
        "MAE",
        "R2",
    ]

    summary = {
        "Dataset": dataset_name
    }

    for model in models:

        for metric in metrics:

            values = []

            for fold in fold_results:

                value = fold["results"].get(
                    model,
                    {},
                ).get(
                    metric,
                    np.nan,
                )

                if np.isfinite(value):
                    values.append(float(value))

            summary[
                f"{model}_{metric}_mean"
            ] = (
                float(np.mean(values))
                if values
                else np.nan
            )

            summary[
                f"{model}_{metric}_std"
            ] = (
                float(np.std(values, ddof=1))
                if len(values) >= 2
                else np.nan
            )

    for metric in metrics:

        molbcat = []
        chemberta = []

        for fold in fold_results:

            a = fold["results"].get(
                "MolBCAT",
                {},
            ).get(
                metric,
                np.nan,
            )

            b = fold["results"].get(
                "ChemBERTa",
                {},
            ).get(
                metric,
                np.nan,
            )

            if np.isfinite(a) and np.isfinite(b):

                molbcat.append(float(a))
                chemberta.append(float(b))

        if len(molbcat) >= 2:

            summary[
                f"p_MolBCAT_vs_ChemBERTa_{metric}"
            ] = float(
                ttest_rel(
                    molbcat,
                    chemberta,
                ).pvalue
            )

            summary[
                f"d_MolBCAT_vs_ChemBERTa_{metric}"
            ] = float(
                cohens_d_paired(
                    molbcat,
                    chemberta,
                )
            )

    return summary

def main(args=None):
    if args is None: 
        args = parse_args()

    with open(
        args.config,
        "r",
        encoding="utf-8",
    ) as f:

        cfg = yaml.safe_load(f)

    device = get_device()

    print(
        f"Device: {device}"
    )

    vocab = load_vocab(
        cfg["weights"]["vocab"]
    )

    

    if args.dataset == "all":

        datasets = cfg["data"]["datasets"]

    else:

        datasets = [
            d
            for d in cfg["data"]["datasets"]
            if d["name"] == args.dataset
        ]

    if not datasets:

        raise ValueError(
            f"Dataset '{args.dataset}' not found."
        )



    all_summaries = []

    for dataset_cfg in datasets:

        name = dataset_cfg["name"]

        print(
            f"\n\n{'#' * 70}"
        )

        print(
            f"# DATASET: {name}"
        )

        print(
            f"{'#' * 70}"
        )

        ds = load_dataset(
            dataset_cfg["hf"]
        )["train"]

        smiles, labels = filter_valid(
            ds["SMILES"],
            ds[dataset_cfg["col"]],
            task="reg",
        )

        print(
            f"Valid molecules: {len(smiles)}"
        )

       
        folds = scaffold_kfold_split_regression(
            smiles,
            labels=labels,
            n_folds=cfg["split"]["n_folds"],
            seed=cfg["split"]["random_seed"],
        )

        fold_results = []

        result_dir = os.path.join(
            cfg["output"]["save_dir"],
            "five_fold",
            name,
        )
        os.makedirs(result_dir, exist_ok=True)

        if args.resume:
            for existing_fold in range(1, cfg["split"]["n_folds"] + 1):
                ckpt_path = fold_checkpoint_path(result_dir, existing_fold)
                cached = load_checkpoint(ckpt_path)
                if cached is not None:
                    fold_results.append(cached)
                    print(f"[RESUME] {name} Fold {existing_fold}: completed checkpoint loaded.")

        completed_folds = {
            int(item["fold"])
            for item in fold_results
            if isinstance(item, dict) and "fold" in item
        }

        for fold_id in range(
            cfg["split"]["n_folds"]
        ):

            fold_number = fold_id + 1

            if args.resume and fold_number in completed_folds:
                print(f"[RESUME] {name} Fold {fold_number}: SKIP (already completed).")
                continue

            test_idx = folds[fold_id]

            test_set = set(test_idx)

            development_idx = [
                i
                for i in range(len(smiles))
                if i not in test_set
            ]


            train_idx, val_idx = scaffold_train_val_split_regression(
                smiles,
                development_idx,
                val_ratio=cfg["split"]["inner_val_ratio"],
                seed=(
                    cfg["split"]["random_seed"]
                    + fold_id
                ),
            )

            assert not (
                set(train_idx)
                & set(val_idx)
            )

            assert not (
                set(train_idx)
                & set(test_idx)
            )

            assert not (
                set(val_idx)
                & set(test_idx)
            )

            overlap = check_scaffold_disjoint(
                smiles,
                train_idx,
                val_idx,
                test_idx,
            )

            print(
                f"    Scaffold overlap: "
                f"train-val={overlap['train_val_overlap']}, "
                f"train-test={overlap['train_test_overlap']}, "
                f"val-test={overlap['val_test_overlap']}"
            )

            if any(value != 0 for value in overlap.values()):
                raise RuntimeError(
                    f"Scaffold leakage detected in {name}, "
                    f"Fold {fold_id + 1}: {overlap}"
                )

            result = run_fold(
                cfg,
                name,
                vocab,
                device,
                smiles,
                labels,
                train_idx,
                val_idx,
                test_idx,
                fold_id + 1,
            )

            fold_results.append(
                result
            )

            fold_ckpt = fold_checkpoint_path(
                result_dir,
                fold_number,
            )
            save_checkpoint(
                fold_ckpt,
                result,
            )
            completed_folds.add(fold_number)

            print(
                f"[CHECKPOINT] {name} Fold {fold_number} saved: "
                f"{fold_ckpt}"
            )

        if len(fold_results) != cfg["split"]["n_folds"]:
            raise RuntimeError(
                 f"Expected {cfg['split']['n_folds']} completed folds, "
                 f"but found {len(fold_results)}."
            )
            
        save_checkpoint(
            os.path.join(result_dir, "fold_results.json"),
            fold_results,
        )

        # Reproducible figures from the current 5-fold scaffold-CV run.
        plot_molbcat_training_curves(
            fold_results,
            name,
            result_dir,
        )
        plot_molbcat_predicted_vs_measured(
            fold_results,
            name,
            result_dir,
        )

        summary = summarize(
            fold_results,
            name,
        )

        save_json(
            os.path.join(
                result_dir,
                "summary.json",
            ),
            summary,
        )

        all_summaries.append(
            summary
        )


    df = pd.DataFrame(
        all_summaries
    )

    output_dir = cfg["output"]["save_dir"]

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    csv_path = os.path.join(
        output_dir,
        "regression_5fold_summary.csv",
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    print(
        f"\nFinal summary saved to:"
        f"\n{csv_path}"
    )


if __name__ == "__main__":
    main()