"""
Classification experiment runner.

Protocol
--------
Outer:
    5-fold scaffold-disjoint cross-validation

Inner:
    Scaffold-disjoint + class-aware train/validation split

For each outer fold:
    1. Split dataset into scaffold-disjoint TEST and DEVELOPMENT.
    2. Split DEVELOPMENT into scaffold-disjoint TRAIN / VALIDATION.
    3. Tune hyperparameters using TRAIN / VALIDATION only.
    4. Select Best HP based on validation ROC-AUC.
    5. Retrain a fresh model using the selected Best HP on TRAIN.
    6. Obtain validation probabilities from the Best-HP model.
    7. Select classification threshold on VALIDATION by maximizing F1.
    8. Freeze the threshold.
    9. Evaluate once on untouched TEST.
   10. Save fold-level results.

"""

import os
import sys
import copy
import argparse

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import numpy as np
import torch
import pandas as pd
import yaml

from src.checkpoint import fold_checkpoint_path, load_checkpoint, save_checkpoint

from datasets import load_dataset

from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeoLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

_PIN = torch.cuda.is_available()
_NW = 2


from src.utils import (
    set_seed,
    get_device,
    load_vocab,
    load_json,
    save_json,
    load_pretrained_encoder,
)

from src.evaluation import (
    make_pos_weight,
    cohens_d_paired,
)

from src.dataset import (
    filter_valid,
    encode_smiles,
    make_gru_loaders,
    scaffold_kfold_split,
    scaffold_train_val_split,
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


def parse_args():

    parser = argparse.ArgumentParser(
        description="Run 5-fold scaffold-disjoint classification experiments"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/classification.yaml",
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

def _build_gru(
    cfg,
    vocab_size,
    hidden,
    device,
    mode,
):

    model = GRUModel(
        vocab_size,
        cfg["model"]["emb_dim"],
        hidden,
    ).to(device)

    if mode in ("frozen", "finetune"):
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


def classification_metrics(
    y_true,
    y_prob,
    threshold,
):

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(np.unique(y_true)) < 2:

        return {
            "ROC_AUC": float("nan"),
            "PR_AUC": float("nan"),
            "Precision": float("nan"),
            "Recall": float("nan"),
            "F1": float("nan"),
            "MCC": float("nan"),
        }

    roc_auc = roc_auc_score(
        y_true,
        y_prob,
    )

    pr_auc = average_precision_score(
        y_true,
        y_prob,
    )

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred,
    )

    return {
        "ROC_AUC": float(roc_auc),
        "PR_AUC": float(pr_auc),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "MCC": float(mcc),
    }


def find_best_threshold(
    y_true,
    y_prob,
    min_threshold=0.05,
    max_threshold=0.95,
    step=0.01,
):

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    thresholds = np.arange(
        min_threshold,
        max_threshold + step / 2,
        step,
    )

    best_threshold = None
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = (
            y_prob >= threshold
        ).astype(int)
        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return (
        float(best_threshold),
        float(best_f1),
    )


def nan_metrics():

    return {
        "ROC_AUC": float("nan"),
        "PR_AUC": float("nan"),
        "Precision": float("nan"),
        "Recall": float("nan"),
        "F1": float("nan"),
        "MCC": float("nan"),
    }


def tune_hyperparameters(
    dataset_cfg,
    cfg,
    vocab,
    device,
    X_tr,
    y_tr,
    X_val,
    y_val,
    fold_id,
):

    name = dataset_cfg["name"]
    print(
        f"\n{'-' * 55}"
        f"\nFold {fold_id + 1} — Hyperparameter tuning"
        f"\n{'-' * 55}"
    )

    best_hps = {}
    pos_weight = make_pos_weight(
        y_tr,
        device,
    )
    
    if pos_weight is None:
        raise ValueError(
            f"{name} Fold {fold_id + 1}: "
            "degenerate training labels."
        )
        
    vocab_size = len(vocab)
    train_loader, val_loader = make_gru_loaders(
        X_tr,
        y_tr,
        X_val,
        y_val,
        vocab,
        cfg["model"]["max_len"],
        cfg["training"]["batch_size"],
    )
    
    for key, mode, grid_key in [
        (
            "GRU_Random",
            "random",
            "gru_random",
        ),

        (
            "GRU_Frozen",
            "frozen",
            "gru_frozen",
        ),

        (
            "GRU_Finetune",
            "finetune",
            "gru_finetune",
        ),

    ]:
        
        best_trial = None
        best_val = -np.inf
        
        for trial in cfg["grids"][grid_key]:
            set_seed(
                cfg["split"]["random_seed"]
                + fold_id
            )
            try:
                model = _build_gru(
                    cfg,
                    vocab_size,
                    trial.get(
                        "hidden",
                        cfg["model"]["hidden_dim"],
                    ),
                    device,
                    mode,
                )
                
                _, val_auc = train_gru(
                    model,
                    train_loader,
                    val_loader,
                    pos_weight,
                    trial["lr"],
                    cfg["training"]["epochs"],
                    cfg["training"]["early_stop"],
                    "cls",
                    device,
                )

                print(
                    f"  {key}: "
                    f"lr={trial['lr']} "
                    f"→ val ROC-AUC={val_auc:.4f}"
                )

                if val_auc > best_val:
                    best_val = val_auc
                    best_trial = copy.deepcopy(
                        trial
                    )      
            except Exception as e:
                print(
                    f"{key} trial failed: {e}"
                )
                
        if best_trial is None:
            raise RuntimeError(
                f"No successful HP trial for {key}"
            )
            
        best_hps[key] = best_trial
        print(
            f"  BEST {key}: "
            f"{best_trial}"
        )


    train_g = build_graph_list(
        X_tr,
        y_tr,
    )

    val_g = build_graph_list(
        X_val,
        y_val,
    )

    best_trial = None
    best_val = -np.inf

    for trial in cfg["grids"]["gin"]:

        set_seed(
            cfg["split"]["random_seed"]
            + fold_id
        )

        try:

            model = GINModel(
                cfg["model"]["node_feat_dim"],
                cfg["model"]["gin_hidden_dim"],
            ).to(device)

            tl = GeoLoader(
                train_g,
                batch_size=cfg["training"]["batch_size"],
                shuffle=True,
            )

            vl = GeoLoader(
                val_g,
                batch_size=cfg["training"]["batch_size"],
            )

            _, val_auc = train_gru(
                model,
                tl,
                vl,
                pos_weight,
                trial["lr"],
                cfg["training"]["epochs"],
                cfg["training"]["early_stop"],
                "cls",
                device,
            )

            print(
                f"  GIN: lr={trial['lr']} "
                f"→ val ROC-AUC={val_auc:.4f}"
            )

            if val_auc > best_val:
                best_val = val_auc
                best_trial = copy.deepcopy(
                    trial
                )

        except Exception as e:
            print(
                f"  GIN trial failed: {e}"
            )

    if best_trial is None:
        raise RuntimeError(
            "No successful HP trial for GIN"
        )

    best_hps["GIN"] = best_trial

    print(
        f"  BEST GIN: {best_trial}"
    )


    train_ds = CrossAttnDataset(
        X_tr,
        y_tr,
        vocab,
        cfg["model"]["max_len"],
    )

    val_ds = CrossAttnDataset(
        X_val,
        y_val,
        vocab,
        cfg["model"]["max_len"],
    )

    tl = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=crossattn_collate,
        num_workers=_NW,
        pin_memory=_PIN,
    )

    vl = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        collate_fn=crossattn_collate,
        num_workers=_NW,
        pin_memory=_PIN,
    )

    best_trial = None
    best_val = -np.inf
    for trial in cfg["grids"]["molbcat"]:
        set_seed(
            cfg["split"]["random_seed"]
            + fold_id
        )

        try:
            model = MolBCAT(
                vocab_size,
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

            _, val_auc, _, _ = train_molbcat(
                model,
                tl,
                vl,
                pos_weight,
                trial["lr_encoder"],
                trial["lr_head"],
                cfg["training"].get(
                    "molbcat_epochs",
                    cfg["training"]["epochs"],
                ),
                cfg["training"]["early_stop"],
                "cls",
                device,
            )

            print(
                f"  MolBCAT: "
                f"lr_enc={trial['lr_encoder']} "
                f"lr_head={trial['lr_head']} "
                f"→ val ROC-AUC={val_auc:.4f}"
            )

            if val_auc > best_val:
                best_val = val_auc
                best_trial = copy.deepcopy(
                    trial
                )

        except Exception as e:
            print(
                f"  MolBCAT trial failed: {e}"
            )

    if best_trial is None:
        raise RuntimeError(
            "No successful HP trial for MolBCAT"
        )

    best_hps["MolBCAT"] = best_trial

    print(
        f"  BEST MolBCAT: {best_trial}"
    )


    tl_cb, vl_cb = make_chemberta_loaders(
        X_tr,
        y_tr,
        X_val,
        y_val,
        cfg["training"]["batch_size"],
    )

    best_trial = None
    best_val = -np.inf

    for trial in cfg["grids"]["chemberta"]:
        set_seed(
            cfg["split"]["random_seed"]
            + fold_id
        )

        try:

            model = ChemBERTaModel().to(
                device
            )

            _, val_auc, _, _ = train_chemberta(
                model,
                tl_cb,
                vl_cb,
                pos_weight,
                trial["lr"],
                trial["lr"] * 10,
                cfg["training"].get(
                    "chemberta_epochs",
                    cfg["training"]["epochs"],
                ),
                cfg["training"]["early_stop"],
                "cls",
                device,
            )

            print(
                f"  ChemBERTa: "
                f"lr={trial['lr']} "
                f"→ val ROC-AUC={val_auc:.4f}"
            )

            if val_auc > best_val:
                best_val = val_auc
                best_trial = copy.deepcopy(
                    trial
                )

        except Exception as e:
            print(
                f"  ChemBERTa trial failed: {e}"
            )

    if best_trial is None:
        raise RuntimeError(
            "No successful HP trial for ChemBERTa"
        )

    best_hps["ChemBERTa"] = best_trial

    print(
        f"  BEST ChemBERTa: {best_trial}"
    )
    return best_hps


def run_fold(
    dataset_cfg,
    cfg,
    vocab,
    device,
    smiles,
    labels,
    train_idx,
    val_idx,
    test_idx,
    best_hps,
    fold_id,
):


    name = dataset_cfg["name"]

    print(
        f"\n{'=' * 60}"
        f"\nFOLD {fold_id + 1}"
        f"\n{'=' * 60}"
    )


    X_tr = [
        smiles[i]
        for i in train_idx
    ]

    y_tr = [
        labels[i]
        for i in train_idx
    ]

    X_val = [
        smiles[i]
        for i in val_idx
    ]

    y_val = [
        labels[i]
        for i in val_idx
    ]

    X_test = [
        smiles[i]
        for i in test_idx
    ]

    y_test = [
        labels[i]
        for i in test_idx
    ]

    print(
        f"Train: {len(X_tr)} | "
        f"Val: {len(X_val)} | "
        f"Test: {len(X_test)}"
    )


    overlap = check_scaffold_disjoint(
        smiles,
        train_idx,
        val_idx,
        test_idx,
    )

    print(
        "Scaffold overlap: "
        f"train-val={overlap['train_val_overlap']}, "
        f"train-test={overlap['train_test_overlap']}, "
        f"val-test={overlap['val_test_overlap']}"
    )

    if any(
        value != 0
        for value in overlap.values()
    ):
        raise RuntimeError(
            f"Scaffold leakage detected in "
            f"{name}, Fold {fold_id + 1}: "
            f"{overlap}"
        )

    pos_weight = make_pos_weight(
        y_tr,
        device,
    )

    if pos_weight is None:
        raise RuntimeError(
            "Degenerate training labels."
        )

    vocab_size = len(vocab)
    fold_results = {}
    train_loader, val_loader = make_gru_loaders(
        X_tr,
        y_tr,
        X_val,
        y_val,
        vocab,
        cfg["model"]["max_len"],
        cfg["training"]["batch_size"],
    )

    X_val_tensor = torch.tensor(
        [
            encode_smiles(
                s,
                vocab,
                cfg["model"]["max_len"],
            )
            for s in X_val
        ]
    )

    X_test_tensor = torch.tensor(
        [
            encode_smiles(
                s,
                vocab,
                cfg["model"]["max_len"],
            )
            for s in X_test
        ]
    )

    for key, mode in [

        ("GRU_Random", "random"),
        ("GRU_Frozen", "frozen"),
        ("GRU_Finetune", "finetune"),

    ]:

        print(
            f"\nTraining {key} "
            f"with Best HP: {best_hps[key]}"
        )

        hp = best_hps[key]
        set_seed(
            cfg["split"]["random_seed"]
            + fold_id
        )

        model = _build_gru(
            cfg,
            vocab_size,
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
            pos_weight,
            hp["lr"],
            cfg["training"]["epochs"],
            cfg["training"]["early_stop"],
            "cls",
            device,
        )

        # Validation probabilities
        val_prob = infer_gru(
            model,
            X_val_tensor,
            "cls",
            256,
            device,
        )

        # Select threshold ONLY from validation
        threshold, val_f1 = find_best_threshold(
            y_val,
            val_prob,
        )

        # Test probabilities
        test_prob = infer_gru(
            model,
            X_test_tensor,
            "cls",
            256,
            device,
        )

        metrics = classification_metrics(
            y_test,
            test_prob,
            threshold,
        )

        fold_results[key] = {
            **metrics,
            "threshold": threshold,
            "val_F1_at_threshold": val_f1,
        }

        print(
            f"{key}: "
            f"threshold={threshold:.2f}, "
            f"val_F1={val_f1:.4f}, "
            f"test_F1={metrics['F1']:.4f}"
        )

    print(
        f"\nTraining GIN "
        f"with Best HP: {best_hps['GIN']}"
    )

    hp = best_hps["GIN"]

    train_g = build_graph_list(
        X_tr,
        y_tr,
    )

    val_g = build_graph_list(
        X_val,
        y_val,
    )

    test_g = build_graph_list(
        X_test,
        y_test,
    )

    set_seed(
        cfg["split"]["random_seed"]
        + fold_id
    )

    model_gin = GINModel(
        cfg["model"]["node_feat_dim"],
        cfg["model"]["gin_hidden_dim"],
    ).to(device)

    tl_g = GeoLoader(
        train_g,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
    )

    vl_g = GeoLoader(
        val_g,
        batch_size=cfg["training"]["batch_size"],
    )

    model_gin, _ = train_gru(
        model_gin,
        tl_g,
        vl_g,
        pos_weight,
        hp["lr"],
        cfg["training"]["epochs"],
        cfg["training"]["early_stop"],
        "cls",
        device,
    )

    val_preds_gin, val_labels_gin = infer_gin(
        model_gin,
        val_g,
        "cls",
        32,
        device,
    )

    threshold, val_f1 = find_best_threshold(
        val_labels_gin,
        val_preds_gin,
    )

    test_preds_gin, test_labels_gin = infer_gin(
        model_gin,
        test_g,
        "cls",
        32,
        device,
    )

    metrics = classification_metrics(
        test_labels_gin,
        test_preds_gin,
        threshold,
    )

    fold_results["GIN"] = {
        **metrics,
        "threshold": threshold,
        "val_F1_at_threshold": val_f1,
    }

    print(
        f"GIN: "
        f"threshold={threshold:.2f}, "
        f"val_F1={val_f1:.4f}, "
        f"test_F1={metrics['F1']:.4f}"
    )


    print(
        f"\nTraining MolBCAT "
        f"with Best HP: {best_hps['MolBCAT']}"
    )

    hp = best_hps["MolBCAT"]

    train_ds = CrossAttnDataset(
        X_tr,
        y_tr,
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

    tl = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=crossattn_collate,
        num_workers=_NW,
        pin_memory=_PIN,
    )

    vl = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        collate_fn=crossattn_collate,
        num_workers=_NW,
        pin_memory=_PIN,
    )

    tl_test = DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"],
        collate_fn=crossattn_collate,
        num_workers=_NW,
        pin_memory=_PIN,
    )

    set_seed(
        cfg["split"]["random_seed"]
        + fold_id
    )

    model_ca = MolBCAT(
        vocab_size,
        cfg["model"]["emb_dim"],
        cfg["model"]["hidden_dim"],
        cfg["model"]["gin_hidden_dim"],
        cfg["model"]["node_feat_dim"],
        cfg["model"]["nhead"],
    ).to(device)

    model_ca.pad_id = vocab.get(
        "<PAD>",
        0,
    )

    load_pretrained_encoder(
        model_ca.gru_encoder,
        cfg["weights"]["pretrained_encoder"],
        device,
    )

    model_ca, _, tr_curve, vl_curve = train_molbcat(
        model_ca,
        tl,
        vl,
        pos_weight,
        hp["lr_encoder"],
        hp["lr_head"],
        cfg["training"].get(
            "molbcat_epochs",
            cfg["training"]["epochs"],
        ),
        cfg["training"]["early_stop"],
        "cls",
        device,
    )

    val_preds_ca, val_labels_ca = infer_molbcat(
        model_ca,
        vl,
        "cls",
        device,
    )

    threshold, val_f1 = find_best_threshold(
        val_labels_ca,
        val_preds_ca,
    )

    # Test prediction
    test_preds_ca, test_labels_ca = infer_molbcat(
        model_ca,
        tl_test,
        "cls",
        device,
    )

    metrics = classification_metrics(
        test_labels_ca,
        test_preds_ca,
        threshold,
    )

    fold_results["MolBCAT"] = {
        **metrics,
        "threshold": threshold,
        "val_F1_at_threshold": val_f1,
    }

    print(
        f"MolBCAT: "
        f"threshold={threshold:.2f}, "
        f"val_F1={val_f1:.4f}, "
        f"test_F1={metrics['F1']:.4f}"
    )


    print(
        f"\nTraining ChemBERTa "
        f"with Best HP: {best_hps['ChemBERTa']}"
    )

    hp = best_hps["ChemBERTa"]

    tl_cb, vl_cb = make_chemberta_loaders(
        X_tr,
        y_tr,
        X_val,
        y_val,
        cfg["training"]["batch_size"],
    )

    tl_test_cb = DataLoader(
        ChemBERTaDataset(
            X_test,
            y_test,
        ),
        batch_size=cfg["training"]["batch_size"],
        num_workers=_NW,
        pin_memory=_PIN,
    )

    set_seed(
        cfg["split"]["random_seed"]
        + fold_id
    )

    model_cb = ChemBERTaModel().to(
        device
    )

    lr = hp["lr"]

    model_cb, _, tr_curve_cb, vl_curve_cb = train_chemberta(
        model_cb,
        tl_cb,
        vl_cb,
        pos_weight,
        lr,
        lr * 10,
        cfg["training"].get(
            "chemberta_epochs",
            cfg["training"]["epochs"],
        ),
        cfg["training"]["early_stop"],
        "cls",
        device,
    )

    val_preds_cb, val_labels_cb = infer_chemberta(
        model_cb,
        vl_cb,
        "cls",
        device,
    )

    threshold, val_f1 = find_best_threshold(
        val_labels_cb,
        val_preds_cb,
    )

    test_preds_cb, test_labels_cb = infer_chemberta(
        model_cb,
        tl_test_cb,
        "cls",
        device,
    )

    metrics = classification_metrics(
        test_labels_cb,
        test_preds_cb,
        threshold,
    )

    fold_results["ChemBERTa"] = {
        **metrics,
        "threshold": threshold,
        "val_F1_at_threshold": val_f1,
    }

    print(
        f"ChemBERTa: "
        f"threshold={threshold:.2f}, "
        f"val_F1={val_f1:.4f}, "
        f"test_F1={metrics['F1']:.4f}"
    )

    return fold_results


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
        "ROC_AUC",
        "PR_AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC",
    ]

    result = {
        "Dataset": dataset_name
    }

    for model in models:

        for metric in metrics:

            vals = []

            for fold_id in fold_results:
                value = fold_results[
                    fold_id
                ][model][metric]
                if not np.isnan(value):
                    vals.append(value)

            if vals:
                result[
                    f"{model}_{metric}_mean"
                ] = float(
                    np.mean(vals)
                )

                result[
                    f"{model}_{metric}_std"
                ] = float(
                    np.std(
                        vals,
                        ddof=1,
                    )
                ) if len(vals) > 1 else 0.0

            else:

                result[
                    f"{model}_{metric}_mean"
                ] = float("nan")

                result[
                    f"{model}_{metric}_std"
                ] = float("nan")

    ca = []
    cb = []

    for fold_id in fold_results:

        ca_value = fold_results[
            fold_id
        ]["MolBCAT"]["ROC_AUC"]

        cb_value = fold_results[
            fold_id
        ]["ChemBERTa"]["ROC_AUC"]

        if (
            not np.isnan(ca_value)
            and not np.isnan(cb_value)
        ):

            ca.append(ca_value)
            cb.append(cb_value)

    if len(ca) >= 2:
        p_value = ttest_rel(
            ca,
            cb,
        ).pvalue

        effect = cohens_d_paired(
            ca,
            cb,
        )

        result[
            "p_MolBCAT_vs_ChemBERTa"
        ] = float(p_value)

        result[
            "d_MolBCAT_vs_ChemBERTa"
        ] = float(effect)
    return result

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

    vocab = load_vocab(
        cfg["weights"]["vocab"]
    )

    datasets = (
        cfg["data"]["datasets"]
        if args.dataset == "all"
        else [
            d
            for d in cfg["data"]["datasets"]
            if d["name"] == args.dataset
        ]
    )

    if not datasets:
        print(
            f"ERROR: Dataset "
            f"'{args.dataset}' not found."
        )
        sys.exit(1)

    all_results = []
    
    for dataset_cfg in datasets:

        name = dataset_cfg["name"]

        print(
            f"\n{'#' * 70}"
            f"\n# Dataset: {name}"
            f"\n{'#' * 70}"
        )


        ds = load_dataset(
            dataset_cfg["hf"]
        )["train"]

        smiles, labels = filter_valid(
            ds["SMILES"],
            ds[dataset_cfg["col"]],
            task="cls",
        )

        print(
            f"Dataset size after preprocessing: "
            f"{len(smiles)}"
        )

        n_folds = cfg["split"]["n_folds"]
        folds = scaffold_kfold_split(
            smiles,
            labels=labels,
            n_folds=n_folds,
            seed=cfg["split"]["random_seed"],
        )

        fold_results = {}
        out_dir = os.path.join(
            cfg["output"]["save_dir"],
            "5fold",
            name,
        )
        os.makedirs(out_dir, exist_ok=True)

        if args.resume:
            for existing_fold in range(1, n_folds + 1):
                ckpt_path = fold_checkpoint_path(out_dir, existing_fold)
                cached = load_checkpoint(ckpt_path)
                if cached is not None:
                    fold_results[str(existing_fold)] = cached
                    print(f"[RESUME] {name} Fold {existing_fold}: completed checkpoint loaded.")


        for fold_id in range(n_folds):
            fold_number = fold_id + 1

            if args.resume and str(fold_number) in fold_results:
                print(f"[RESUME] {name} Fold {fold_number}: SKIP (already completed).")
                continue

            print(
                f"\n{'#' * 60}"
                f"\n# {name} — Fold {fold_id + 1}/{n_folds}"
                f"\n{'#' * 60}"
            )


            test_idx = folds[fold_id]
            
            dev_idx = []

            for j, fold in enumerate(folds):

                if j != fold_id:
                    dev_idx.extend(fold)


            train_idx, val_idx = (
                scaffold_train_val_split(
                    smiles,
                    dev_idx,
                    labels=labels,
                    val_ratio=cfg["split"]["inner_val_ratio"],
                    seed=(
                        cfg["split"]["random_seed"]
                        + fold_id
                    ),
                )
            )

            print(
                f"Split sizes: "
                f"Train={len(train_idx)}, "
                f"Val={len(val_idx)}, "
                f"Test={len(test_idx)}"
            )

            overlap = check_scaffold_disjoint(
                smiles,
                train_idx,
                val_idx,
                test_idx,
            )

            print(
                "Scaffold overlap: "
                f"train-val={overlap['train_val_overlap']}, "
                f"train-test={overlap['train_test_overlap']}, "
                f"val-test={overlap['val_test_overlap']}"
            )

            if any(
                value != 0
                for value in overlap.values()
            ):

                raise RuntimeError(
                    f"Scaffold leakage detected "
                    f"in {name}, Fold {fold_id + 1}: "
                    f"{overlap}"
                )

            X_tr = [
                smiles[i]
                for i in train_idx
            ]

            y_tr = [
                labels[i]
                for i in train_idx
            ]

            X_val = [
                smiles[i]
                for i in val_idx
            ]

            y_val = [
                labels[i]
                for i in val_idx
            ]
            
            y_test = [
                labels[i]
                for i in test_idx
            ]

            if len(np.unique(y_tr)) < 2:
                raise ValueError(
                    f"{name} Fold {fold_id + 1}: "
                "training set contains only one class."
               )

            if len(np.unique(y_val)) < 2:
                raise ValueError(
                    f"{name} Fold {fold_id + 1}: "
                     "validation set contains only one class."
                 )

            if len(np.unique(y_test)) < 2:
                 raise ValueError(
                 f"{name} Fold {fold_id + 1}: "
                "test set contains only one class."
                )

            best_hps = tune_hyperparameters(
                dataset_cfg,
                cfg,
                vocab,
                device,
                X_tr,
                y_tr,
                X_val,
                y_val,
                fold_id,
            )

            fold_result = run_fold(
                dataset_cfg,
                cfg,
                vocab,
                device,
                smiles,
                labels,
                train_idx,
                val_idx,
                test_idx,
                best_hps,
                fold_id,
            )

            fold_results[
                str(fold_id + 1)
            ] = fold_result


            fold_ckpt = fold_checkpoint_path(
                out_dir,
                fold_number,
            )
            save_checkpoint(
                fold_ckpt,
                fold_result,
            )

            save_json(
                os.path.join(
                    out_dir,
                    f"fold_{fold_number}_hps.json",
                ),
                best_hps,
            )

            print(
                f"[CHECKPOINT] {name} Fold {fold_number} saved: "
                f"{fold_ckpt}"
            )
        if len(fold_results) != n_folds:
            raise RuntimeError(
                 f"Expected {n_folds} completed folds, "
                 f"but found {len(fold_results)}."
            )

        result = summarize(
            fold_results,
            name,
        )

        all_results.append(
            result
        )

        out_dir = os.path.join(
            cfg["output"]["save_dir"],
            "5fold",
            name,
        )

        save_json(
            os.path.join(
                out_dir,
                "all_folds.json",
            ),
            fold_results,
        )

    p_values = [
        r["p_MolBCAT_vs_ChemBERTa"]
        for r in all_results
        if "p_MolBCAT_vs_ChemBERTa" in r
    ]

    if p_values:
        _, p_holm, _, _ = multipletests(
            p_values,
            alpha=0.05,
            method="holm"
        )

        i = 0
        for r in all_results:
            if "p_MolBCAT_vs_ChemBERTa" in r:
                r["p_MolBCAT_vs_ChemBERTa_Holm"] = float(
                    p_holm[i]
                )
                i += 1


    if not all_results:

        print(
            "No results to summarize."
        )

        return

    models = [
        "GRU_Random",
        "GRU_Frozen",
        "GRU_Finetune",
        "GIN",
        "MolBCAT",
        "ChemBERTa",
    ]

    metrics = [
        "ROC_AUC",
        "PR_AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC",
    ]

    summary = pd.DataFrame(
        all_results
    ).set_index(
        "Dataset"
    )

    for metric in metrics:
        print(
            f"\n{'=' * 100}"
        )
        print(
            f"CLASSIFICATION — {metric} "
            f"(5-fold scaffold CV; mean ± sample SD)"
        )
        print(
            "=" * 100
        )
        header = (
            f"{'Dataset':<16}"
            + "".join(
                f"{m:>20}"
                for m in models
            )
        )

        print(header)

        print(
            "-" * 100
        )

        for dataset_name in summary.index:

            row = summary.loc[
                dataset_name
            ]

            line = (
                f"{dataset_name:<16}"
            )

            for model in models:
                mean_v = row.get(
                    f"{model}_{metric}_mean",
                    float("nan"),
                )

                std_v = row.get(
                    f"{model}_{metric}_std",
                    float("nan"),
                )

                if np.isnan(mean_v):

                    line += (
                        f"{'N/A':>20}"
                    )

                else:

                    line += (
                        f"{mean_v:>8.4f}"
                        f"±{std_v:<8.4f}"
                    )

            print(line)



    out_csv = os.path.join(
        cfg["output"]["save_dir"],
        "classification_5fold_summary.csv",
    )

    os.makedirs(
        cfg["output"]["save_dir"],
        exist_ok=True,
    )

    summary.to_csv(
        out_csv
    )

    print(
        f"\nFull 5-fold results saved to:"
        f"\n{out_csv}"
    )


if __name__ == "__main__":
    main()