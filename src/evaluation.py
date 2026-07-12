"""
Evaluation metrics for classification and regression tasks.
"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)


def calc_cls_metrics(y_true, y_prob) -> dict:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities.

    Returns:
        Dict with ROC_AUC, PR_AUC, Precision, Recall, F1.
        All NaN if only one class is present.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob).reshape(-1)

    if len(np.unique(y_true)) < 2:
        return {k: float('nan') for k in
                ['ROC_AUC', 'PR_AUC', 'Precision', 'Recall', 'F1']}

    y_pred = (y_prob >= 0.5).astype(int)
    return {
        'ROC_AUC':   float(roc_auc_score(y_true, y_prob)),
        'PR_AUC':    float(average_precision_score(y_true, y_prob)),
        'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall':    float(recall_score(y_true, y_pred, zero_division=0)),
        'F1':        float(f1_score(y_true, y_pred, zero_division=0)),
    }


def calc_reg_metrics(y_true, y_pred) -> dict:
    """
    Compute regression metrics.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Dict with RMSE, MAE, R2.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred).reshape(-1)
    return {
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE':  float(mean_absolute_error(y_true, y_pred)),
        'R2': float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float('nan')
    }


def safe_auc(y_true, y_prob) -> float:
    """ROC-AUC that returns NaN when only one class is present."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob).reshape(-1)
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return float(roc_auc_score(y_true, y_prob))


def make_pos_weight(labels, device, max_weight: float = 5.0):
    """
    Compute positive class weight for imbalanced classification.

    Returns:
        torch.Tensor of shape (1,) on the given device, or None if degenerate.
    """
    import torch
    pos = float(sum(labels))
    neg = float(len(labels) - pos)
    if pos == 0 or neg == 0:
        return None
    return torch.tensor([min(neg / pos, max_weight)],
                        dtype=torch.float32).to(device)


def cohens_d_paired(x, y) -> float:
    """Cohen's d effect size for paired samples."""
    diff = np.array(x) - np.array(y)
    std  = np.std(diff, ddof=1)
    return float(np.mean(diff) / std) if std != 0 else float('nan')
