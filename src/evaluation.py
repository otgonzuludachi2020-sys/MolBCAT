"""
Evaluation metrics for classification and regression tasks.
"""

import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def calc_cls_metrics(
    y_true,
    y_prob,
    threshold: float = 0.5,
) -> dict:
   

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob).reshape(-1)

    if len(np.unique(y_true)) < 2:
        return {
            "ROC_AUC": float("nan"),
            "PR_AUC": float("nan"),
            "Precision": float("nan"),
            "Recall": float("nan"),
            "F1": float("nan"),
            "MCC": float("nan"),
        }

    y_pred = (y_prob >= threshold).astype(int)

    return {
        "ROC_AUC": float(
            roc_auc_score(y_true, y_prob)
        ),

        "PR_AUC": float(
            average_precision_score(y_true, y_prob)
        ),

        "Precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "Recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "F1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "MCC": float(
            matthews_corrcoef(
                y_true,
                y_pred,
            )
        ),
    }


def calc_reg_metrics(y_true, y_pred) -> dict:
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred).reshape(-1)

    return {
        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),

        "MAE": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),

        "R2": (
            float(r2_score(y_true, y_pred))
            if len(np.unique(y_true)) > 1
            else float("nan")
        ),
    }


def safe_auc(y_true, y_prob) -> float:
    
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob).reshape(-1)

    if len(np.unique(y_true)) < 2:
        return float("nan")

    return float(
        roc_auc_score(
            y_true,
            y_prob,
        )
    )


def make_pos_weight(
    labels,
    device,
    max_weight: float = 10.0,
):
   
    import torch

    pos = float(sum(labels))
    neg = float(len(labels) - pos)

    if pos == 0 or neg == 0:
        return None

    return torch.tensor(
        [min(neg / pos, max_weight)],
        dtype=torch.float32,
        device=device,
    )


def cohens_d_paired(x, y) -> float:
   

    diff = np.asarray(x) - np.asarray(y)

    std = np.std(
        diff,
        ddof=1,
    )

    return (
        float(np.mean(diff) / std)
        if std != 0
        else float("nan")
    )