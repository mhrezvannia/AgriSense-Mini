from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.evaluation.calibration import compute_calibration_summary


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    calibration_bins: int = 10,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "class_names": class_names,
    }
    try:
        if len(class_names) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            metrics["roc_auc_ovr"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
    except ValueError:
        metrics["roc_auc_ovr"] = None
    metrics.update(compute_calibration_summary(y_true=y_true, y_prob=y_prob, num_bins=calibration_bins))
    return metrics


def metrics_to_row(model_name: str, split_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    row = {
        "model": model_name,
        "split": split_name,
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "ece": metrics.get("ece"),
    }
    if "roc_auc" in metrics:
        row["roc_auc"] = metrics["roc_auc"]
    if "roc_auc_ovr" in metrics:
        row["roc_auc_ovr"] = metrics["roc_auc_ovr"]
    return row
