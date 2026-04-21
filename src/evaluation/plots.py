from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.calibration import calibration_curve_points
from src.utils.io import ensure_dir


def plot_training_history(history: dict[str, list[float]], path: str | Path) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    axis.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
    if history.get("val_accuracy"):
        axis.plot(epochs, history["val_accuracy"], label="Val Accuracy", linewidth=2)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Metric")
    axis.set_title("Training History")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion_matrix(
    confusion: Sequence[Sequence[int]],
    class_names: Sequence[str],
    path: str | Path,
) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    matrix = np.array(confusion)
    fig, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Greens")
    axis.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=30, ha="right")
    axis.set_yticks(np.arange(len(class_names)), labels=class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion Matrix")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            axis.text(col_index, row_index, matrix[row_index, col_index], ha="center", va="center", color="black")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    path: str | Path,
    num_bins: int = 10,
) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    prob_true, prob_pred = calibration_curve_points(y_true=y_true, y_prob=y_prob, num_bins=num_bins)
    fig, axis = plt.subplots(figsize=(5, 4))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    axis.plot(prob_pred, prob_true, marker="o", linewidth=2, color="#2E7D32", label="Model")
    axis.set_xlabel("Predicted confidence")
    axis.set_ylabel("Observed frequency")
    axis.set_title("Calibration Curve")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_feature_importance(
    feature_names: Sequence[str],
    importances: Sequence[float],
    path: str | Path,
    title: str = "Metadata Feature Importance",
) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    fig, axis = plt.subplots(figsize=(7, 4))
    order = np.argsort(importances)
    sorted_features = np.array(feature_names)[order]
    sorted_importances = np.array(importances)[order]
    axis.barh(sorted_features, sorted_importances, color="#1565C0")
    axis.set_title(title)
    axis.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
