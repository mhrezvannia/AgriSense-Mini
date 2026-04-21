from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, num_bins: int = 10) -> float:
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    correctness = (predictions == y_true).astype(np.float32)
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences >= left) & (confidences < right)
        if not np.any(in_bin):
            continue
        bin_accuracy = correctness[in_bin].mean()
        bin_confidence = confidences[in_bin].mean()
        ece += float(np.abs(bin_accuracy - bin_confidence) * in_bin.mean())
    return ece


def compute_calibration_summary(y_true: np.ndarray, y_prob: np.ndarray, num_bins: int = 10) -> dict[str, float]:
    summary = {"ece": expected_calibration_error(y_true=y_true, y_prob=y_prob, num_bins=num_bins)}
    if y_prob.shape[1] == 2:
        summary["brier_score"] = float(brier_score_loss(y_true, y_prob[:, 1]))
    return summary


def calibration_curve_points(y_true: np.ndarray, y_prob: np.ndarray, num_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    if y_prob.shape[1] == 2:
        prob_true, prob_pred = calibration_curve(y_true, y_prob[:, 1], n_bins=num_bins)
        return prob_true, prob_pred
    predicted_labels = np.argmax(y_prob, axis=1)
    confidence = np.max(y_prob, axis=1)
    correctness = (predicted_labels == y_true).astype(np.int32)
    prob_true, prob_pred = calibration_curve(correctness, confidence, n_bins=num_bins)
    return prob_true, prob_pred
