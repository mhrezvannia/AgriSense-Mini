from __future__ import annotations

import numpy as np

from src.evaluation.calibration import expected_calibration_error
from src.evaluation.metrics import compute_classification_metrics


def test_metrics_include_calibration_and_confusion_matrix() -> None:
    y_true = np.array([0, 1, 2, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_prob = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.5, 0.3],
            [0.1, 0.6, 0.3],
        ]
    )
    metrics = compute_classification_metrics(y_true, y_pred, y_prob, class_names=["a", "b", "c"])
    assert "accuracy" in metrics
    assert "ece" in metrics
    assert len(metrics["confusion_matrix"]) == 3


def test_expected_calibration_error_bounds() -> None:
    y_true = np.array([0, 1])
    y_prob = np.array([[0.8, 0.2], [0.4, 0.6]])
    ece = expected_calibration_error(y_true, y_prob, num_bins=5)
    assert 0.0 <= ece <= 1.0

