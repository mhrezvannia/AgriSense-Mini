from __future__ import annotations

from typing import Any

import pandas as pd

from src.evaluation.metrics import metrics_to_row


def summarize_shift_gap(base_metrics: dict[str, Any], shift_metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "accuracy_drop": float(base_metrics["accuracy"] - shift_metrics["accuracy"]),
        "f1_drop": float(base_metrics["f1_macro"] - shift_metrics["f1_macro"]),
        "ece_increase": float(shift_metrics.get("ece", 0.0) - base_metrics.get("ece", 0.0)),
    }


def build_comparison_table(report_bundle: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, split_reports in report_bundle.items():
        for split_name, metrics in split_reports.items():
            rows.append(metrics_to_row(model_name=model_name, split_name=split_name, metrics=metrics))
    return pd.DataFrame(rows).sort_values(by=["split", "f1_macro"], ascending=[True, False]).reset_index(drop=True)
