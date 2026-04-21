from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.evaluation.robustness import build_comparison_table, summarize_shift_gap
from src.utils.config import load_config
from src.utils.io import load_json, save_json


def _load_report_bundle(report_dir: Path) -> dict[str, dict[str, Any]]:
    bundle: dict[str, dict[str, Any]] = {}
    metadata_path = report_dir / "metadata_reports.json"
    if metadata_path.exists():
        bundle.update(load_json(metadata_path))
    for name, path in {
        "image_model": report_dir / "image_reports.json",
        "fusion_model": report_dir / "fusion_reports.json",
    }.items():
        if path.exists():
            bundle[name] = load_json(path)
    return bundle


def build_robustness_summary(report_bundle: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for model_name, reports in report_bundle.items():
        if "test" in reports and "test_shift" in reports:
            summary[model_name] = summarize_shift_gap(reports["test"], reports["test_shift"])
    return summary


def main(config_path: str) -> None:
    config = load_config(config_path)
    report_dir = Path(config["paths"]["report_dir"])
    bundle = _load_report_bundle(report_dir)
    comparison_table = build_comparison_table(bundle)
    comparison_table.to_csv(report_dir / "comparison_table.csv", index=False)
    save_json(build_robustness_summary(bundle), report_dir / "robustness_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate AgriSense Mini evaluation reports.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
