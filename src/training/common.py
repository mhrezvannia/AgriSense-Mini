from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import AgricultureDataset, build_label_mapping, default_image_transform, load_metadata_table
from src.data.preprocessing import MetadataPreprocessor
from src.data.splits import create_splits
from src.evaluation.plots import plot_calibration_curve, plot_confusion_matrix, plot_training_history
from src.utils.io import ensure_dir, save_json, save_joblib


def prepare_data_bundle(config: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], MetadataPreprocessor, dict[str, int]]:
    paths = config["paths"]
    data_config = config["data"]
    frame = load_metadata_table(
        metadata_csv=paths["metadata_csv"],
        image_dir=paths["image_dir"],
        id_col=data_config["id_col"],
        filename_col=data_config["filename_col"],
    )
    splits = create_splits(
        frame=frame,
        label_col=data_config["label_col"],
        val_size=data_config["split"]["val_size"],
        test_size=data_config["split"]["test_size"],
        random_state=config["training"]["seed"],
        shift_column=data_config["split"]["shift_column"],
        shift_values=data_config["split"]["shift_values"],
    )
    preprocessor = MetadataPreprocessor(
        categorical_columns=data_config["categorical_columns"],
        numeric_columns=data_config["numeric_columns"],
    )
    preprocessor.fit(splits["train"])
    label_mapping = build_label_mapping(frame, data_config["label_col"])
    return splits, preprocessor, label_mapping


def build_dataloader(
    frame: pd.DataFrame,
    label_mapping: dict[str, int],
    metadata_features,
    image_size: int,
    batch_size: int,
    train: bool,
) -> DataLoader:
    dataset = AgricultureDataset(
        frame=frame,
        label_mapping=label_mapping,
        metadata_features=metadata_features,
        transform=default_image_transform(image_size=image_size, train=train),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=0)


def export_split_artifacts(
    config: dict[str, Any],
    model_name: str,
    history: dict[str, list[float]] | None,
    reports: dict[str, dict[str, Any]],
) -> None:
    figure_dir = ensure_dir(Path(config["paths"]["figure_dir"]) / model_name)
    report_dir = ensure_dir(Path(config["paths"]["report_dir"]) / model_name)
    if history is not None:
        plot_training_history(history=history, path=figure_dir / "training_history.png")
        save_json(history, report_dir / "training_history.json")
    for split_name, metrics in reports.items():
        plot_confusion_matrix(
            confusion=metrics["confusion_matrix"],
            class_names=metrics["class_names"],
            path=figure_dir / f"confusion_{split_name}.png",
        )
        plot_calibration_curve(
            y_true=torch.tensor(metrics["y_true"]).numpy(),
            y_prob=torch.tensor(metrics["y_prob"]).numpy(),
            path=figure_dir / f"calibration_{split_name}.png",
            num_bins=config["evaluation"]["calibration_bins"],
        )
        save_json(metrics, report_dir / f"{split_name}_metrics.json")


def save_preprocessor(preprocessor: MetadataPreprocessor, config: dict[str, Any]) -> None:
    model_dir = ensure_dir(config["paths"]["model_dir"])
    save_joblib(preprocessor, Path(model_dir) / "metadata_preprocessor.joblib")
