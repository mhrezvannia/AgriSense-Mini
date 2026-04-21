from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from src.evaluation.plots import plot_feature_importance
from src.explainability.feature_importance import extract_model_feature_importance
from src.models.metadata_model import MetadataClassifier
from src.training.common import export_split_artifacts, prepare_data_bundle, save_preprocessor
from src.training.trainer import evaluate_model, fit_model, resolve_device
from src.utils.config import load_config
from src.utils.io import save_joblib, save_json, save_torch_checkpoint
from src.utils.seed import set_seed


def train_sklearn_baseline(config: dict[str, Any], split_frames, preprocessor, label_mapping: dict[str, int]) -> dict[str, Any]:
    class_names = config["model"]["class_names"]
    x_train = preprocessor.transform(split_frames["train"])
    y_train = split_frames["train"][config["data"]["label_col"]].map(label_mapping).to_numpy()
    baseline = LogisticRegression(max_iter=1000, class_weight="balanced")
    baseline.fit(x_train, y_train)

    reports: dict[str, Any] = {}
    for split_name, split_frame in split_frames.items():
        if split_frame.empty:
            continue
        x_split = preprocessor.transform(split_frame)
        y_true = split_frame[config["data"]["label_col"]].map(label_mapping).to_numpy()
        y_prob = baseline.predict_proba(x_split)
        y_pred = np.argmax(y_prob, axis=1)
        from src.evaluation.metrics import compute_classification_metrics

        metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            class_names=class_names,
            calibration_bins=config["evaluation"]["calibration_bins"],
        )
        metrics["y_true"] = y_true.tolist()
        metrics["y_pred"] = y_pred.tolist()
        metrics["y_prob"] = y_prob.tolist()
        reports[split_name] = metrics

    feature_importance = extract_model_feature_importance(baseline, preprocessor.get_feature_names())
    plot_feature_importance(
        feature_names=feature_importance["feature_names"],
        importances=feature_importance["importances"],
        path=Path(config["paths"]["figure_dir"]) / "sklearn_metadata" / "feature_importance.png",
    )
    save_joblib(baseline, Path(config["paths"]["model_dir"]) / "sklearn_metadata.joblib")
    save_json(feature_importance, Path(config["paths"]["report_dir"]) / "sklearn_metadata" / "feature_importance.json")
    export_split_artifacts(config=config, model_name="sklearn_metadata", history=None, reports=reports)
    return reports


def train_torch_metadata(config: dict[str, Any], split_frames, preprocessor, label_mapping: dict[str, int]) -> dict[str, Any]:
    class_names = config["model"]["class_names"]
    batch_size = config["data"]["batch_size"]
    image_size = config["data"]["image_size"]
    split_features = {name: preprocessor.transform(frame) for name, frame in split_frames.items()}
    from src.training.common import build_dataloader

    train_loader = build_dataloader(
        frame=split_frames["train"],
        label_mapping=label_mapping,
        metadata_features=split_features["train"],
        image_size=image_size,
        batch_size=batch_size,
        train=True,
    )
    val_loader = build_dataloader(
        frame=split_frames["val"],
        label_mapping=label_mapping,
        metadata_features=split_features["val"],
        image_size=image_size,
        batch_size=batch_size,
        train=False,
    )

    model = MetadataClassifier(
        input_dim=split_features["train"].shape[1],
        hidden_dims=config["model"]["metadata_hidden_dims"],
        num_classes=len(class_names),
        dropout=config["model"]["dropout"],
    )
    device = resolve_device(config["training"]["device"])
    checkpoint_path = Path(config["paths"]["model_dir"]) / "torch_metadata.pt"
    model, history, _ = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=class_names,
        task="metadata",
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        epochs=config["training"]["epochs"],
        patience=config["training"]["patience"],
        device=device,
        checkpoint_path=checkpoint_path,
    )

    reports: dict[str, Any] = {}
    for split_name, split_frame in split_frames.items():
        if split_frame.empty:
            continue
        loader = build_dataloader(
            frame=split_frame,
            label_mapping=label_mapping,
            metadata_features=split_features[split_name],
            image_size=image_size,
            batch_size=batch_size,
            train=False,
        )
        reports[split_name] = evaluate_model(
            model=model,
            loader=loader,
            class_names=class_names,
            task="metadata",
            device=device,
        )

    save_torch_checkpoint(
        {
            "state_dict": model.state_dict(),
            "metadata_dim": split_features["train"].shape[1],
            "class_names": class_names,
            "model_type": "torch_metadata",
        },
        checkpoint_path,
    )
    export_split_artifacts(config=config, model_name="torch_metadata", history=history, reports=reports)
    return reports


def main(config_path: str) -> None:
    config = load_config(config_path)
    set_seed(config["training"]["seed"])
    split_frames, preprocessor, label_mapping = prepare_data_bundle(config)
    save_preprocessor(preprocessor, config)
    reports = {
        "sklearn_metadata": train_sklearn_baseline(config, split_frames, preprocessor, label_mapping),
        "torch_metadata": train_torch_metadata(config, split_frames, preprocessor, label_mapping),
    }
    save_json(reports, Path(config["paths"]["report_dir"]) / "metadata_reports.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train metadata baselines for AgriSense Mini.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
