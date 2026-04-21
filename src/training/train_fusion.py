from __future__ import annotations

import argparse
from pathlib import Path

from src.models.fusion_model import FusionClassifier
from src.training.common import build_dataloader, export_split_artifacts, prepare_data_bundle, save_preprocessor
from src.training.trainer import evaluate_model, fit_model, resolve_device
from src.utils.config import load_config
from src.utils.io import save_json, save_torch_checkpoint
from src.utils.seed import set_seed


def main(config_path: str) -> None:
    config = load_config(config_path)
    set_seed(config["training"]["seed"])
    split_frames, preprocessor, label_mapping = prepare_data_bundle(config)
    save_preprocessor(preprocessor, config)
    split_features = {name: preprocessor.transform(frame) for name, frame in split_frames.items()}
    class_names = config["model"]["class_names"]
    image_size = config["data"]["image_size"]
    batch_size = config["data"]["batch_size"]

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
    model = FusionClassifier(
        metadata_input_dim=split_features["train"].shape[1],
        num_classes=len(class_names),
        image_embedding_dim=config["model"]["image_embedding_dim"],
        metadata_hidden_dims=config["model"]["metadata_hidden_dims"],
        fusion_hidden_dims=config["model"]["fusion_hidden_dims"],
        dropout=config["model"]["dropout"],
        pretrained_backbone=config["model"]["pretrained_backbone"],
    )
    device = resolve_device(config["training"]["device"])
    checkpoint_path = Path(config["paths"]["model_dir"]) / "fusion_model.pt"
    model, history, _ = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=class_names,
        task="fusion",
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        epochs=config["training"]["epochs"],
        patience=config["training"]["patience"],
        device=device,
        checkpoint_path=checkpoint_path,
    )
    reports = {}
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
            task="fusion",
            device=device,
        )
    save_torch_checkpoint(
        {
            "state_dict": model.state_dict(),
            "metadata_dim": split_features["train"].shape[1],
            "class_names": class_names,
            "model_type": "fusion",
        },
        checkpoint_path,
    )
    export_split_artifacts(config=config, model_name="fusion_model", history=history, reports=reports)
    save_json(reports, Path(config["paths"]["report_dir"]) / "fusion_reports.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the fused multimodal model for AgriSense Mini.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
