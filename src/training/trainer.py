from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_classification_metrics
from src.utils.io import save_torch_checkpoint


@dataclass
class EpochOutputs:
    loss: float
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _extract_logits(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, dict):
        return outputs["logits"]
    return outputs


def _forward_for_task(model: nn.Module, batch: dict[str, torch.Tensor], task: str) -> Any:
    if task == "image":
        return model(batch["image"])
    if task == "metadata":
        return model(batch["metadata"])
    if task == "fusion":
        return model(batch["image"], batch["metadata"])
    raise ValueError(f"Unsupported task type: {task}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    task: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochOutputs:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    y_prob_parts: list[np.ndarray] = []

    for batch in loader:
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        outputs = _forward_for_task(model=model, batch=batch, task=task)
        logits = _extract_logits(outputs)
        loss = criterion(logits, batch["label"])

        if training and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        probabilities = torch.softmax(logits.detach(), dim=1)
        predictions = probabilities.argmax(dim=1)
        total_loss += float(loss.item()) * batch["label"].shape[0]
        y_true_parts.append(batch["label"].detach().cpu().numpy())
        y_pred_parts.append(predictions.detach().cpu().numpy())
        y_prob_parts.append(probabilities.cpu().numpy())

    sample_count = sum(len(part) for part in y_true_parts)
    return EpochOutputs(
        loss=total_loss / max(sample_count, 1),
        y_true=np.concatenate(y_true_parts, axis=0),
        y_pred=np.concatenate(y_pred_parts, axis=0),
        y_prob=np.concatenate(y_prob_parts, axis=0),
    )


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_names: list[str],
    task: str,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    device: torch.device,
    checkpoint_path: str | Path,
) -> tuple[nn.Module, dict[str, list[float]], dict[str, Any]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )
    model.to(device)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_f1": [], "lr": []}
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    best_metrics: dict[str, Any] = {}
    stale_epochs = 0

    for _ in range(epochs):
        train_outputs = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            task=task,
            optimizer=optimizer,
        )
        val_outputs = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            task=task,
        )
        val_metrics = compute_classification_metrics(
            y_true=val_outputs.y_true,
            y_pred=val_outputs.y_pred,
            y_prob=val_outputs.y_prob,
            class_names=class_names,
        )
        history["train_loss"].append(train_outputs.loss)
        history["val_loss"].append(val_outputs.loss)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1_macro"])
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        scheduler.step(val_outputs.loss)

        if val_outputs.loss < best_val_loss:
            best_val_loss = val_outputs.loss
            stale_epochs = 0
            best_metrics = val_metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            save_torch_checkpoint({"state_dict": best_state, "val_metrics": best_metrics}, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training ended without a valid checkpoint.")
    model.load_state_dict(best_state)
    return model, history, best_metrics


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    class_names: list[str],
    task: str,
    device: torch.device,
) -> dict[str, Any]:
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    outputs = run_epoch(model=model, loader=loader, criterion=criterion, device=device, task=task)
    metrics = compute_classification_metrics(
        y_true=outputs.y_true,
        y_pred=outputs.y_pred,
        y_prob=outputs.y_prob,
        class_names=class_names,
    )
    metrics["loss"] = outputs.loss
    metrics["y_true"] = outputs.y_true.tolist()
    metrics["y_pred"] = outputs.y_pred.tolist()
    metrics["y_prob"] = outputs.y_prob.tolist()
    return metrics
