from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import torch


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_joblib(obj: Any, path: str | Path) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    joblib.dump(obj, output_path)


def load_joblib(path: str | Path) -> Any:
    return joblib.load(path)


def save_torch_checkpoint(payload: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    torch.save(payload, output_path)


def load_torch_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint format in {path}")
    return checkpoint

