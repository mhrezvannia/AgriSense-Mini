from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def default_image_transform(image_size: int, train: bool = False) -> transforms.Compose:
    steps: list[Any] = []
    if train:
        steps.extend(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.82, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=12),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
            ]
        )
    else:
        steps.append(transforms.Resize((image_size, image_size)))
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(steps)


def load_metadata_table(
    metadata_csv: str | Path,
    image_dir: str | Path,
    id_col: str,
    filename_col: str,
) -> pd.DataFrame:
    frame = pd.read_csv(metadata_csv)
    required_columns = {id_col, filename_col, "label"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Metadata file is missing required columns: {sorted(missing_columns)}")
    image_root = Path(image_dir)
    frame["image_path"] = frame[filename_col].map(lambda name: str((image_root / name).resolve()))
    missing_paths = [path for path in frame["image_path"] if not Path(path).exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing image files for {len(missing_paths)} samples. Example: {missing_paths[0]}")
    frame[id_col] = frame[id_col].astype(str)
    return frame


def build_label_mapping(frame: pd.DataFrame, label_col: str) -> dict[str, int]:
    labels = sorted(frame[label_col].unique().tolist())
    return {label: index for index, label in enumerate(labels)}


class AgricultureDataset(Dataset[dict[str, Any]]):
    """Dataset supporting image-only, metadata-only, and fused training."""

    def __init__(
        self,
        frame: pd.DataFrame,
        label_mapping: dict[str, int],
        metadata_features: np.ndarray | None = None,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.label_mapping = label_mapping
        self.metadata_features = metadata_features
        self.transform = transform
        if self.metadata_features is not None and len(self.metadata_features) != len(self.frame):
            raise ValueError("Metadata feature matrix length must match dataframe length.")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        image_tensor = self.transform(image) if self.transform is not None else transforms.ToTensor()(image)
        sample = {
            "image": image_tensor,
            "label": torch.tensor(self.label_mapping[row["label"]], dtype=torch.long),
            "sample_id": str(row["sample_id"]),
            "image_path": row["image_path"],
        }
        if self.metadata_features is not None:
            sample["metadata"] = torch.tensor(self.metadata_features[index], dtype=torch.float32)
        else:
            sample["metadata"] = torch.zeros(0, dtype=torch.float32)
        return sample
