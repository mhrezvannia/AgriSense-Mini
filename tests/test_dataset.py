from __future__ import annotations

from pathlib import Path

from src.data.dataset import AgricultureDataset, build_label_mapping, default_image_transform, load_metadata_table
from src.utils.sample_data import generate_sample_dataset


def test_dataset_returns_image_and_metadata(tmp_path: Path) -> None:
    image_dir = tmp_path / "sample_data"
    metadata_path = tmp_path / "metadata" / "sample_metadata.csv"
    generate_sample_dataset(image_root=image_dir, metadata_path=metadata_path, samples_per_class=6)
    frame = load_metadata_table(
        metadata_csv=metadata_path,
        image_dir=image_dir,
        id_col="sample_id",
        filename_col="image_filename",
    )
    label_mapping = build_label_mapping(frame, "label")
    dataset = AgricultureDataset(
        frame=frame,
        label_mapping=label_mapping,
        metadata_features=[[0.1, 0.2]] * len(frame),
        transform=default_image_transform(128),
    )
    sample = dataset[0]
    assert sample["image"].shape == (3, 128, 128)
    assert sample["metadata"].shape[0] == 2
    assert sample["label"].ndim == 0

