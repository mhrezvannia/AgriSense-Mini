from __future__ import annotations

import torch

from src.models.fusion_model import FusionClassifier
from src.models.image_model import ImageClassifier
from src.models.metadata_model import MetadataClassifier


def test_image_model_output_shape() -> None:
    model = ImageClassifier(num_classes=3)
    logits = model(torch.randn(2, 3, 128, 128))
    assert logits.shape == (2, 3)


def test_metadata_model_output_shape() -> None:
    model = MetadataClassifier(input_dim=10, hidden_dims=[16, 8], num_classes=3)
    logits = model(torch.randn(4, 10))
    assert logits.shape == (4, 3)


def test_fusion_model_output_shapes() -> None:
    model = FusionClassifier(metadata_input_dim=10, num_classes=3)
    outputs = model(torch.randn(2, 3, 128, 128), torch.randn(2, 10), return_parts=True)
    assert outputs["logits"].shape == (2, 3)
    assert outputs["image_logits"].shape == (2, 3)
    assert outputs["metadata_logits"].shape == (2, 3)

