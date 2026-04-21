from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ImageEncoder(nn.Module):
    """Compact CNN encoder backed by a torchvision ResNet."""

    def __init__(
        self,
        embedding_dim: int = 128,
        dropout: float = 0.2,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        weights = None
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        try:
            self.backbone = resnet18(weights=weights)
        except Exception:
            self.backbone = resnet18(weights=None)
        backbone_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(backbone_features, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.target_layer = self.backbone.layer4[-1]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return self.projector(features)


class ImageClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 128,
        dropout: float = 0.2,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = ImageEncoder(
            embedding_dim=embedding_dim,
            dropout=dropout,
            pretrained=pretrained,
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, images: torch.Tensor, return_embedding: bool = False) -> dict[str, Any] | torch.Tensor:
        embedding = self.encoder(images)
        logits = self.classifier(embedding)
        if return_embedding:
            return {"logits": logits, "embedding": embedding}
        return logits
