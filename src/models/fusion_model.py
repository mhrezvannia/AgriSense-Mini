from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from src.models.image_model import ImageEncoder
from src.models.metadata_model import MetadataEncoder


class FusionClassifier(nn.Module):
    """Late-fusion classifier with modality-specific heads for inspection."""

    def __init__(
        self,
        metadata_input_dim: int,
        num_classes: int,
        image_embedding_dim: int = 128,
        metadata_hidden_dims: Sequence[int] = (64, 32),
        fusion_hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.2,
        pretrained_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(
            embedding_dim=image_embedding_dim,
            dropout=dropout,
            pretrained=pretrained_backbone,
        )
        self.metadata_encoder = MetadataEncoder(
            input_dim=metadata_input_dim,
            hidden_dims=metadata_hidden_dims,
            dropout=dropout,
        )
        self.image_head = nn.Linear(image_embedding_dim, num_classes)
        self.metadata_head = nn.Linear(self.metadata_encoder.output_dim, num_classes)

        fusion_layers: list[nn.Module] = []
        previous_dim = image_embedding_dim + self.metadata_encoder.output_dim
        for hidden_dim in fusion_hidden_dims:
            fusion_layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim
        fusion_layers.append(nn.Linear(previous_dim, num_classes))
        self.fusion_head = nn.Sequential(*fusion_layers)

    def forward(
        self,
        images: torch.Tensor,
        metadata: torch.Tensor,
        return_parts: bool = False,
    ) -> dict[str, Any] | torch.Tensor:
        image_embedding = self.image_encoder(images)
        metadata_embedding = self.metadata_encoder(metadata)
        fused_embedding = torch.cat([image_embedding, metadata_embedding], dim=1)
        image_logits = self.image_head(image_embedding)
        metadata_logits = self.metadata_head(metadata_embedding)
        logits = self.fusion_head(fused_embedding)
        if return_parts:
            return {
                "logits": logits,
                "image_logits": image_logits,
                "metadata_logits": metadata_logits,
                "image_embedding": image_embedding,
                "metadata_embedding": metadata_embedding,
            }
        return logits
