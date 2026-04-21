from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn


class MetadataEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim
        self.network = nn.Sequential(*layers)
        self.output_dim = previous_dim

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        return self.network(metadata)


class MetadataClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        num_classes: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = MetadataEncoder(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
        self.classifier = nn.Linear(self.encoder.output_dim, num_classes)

    def forward(self, metadata: torch.Tensor, return_embedding: bool = False) -> dict[str, Any] | torch.Tensor:
        embedding = self.encoder(metadata)
        logits = self.classifier(embedding)
        if return_embedding:
            return {"logits": logits, "embedding": embedding}
        return logits
