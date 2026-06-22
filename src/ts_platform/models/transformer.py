"""Transformer encoder forecasting model."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from ts_platform.models.base import BaseForecastModel
from ts_platform.models.registry import MODEL_REGISTRY


class TransformerForecastModel(BaseForecastModel):
    """Lightweight Transformer encoder for direct multi-step forecasting."""

    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_features: int | None = None,
        d_model: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.0,
        *,
        input_dim: int | None = None,
        target_dim: int | None = None,
    ) -> None:
        super().__init__(
            input_len,
            output_len,
            num_features,
            input_dim=input_dim,
            target_dim=target_dim,
        )
        if d_model <= 0:
            msg = "d_model must be positive"
            raise ValueError(msg)
        if num_heads <= 0:
            msg = "num_heads must be positive"
            raise ValueError(msg)
        if d_model % num_heads != 0:
            msg = "d_model must be divisible by num_heads"
            raise ValueError(msg)
        if num_layers <= 0:
            msg = "num_layers must be positive"
            raise ValueError(msg)
        if dim_feedforward <= 0:
            msg = "dim_feedforward must be positive"
            raise ValueError(msg)
        if dropout < 0 or dropout >= 1:
            msg = "dropout must be >= 0 and < 1"
            raise ValueError(msg)

        self.input_projection = nn.Linear(self.input_dim, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, input_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projection = nn.Linear(d_model, output_len * self.target_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input history and project the final token to the horizon."""

        self.validate_input(x)
        batch_size = x.shape[0]
        encoded = self.input_projection(x) + self.position_embedding
        memory = self.encoder(encoded)
        output = self.projection(memory[:, -1, :])
        return cast(torch.Tensor, output.reshape(batch_size, self.output_len, self.target_dim))


if "transformer" not in MODEL_REGISTRY.names():
    MODEL_REGISTRY.register("transformer", TransformerForecastModel)
