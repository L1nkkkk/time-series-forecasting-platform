"""MLP forecasting model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import nn

from ts_platform.models.base import BaseForecastModel
from ts_platform.models.registry import MODEL_REGISTRY


class MLPForecastModel(BaseForecastModel):
    """Multi-layer perceptron forecaster."""

    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_features: int | None = None,
        hidden_sizes: Sequence[int] = (64, 32),
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
        layers: list[nn.Module] = []
        in_features = input_len * self.input_dim
        for hidden_size in hidden_sizes:
            if hidden_size <= 0:
                msg = "hidden_sizes must contain positive integers"
                raise ValueError(msg)
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_size
        layers.append(nn.Linear(in_features, output_len * self.target_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP forecast."""

        self.validate_input(x)
        batch_size = x.shape[0]
        flattened = x.reshape(batch_size, self.input_len * self.input_dim)
        output = self.network(flattened)
        return cast(torch.Tensor, output.reshape(batch_size, self.output_len, self.target_dim))


if "mlp" not in MODEL_REGISTRY.names():
    MODEL_REGISTRY.register("mlp", MLPForecastModel)
