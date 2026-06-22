"""N-BEATS-style forecasting model."""

from __future__ import annotations

import torch
from torch import nn

from ts_platform.models.base import BaseForecastModel
from ts_platform.models.registry import MODEL_REGISTRY


class _NBeatsBlock(nn.Module):
    """One generic N-BEATS block with backcast and forecast heads."""

    def __init__(
        self,
        *,
        input_size: int,
        forecast_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_size
        self.network = nn.Sequential(*layers)
        self.backcast = nn.Linear(hidden_size, input_size)
        self.forecast = nn.Linear(hidden_size, forecast_size)

    def forward(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return backcast and forecast components for a flattened residual."""

        hidden = self.network(residual)
        return self.backcast(hidden), self.forecast(hidden)


class NBeatsForecastModel(BaseForecastModel):
    """Compact generic N-BEATS forecaster for direct multi-step prediction."""

    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_features: int | None = None,
        hidden_size: int = 64,
        num_blocks: int = 3,
        num_layers: int = 2,
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
        if hidden_size <= 0:
            msg = "hidden_size must be positive"
            raise ValueError(msg)
        if num_blocks <= 0:
            msg = "num_blocks must be positive"
            raise ValueError(msg)
        if num_layers <= 0:
            msg = "num_layers must be positive"
            raise ValueError(msg)
        if dropout < 0 or dropout >= 1:
            msg = "dropout must be >= 0 and < 1"
            raise ValueError(msg)

        input_size = input_len * self.input_dim
        forecast_size = output_len * self.target_dim
        self.blocks = nn.ModuleList(
            [
                _NBeatsBlock(
                    input_size=input_size,
                    forecast_size=forecast_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run residual block stack and return accumulated forecasts."""

        self.validate_input(x)
        batch_size = x.shape[0]
        residual = x.reshape(batch_size, self.input_len * self.input_dim)
        forecast = residual.new_zeros(batch_size, self.output_len * self.target_dim)
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        return forecast.reshape(batch_size, self.output_len, self.target_dim)


if "nbeats" not in MODEL_REGISTRY.names():
    MODEL_REGISTRY.register("nbeats", NBeatsForecastModel)
