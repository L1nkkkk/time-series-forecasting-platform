"""Modern linear-family forecasting baselines."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from ts_platform.models.base import BaseForecastModel
from ts_platform.models.registry import MODEL_REGISTRY


class MovingAverageDecomposition(nn.Module):
    """Series decomposition into seasonal residual and moving-average trend."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        if kernel_size <= 0:
            msg = "kernel_size must be positive"
            raise ValueError(msg)
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return seasonal and trend components shaped like ``x``."""

        left = (self.kernel_size - 1) // 2
        right = self.kernel_size - 1 - left
        padded = torch.cat(
            [
                x[:, :1, :].repeat(1, left, 1),
                x,
                x[:, -1:, :].repeat(1, right, 1),
            ],
            dim=1,
        )
        trend = self.pool(padded.transpose(1, 2)).transpose(1, 2)
        return x - trend, trend


class NLinearForecastModel(BaseForecastModel):
    """Normalized linear baseline from the LTSF-Linear family."""

    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_features: int | None = None,
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
        self.projection = nn.Linear(input_len, output_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast each target channel after subtracting its latest observed value."""

        target_x = self.target_slice(x)
        last_value = target_x[:, -1:, :].detach()
        normalized = target_x - last_value
        output = self.projection(normalized.transpose(1, 2)).transpose(1, 2)
        return cast(torch.Tensor, output + last_value)


class DLinearForecastModel(BaseForecastModel):
    """Decomposition-linear baseline from the LTSF-Linear family."""

    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_features: int | None = None,
        *,
        kernel_size: int = 25,
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
        self.decomposition = MovingAverageDecomposition(kernel_size)
        self.seasonal_projection = nn.Linear(input_len, output_len)
        self.trend_projection = nn.Linear(input_len, output_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast target channels from decomposed seasonal and trend components."""

        target_x = self.target_slice(x)
        seasonal, trend = self.decomposition(target_x)
        seasonal_output = self.seasonal_projection(seasonal.transpose(1, 2))
        trend_output = self.trend_projection(trend.transpose(1, 2))
        return cast(torch.Tensor, (seasonal_output + trend_output).transpose(1, 2))


if "nlinear" not in MODEL_REGISTRY.names():
    MODEL_REGISTRY.register("nlinear", NLinearForecastModel)
if "dlinear" not in MODEL_REGISTRY.names():
    MODEL_REGISTRY.register("dlinear", DLinearForecastModel)
