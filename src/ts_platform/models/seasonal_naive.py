"""Seasonal naive forecasting baseline."""

from __future__ import annotations

import math

import torch

from ts_platform.models.base import BaseForecastModel
from ts_platform.models.registry import MODEL_REGISTRY


class SeasonalNaiveForecastModel(BaseForecastModel):
    """Repeat the final observed season over the forecast horizon."""

    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_features: int | None = None,
        season_length: int | None = None,
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
        if season_length is None:
            msg = "season_length is required"
            raise ValueError(msg)
        if season_length <= 0:
            msg = "season_length must be positive"
            raise ValueError(msg)
        if season_length > input_len:
            msg = "season_length cannot be greater than input_len"
            raise ValueError(msg)
        self.season_length = season_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Cycle the last season until output_len predictions are produced."""

        target_x = self.target_slice(x)
        season = target_x[:, -self.season_length :, :]
        repeats = math.ceil(self.output_len / self.season_length)
        return season.repeat(1, repeats, 1)[:, : self.output_len, :]


if "seasonal_naive" not in MODEL_REGISTRY.names():
    MODEL_REGISTRY.register("seasonal_naive", SeasonalNaiveForecastModel)
