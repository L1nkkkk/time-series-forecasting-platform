"""Naive forecasting model."""

from __future__ import annotations

import torch

from ts_platform.models.base import BaseForecastModel
from ts_platform.models.registry import MODEL_REGISTRY


class NaiveLastValueModel(BaseForecastModel):
    """Repeat the last observed value for every forecast step."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return repeated last values."""

        target_x = self.target_slice(x)
        return target_x[:, -1:, :].repeat(1, self.output_len, 1)


if "naive" not in MODEL_REGISTRY.names():
    MODEL_REGISTRY.register("naive", NaiveLastValueModel)
