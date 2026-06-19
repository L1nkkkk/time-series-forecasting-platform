"""Linear forecasting model."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from ts_platform.models.base import BaseForecastModel
from ts_platform.models.registry import MODEL_REGISTRY


class LinearForecastModel(BaseForecastModel):
    """Single linear projection from history to forecast horizon."""

    def __init__(self, input_len: int, output_len: int, num_features: int) -> None:
        super().__init__(input_len, output_len, num_features)
        self.projection = nn.Linear(input_len * num_features, output_len * num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project flattened history into the forecast horizon."""

        if x.ndim != 3:
            msg = "x must be shaped [batch, input_len, num_features]"
            raise ValueError(msg)
        batch_size = x.shape[0]
        flattened = x.reshape(batch_size, self.input_len * self.num_features)
        output = self.projection(flattened)
        return cast(torch.Tensor, output.reshape(batch_size, self.output_len, self.num_features))


if "linear" not in MODEL_REGISTRY.names():
    MODEL_REGISTRY.register("linear", LinearForecastModel)
