"""Base model interface."""

from __future__ import annotations

from abc import abstractmethod

import torch
from torch import nn

from ts_platform.data.base import ForecastDimensions


class BaseForecastModel(nn.Module):
    """Base class for forecasting models."""

    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_features: int | None = None,
        *,
        input_dim: int | None = None,
        target_dim: int | None = None,
    ) -> None:
        super().__init__()
        if num_features is not None and (input_dim is not None or target_dim is not None):
            msg = "Pass either num_features or input_dim/target_dim, not both"
            raise ValueError(msg)
        if num_features is not None:
            resolved_input_dim = num_features
            resolved_target_dim = num_features
        else:
            if input_dim is None or target_dim is None:
                msg = "Pass num_features or both input_dim and target_dim"
                raise ValueError(msg)
            resolved_input_dim = input_dim
            resolved_target_dim = target_dim

        dimensions = ForecastDimensions(
            input_len=input_len,
            output_len=output_len,
            input_dim=resolved_input_dim,
            target_dim=resolved_target_dim,
        )
        self.input_len = dimensions.input_len
        self.output_len = dimensions.output_len
        self.input_dim = dimensions.input_dim
        self.target_dim = dimensions.target_dim
        self.num_features = dimensions.num_features

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return forecasts shaped [batch, output_len, target_dim]."""
