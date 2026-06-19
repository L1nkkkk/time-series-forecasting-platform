"""Base model interface."""

from __future__ import annotations

from abc import abstractmethod

import torch
from torch import nn


class BaseForecastModel(nn.Module):
    """Base class for forecasting models."""

    def __init__(self, input_len: int, output_len: int, num_features: int) -> None:
        super().__init__()
        if input_len <= 0 or output_len <= 0 or num_features <= 0:
            msg = "input_len, output_len, and num_features must be positive"
            raise ValueError(msg)
        self.input_len = input_len
        self.output_len = output_len
        self.num_features = num_features

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return forecasts shaped [batch, output_len, num_features]."""
