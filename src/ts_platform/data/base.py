"""Base dataset interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict

import torch
from torch.utils.data import Dataset


class ForecastBatch(TypedDict):
    """A single forecasting sample."""

    x: torch.Tensor
    y: torch.Tensor


class ForecastingDataset(Dataset[ForecastBatch], ABC):
    """Base class for forecasting datasets."""

    input_len: int
    output_len: int
    num_features: int

    @abstractmethod
    def __len__(self) -> int:
        """Return number of available forecasting windows."""

    @abstractmethod
    def __getitem__(self, index: int) -> ForecastBatch:
        """Return one forecasting window."""

    @abstractmethod
    def scaler_fit_values(self) -> torch.Tensor:
        """Return values that should be used to fit scalers."""
