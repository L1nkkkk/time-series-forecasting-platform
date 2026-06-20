"""Base dataset interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypedDict

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ForecastDimensions:
    """Forecasting window and feature dimensions."""

    input_len: int
    output_len: int
    input_dim: int
    target_dim: int

    def __post_init__(self) -> None:
        if self.input_len <= 0:
            msg = "input_len must be positive"
            raise ValueError(msg)
        if self.output_len <= 0:
            msg = "output_len must be positive"
            raise ValueError(msg)
        if self.input_dim <= 0:
            msg = "input_dim must be positive"
            raise ValueError(msg)
        if self.target_dim <= 0:
            msg = "target_dim must be positive"
            raise ValueError(msg)
        if self.input_dim < self.target_dim:
            msg = "input_dim must be greater than or equal to target_dim"
            raise ValueError(msg)

    @property
    def num_features(self) -> int:
        """Compatibility alias for target-only code paths."""

        return self.target_dim


class _OptionalForecastBatchFields(TypedDict, total=False):
    target_x: torch.Tensor
    feature_x: torch.Tensor
    metadata: dict[str, Any]


class ForecastBatch(_OptionalForecastBatchFields):
    """A single forecasting sample."""

    x: torch.Tensor
    y: torch.Tensor


class ForecastingDataset(Dataset[ForecastBatch], ABC):
    """Base class for forecasting datasets."""

    input_len: int
    output_len: int
    input_dim: int
    target_dim: int
    num_features: int

    @property
    def dimensions(self) -> ForecastDimensions:
        """Return the forecasting dimensions for this dataset."""

        return ForecastDimensions(
            input_len=self.input_len,
            output_len=self.output_len,
            input_dim=self.input_dim,
            target_dim=self.target_dim,
        )

    @abstractmethod
    def __len__(self) -> int:
        """Return number of available forecasting windows."""

    @abstractmethod
    def __getitem__(self, index: int) -> ForecastBatch:
        """Return one forecasting window."""

    @abstractmethod
    def scaler_fit_values(self) -> torch.Tensor:
        """Return values that should be used to fit scalers."""
