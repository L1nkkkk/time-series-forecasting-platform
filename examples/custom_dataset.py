"""Example showing how to register a custom dataset."""

from __future__ import annotations

import torch

from ts_platform.data.base import ForecastBatch, ForecastingDataset
from ts_platform.data.registry import DATASET_REGISTRY


class CustomTinyDataset(ForecastingDataset):
    """Small custom dataset example."""

    def __init__(self, input_len: int = 4, output_len: int = 2, num_features: int = 1) -> None:
        self.input_len = input_len
        self.output_len = output_len
        self.num_features = num_features
        self._values = torch.arange(30, dtype=torch.float32).reshape(30, 1)

    def __len__(self) -> int:
        """Return number of windows."""

        return len(self._values) - self.input_len - self.output_len + 1

    def __getitem__(self, index: int) -> ForecastBatch:
        """Return one custom sample."""

        x_end = index + self.input_len
        y_end = x_end + self.output_len
        return {"x": self._values[index:x_end], "y": self._values[x_end:y_end]}

    def scaler_fit_values(self) -> torch.Tensor:
        """Return values for scaler fitting."""

        return self._values


if "custom_tiny" not in DATASET_REGISTRY.names():
    DATASET_REGISTRY.register("custom_tiny", CustomTinyDataset)
