"""Dataset transform wrappers."""

from __future__ import annotations

import torch

from ts_platform.data.base import ForecastBatch, ForecastingDataset
from ts_platform.scaler.base import BaseScaler


class ScaledForecastingDataset(ForecastingDataset):
    """Apply a fitted scaler to both inputs and targets."""

    def __init__(self, dataset: ForecastingDataset, scaler: BaseScaler) -> None:
        self.dataset = dataset
        self.scaler = scaler
        self.input_len = dataset.input_len
        self.output_len = dataset.output_len
        self.input_dim = dataset.input_dim
        self.target_dim = dataset.target_dim
        self.num_features = dataset.num_features

    def __len__(self) -> int:
        """Return wrapped dataset length."""

        return len(self.dataset)

    def __getitem__(self, index: int) -> ForecastBatch:
        """Return scaled sample."""

        sample = self.dataset[index]
        return {
            "x": self.scaler.transform(sample["x"]),
            "y": self.scaler.transform(sample["y"]),
        }

    def scaler_fit_values(self) -> torch.Tensor:
        """Delegate scaler fit values to wrapped dataset."""

        return self.dataset.scaler_fit_values()
