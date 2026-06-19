"""Built-in dataset implementations and loading helpers."""

from __future__ import annotations

import math
from typing import Any, Literal, cast

import torch

from ts_platform.config.schema import DataConfig
from ts_platform.data.base import ForecastBatch, ForecastingDataset
from ts_platform.data.catalog import DATASET_CATALOG, DatasetMetadata
from ts_platform.data.csv_dataset import CSVForecastDataset
from ts_platform.data.registry import DATASET_REGISTRY
from ts_platform.data.splits import compute_split_indices

SplitName = Literal["train", "val", "test"]


class SyntheticForecastDataset(ForecastingDataset):
    """Deterministic synthetic sine-wave forecasting dataset."""

    def __init__(
        self,
        input_len: int,
        output_len: int,
        mode: SplitName,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        *,
        length: int = 256,
        num_features: int = 1,
        noise_std: float = 0.0,
        seed: int = 42,
    ) -> None:
        if length <= input_len + output_len:
            msg = "length must be greater than input_len + output_len"
            raise ValueError(msg)
        if num_features <= 0:
            msg = "num_features must be positive"
            raise ValueError(msg)

        self.input_len = input_len
        self.output_len = output_len
        self.num_features = num_features
        self.mode = mode
        self._values = self._generate_values(length, num_features, noise_std, seed)

        total_windows = length - input_len - output_len + 1
        split_indices = compute_split_indices(total_windows, train_ratio, val_ratio, test_ratio)
        positions = {
            "train": split_indices.train,
            "val": split_indices.val,
            "test": split_indices.test,
        }[mode]
        self._starts = positions

    def __len__(self) -> int:
        """Return number of forecasting windows."""

        return len(self._starts)

    def __getitem__(self, index: int) -> ForecastBatch:
        """Return one forecasting window."""

        start = self._starts[index]
        x_end = start + self.input_len
        y_end = x_end + self.output_len
        return {"x": self._values[start:x_end], "y": self._values[x_end:y_end]}

    def scaler_fit_values(self) -> torch.Tensor:
        """Return train split windows flattened over time."""

        if not self._starts:
            msg = "dataset split has no samples"
            raise ValueError(msg)
        segments = [
            self._values[start : start + self.input_len + self.output_len] for start in self._starts
        ]
        return torch.cat(segments, dim=0)

    @staticmethod
    def _generate_values(
        length: int,
        num_features: int,
        noise_std: float,
        seed: int,
    ) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        time = torch.linspace(0, 8 * math.pi, steps=length)
        features = []
        for feature_idx in range(num_features):
            phase = feature_idx * math.pi / max(1, num_features)
            signal = torch.sin(time + phase) + 0.25 * torch.cos((feature_idx + 1) * time / 2)
            features.append(signal)
        values = torch.stack(features, dim=-1).to(torch.float32)
        if noise_std > 0:
            values = values + torch.randn(values.shape, generator=generator) * noise_std
        return values


def register_builtin_datasets() -> None:
    """Register built-in datasets once."""

    if "synthetic" not in DATASET_REGISTRY.names():
        DATASET_REGISTRY.register("synthetic", SyntheticForecastDataset)
        DATASET_CATALOG.register(
            DatasetMetadata(
                name="synthetic",
                domain="demo",
                description="Deterministic sine-wave dataset for smoke tests and examples.",
                source="generated",
                dataset_type="synthetic",
                frequency="synthetic",
            )
        )
    if "csv" not in DATASET_REGISTRY.names():
        DATASET_REGISTRY.register("csv", CSVForecastDataset)
        DATASET_CATALOG.register(
            DatasetMetadata(
                name="csv",
                domain="local",
                description="Local CSV time series dataset with time-based splits.",
                source="local-file",
                dataset_type="csv",
            )
        )


def build_dataset(config: DataConfig, mode: SplitName, seed: int) -> ForecastingDataset:
    """Build a dataset instance from config."""

    register_builtin_datasets()
    dataset_factory = cast(Any, DATASET_REGISTRY.get(config.name))
    dataset = dataset_factory(
        input_len=config.input_len,
        output_len=config.output_len,
        mode=mode,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=seed,
        **config.params,
    )
    return cast(ForecastingDataset, dataset)


register_builtin_datasets()
