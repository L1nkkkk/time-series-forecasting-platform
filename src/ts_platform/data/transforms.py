"""Dataset transform wrappers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ts_platform.data.base import ForecastBatch, ForecastingDataset
from ts_platform.scaler.base import BaseScaler


@dataclass(frozen=True)
class FeatureAwareScalerBundle:
    """Separate scalers for target and input-only feature values."""

    target: BaseScaler
    features: BaseScaler | None = None

    def has_features(self) -> bool:
        """Return whether the bundle carries a feature scaler."""

        return self.features is not None


class ScaledForecastingDataset(ForecastingDataset):
    """Apply a fitted scaler to both inputs and targets."""

    def __init__(
        self,
        dataset: ForecastingDataset,
        scaler: BaseScaler | FeatureAwareScalerBundle,
    ) -> None:
        self.dataset = dataset
        if isinstance(scaler, FeatureAwareScalerBundle):
            self.scaler = scaler.target
            self.target_scaler = scaler.target
            self.feature_scaler = scaler.features
        else:
            if dataset.feature_dim > 0:
                msg = "feature-aware datasets require FeatureAwareScalerBundle"
                raise ValueError(msg)
            self.scaler = scaler
            self.target_scaler = scaler
            self.feature_scaler = None
        if dataset.feature_dim > 0 and self.feature_scaler is None:
            msg = "feature-aware datasets require feature scaler"
            raise ValueError(msg)
        self.input_len = dataset.input_len
        self.output_len = dataset.output_len
        self.input_dim = dataset.input_dim
        self.target_dim = dataset.target_dim
        self.feature_dim = dataset.feature_dim
        self.num_features = dataset.num_features

    def __len__(self) -> int:
        """Return wrapped dataset length."""

        return len(self.dataset)

    def __getitem__(self, index: int) -> ForecastBatch:
        """Return scaled sample."""

        sample = self.dataset[index]
        if self.feature_dim == 0:
            scaled_sample: ForecastBatch = {
                "x": self.target_scaler.transform(sample["x"]),
                "y": self.target_scaler.transform(sample["y"]),
            }
            if "metadata" in sample:
                scaled_sample["metadata"] = sample["metadata"]
            return scaled_sample

        target_x = sample.get("target_x")
        feature_x = sample.get("feature_x")
        if target_x is None or feature_x is None:
            msg = "feature-aware samples must include target_x and feature_x"
            raise ValueError(msg)
        if self.feature_scaler is None:
            msg = "feature-aware datasets require feature scaler"
            raise ValueError(msg)
        scaled_target_x = self.target_scaler.transform(target_x)
        scaled_y = self.target_scaler.transform(sample["y"])
        scaled_feature_x = self.feature_scaler.transform(feature_x)
        scaled_sample = {
            "x": torch.cat([scaled_target_x, scaled_feature_x], dim=-1),
            "y": scaled_y,
            "target_x": scaled_target_x,
            "feature_x": scaled_feature_x,
        }
        if "metadata" in sample:
            scaled_sample["metadata"] = sample["metadata"]
        return scaled_sample

    def scaler_fit_values(self) -> torch.Tensor:
        """Delegate scaler fit values to wrapped dataset."""

        return self.dataset.scaler_fit_values()

    def target_scaler_fit_values(self) -> torch.Tensor:
        """Delegate target scaler fit values to wrapped dataset."""

        return self.dataset.target_scaler_fit_values()

    def feature_scaler_fit_values(self) -> torch.Tensor:
        """Delegate feature scaler fit values to wrapped dataset."""

        return self.dataset.feature_scaler_fit_values()
