from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ts_platform.data.base import ForecastBatch, ForecastDimensions
from ts_platform.data.csv_dataset import CSVForecastDataset
from ts_platform.data.loaders import SyntheticForecastDataset
from ts_platform.data.transforms import ScaledForecastingDataset
from ts_platform.scaler.base import IdentityScaler

FIXTURE = Path("tests/fixtures/tiny_series.csv")


def test_forecast_dimensions_target_only() -> None:
    dimensions = ForecastDimensions(input_len=8, output_len=2, input_dim=3, target_dim=3)

    assert dimensions.input_len == 8
    assert dimensions.output_len == 2
    assert dimensions.input_dim == 3
    assert dimensions.target_dim == 3


@pytest.mark.parametrize(
    "kwargs",
    (
        {"input_len": 0, "output_len": 2, "input_dim": 1, "target_dim": 1},
        {"input_len": 8, "output_len": 0, "input_dim": 1, "target_dim": 1},
        {"input_len": 8, "output_len": 2, "input_dim": 0, "target_dim": 1},
        {"input_len": 8, "output_len": 2, "input_dim": 1, "target_dim": 0},
    ),
)
def test_forecast_dimensions_rejects_invalid_lengths(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        ForecastDimensions(**kwargs)


def test_forecast_dimensions_rejects_input_dim_less_than_target_dim() -> None:
    with pytest.raises(ValueError, match="input_dim must be greater than or equal to target_dim"):
        ForecastDimensions(input_len=8, output_len=2, input_dim=1, target_dim=2)


def test_forecast_dimensions_num_features_alias() -> None:
    dimensions = ForecastDimensions(input_len=8, output_len=2, input_dim=4, target_dim=2)

    assert dimensions.num_features == 2


def test_forecast_dimensions_feature_dim_target_only() -> None:
    dimensions = ForecastDimensions(input_len=8, output_len=2, input_dim=3, target_dim=3)

    assert dimensions.feature_dim == 0


def test_forecast_dimensions_feature_dim_feature_aware() -> None:
    dimensions = ForecastDimensions(input_len=8, output_len=2, input_dim=5, target_dim=2)

    assert dimensions.feature_dim == 3


def test_forecast_batch_allows_optional_metadata_runtime() -> None:
    batch: ForecastBatch = {
        "x": torch.zeros(8, 1),
        "y": torch.ones(2, 1),
        "target_x": torch.zeros(8, 1),
        "metadata": {"source": "unit-test"},
    }

    assert batch["x"].shape == (8, 1)
    assert batch["y"].shape == (2, 1)
    assert batch["target_x"].shape == (8, 1)
    assert batch["metadata"] == {"source": "unit-test"}
    assert "feature_x" not in batch


def test_synthetic_dataset_dimensions_target_only() -> None:
    dataset = SyntheticForecastDataset(
        input_len=8,
        output_len=2,
        mode="train",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        length=32,
        num_features=3,
    )

    assert dataset.input_dim == 3
    assert dataset.target_dim == 3
    assert dataset.num_features == 3
    assert dataset.dimensions == ForecastDimensions(
        input_len=8,
        output_len=2,
        input_dim=3,
        target_dim=3,
    )
    sample = dataset[0]
    assert sample["x"].shape == (8, 3)
    assert sample["y"].shape == (2, 3)


def test_csv_dataset_dimensions_target_only() -> None:
    dataset = CSVForecastDataset(
        input_len=8,
        output_len=2,
        mode="train",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        path=FIXTURE,
        timestamp_col="timestamp",
        target_cols=["value"],
        missing_policy="error",
    )

    assert dataset.input_dim == 1
    assert dataset.target_dim == 1
    assert dataset.num_features == 1
    assert dataset.dimensions == ForecastDimensions(
        input_len=8,
        output_len=2,
        input_dim=1,
        target_dim=1,
    )
    sample = dataset[0]
    assert sample["x"].shape == (8, 1)
    assert sample["y"].shape == (2, 1)


def test_forecasting_dataset_exposes_feature_dim() -> None:
    synthetic = SyntheticForecastDataset(
        input_len=8,
        output_len=2,
        mode="train",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        length=32,
        num_features=2,
    )
    csv = CSVForecastDataset(
        input_len=8,
        output_len=2,
        mode="train",
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        path=Path("tests/fixtures/tiny_series_with_features.csv"),
        timestamp_col="timestamp",
        target_cols=["value"],
        feature_cols=["temperature", "holiday"],
        missing_policy="error",
    )

    assert synthetic.feature_dim == 0
    assert synthetic.dimensions.feature_dim == 0
    assert csv.feature_dim == 2
    assert csv.dimensions.feature_dim == 2


def test_scaled_dataset_preserves_dimensions() -> None:
    dataset = SyntheticForecastDataset(
        input_len=6,
        output_len=2,
        mode="train",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        length=32,
        num_features=2,
    )
    scaled = ScaledForecastingDataset(dataset, IdentityScaler())

    assert scaled.input_dim == dataset.input_dim
    assert scaled.target_dim == dataset.target_dim
    assert scaled.feature_dim == dataset.feature_dim
    assert scaled.num_features == dataset.num_features
    assert scaled.dimensions == dataset.dimensions
    assert scaled.dimensions.feature_dim == 0
    assert scaled[0]["x"].shape == (6, 2)
    assert scaled[0]["y"].shape == (2, 2)
