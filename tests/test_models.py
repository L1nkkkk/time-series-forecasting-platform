from __future__ import annotations

import pytest
import torch

from ts_platform.models import MODEL_REGISTRY
from ts_platform.models.linear import LinearForecastModel
from ts_platform.models.mlp import MLPForecastModel
from ts_platform.models.moving_average import MovingAverageForecastModel
from ts_platform.models.naive import NaiveLastValueModel
from ts_platform.models.seasonal_naive import SeasonalNaiveForecastModel


def test_model_forward_shapes() -> None:
    x = torch.randn(4, 8, 2)
    models = [
        NaiveLastValueModel(input_len=8, output_len=3, num_features=2),
        MovingAverageForecastModel(input_len=8, output_len=3, num_features=2),
        SeasonalNaiveForecastModel(input_len=8, output_len=3, num_features=2, season_length=4),
        LinearForecastModel(input_len=8, output_len=3, num_features=2),
        MLPForecastModel(input_len=8, output_len=3, num_features=2, hidden_sizes=[16]),
    ]

    for model in models:
        assert model(x).shape == (4, 3, 2)


def test_model_registry() -> None:
    assert {"naive", "linear", "mlp"}.issubset(set(MODEL_REGISTRY.names()))


def test_moving_average_model_shape() -> None:
    model = MovingAverageForecastModel(input_len=4, output_len=3, num_features=2, window_size=2)
    x = torch.randn(5, 4, 2)

    assert model(x).shape == (5, 3, 2)


def test_moving_average_model_values() -> None:
    model = MovingAverageForecastModel(input_len=4, output_len=2, num_features=1, window_size=2)
    x = torch.tensor([[[1.0], [2.0], [4.0], [6.0]]])

    y = model(x)

    assert y.tolist() == [[[5.0], [5.0]]]


def test_moving_average_rejects_invalid_window_size() -> None:
    with pytest.raises(ValueError, match="window_size must be positive"):
        MovingAverageForecastModel(input_len=4, output_len=2, num_features=1, window_size=0)
    with pytest.raises(ValueError, match="window_size cannot be greater than input_len"):
        MovingAverageForecastModel(input_len=4, output_len=2, num_features=1, window_size=5)


def test_seasonal_naive_model_shape() -> None:
    model = SeasonalNaiveForecastModel(input_len=5, output_len=4, num_features=2, season_length=3)
    x = torch.randn(5, 5, 2)

    assert model(x).shape == (5, 4, 2)


def test_seasonal_naive_model_values() -> None:
    model = SeasonalNaiveForecastModel(input_len=5, output_len=5, num_features=1, season_length=3)
    x = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]])

    y = model(x)

    assert y.tolist() == [[[3.0], [4.0], [5.0], [3.0], [4.0]]]


def test_seasonal_naive_rejects_invalid_season_length() -> None:
    with pytest.raises(ValueError, match="season_length must be positive"):
        SeasonalNaiveForecastModel(input_len=4, output_len=2, num_features=1, season_length=0)
    with pytest.raises(ValueError, match="season_length cannot be greater than input_len"):
        SeasonalNaiveForecastModel(input_len=4, output_len=2, num_features=1, season_length=5)


def test_new_baseline_models_registered() -> None:
    assert {"moving_average", "seasonal_naive"}.issubset(set(MODEL_REGISTRY.names()))
