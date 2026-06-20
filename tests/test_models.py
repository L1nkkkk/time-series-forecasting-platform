from __future__ import annotations

import pytest
import torch

from ts_platform.config.schema import ModelConfig
from ts_platform.models import MODEL_REGISTRY
from ts_platform.models.base import BaseForecastModel
from ts_platform.models.linear import LinearForecastModel
from ts_platform.models.mlp import MLPForecastModel
from ts_platform.models.moving_average import MovingAverageForecastModel
from ts_platform.models.naive import NaiveLastValueModel
from ts_platform.models.recurrent import GRUForecastModel, LSTMForecastModel, RNNForecastModel
from ts_platform.models.registry import build_model
from ts_platform.models.seasonal_naive import SeasonalNaiveForecastModel
from ts_platform.models.tcn import TCNForecastModel


class _DummyForecastModel(BaseForecastModel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], self.output_len, self.target_dim)


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


def test_base_model_old_num_features_constructor() -> None:
    model = _DummyForecastModel(input_len=8, output_len=2, num_features=3)

    assert model.input_dim == 3
    assert model.target_dim == 3
    assert model.num_features == 3


def test_base_model_new_input_target_dim_constructor() -> None:
    model = _DummyForecastModel(input_len=8, output_len=2, input_dim=5, target_dim=3)

    assert model.input_dim == 5
    assert model.target_dim == 3
    assert model.num_features == 3
    assert model(torch.randn(4, 8, 5)).shape == (4, 2, 3)


def test_base_model_rejects_mixed_dimension_arguments() -> None:
    with pytest.raises(ValueError, match="Pass either num_features or input_dim/target_dim"):
        _DummyForecastModel(input_len=8, output_len=2, num_features=3, input_dim=3, target_dim=3)


def test_base_model_rejects_input_dim_less_than_target_dim() -> None:
    with pytest.raises(ValueError, match="input_dim must be greater than or equal to target_dim"):
        _DummyForecastModel(input_len=8, output_len=2, input_dim=1, target_dim=2)


def test_base_model_num_features_aliases_target_dim() -> None:
    model = _DummyForecastModel(input_len=8, output_len=2, input_dim=4, target_dim=2)

    assert model.num_features == model.target_dim


def test_build_model_with_num_features_still_works() -> None:
    model = build_model(
        ModelConfig(name="linear"),
        input_len=8,
        output_len=2,
        num_features=3,
    )

    assert isinstance(model, LinearForecastModel)
    assert model.input_dim == 3
    assert model.target_dim == 3
    assert model.num_features == 3


def test_build_model_rejects_missing_dimension_arguments() -> None:
    with pytest.raises(ValueError, match="Pass num_features or both input_dim and target_dim"):
        build_model(ModelConfig(name="linear"), input_len=8, output_len=2)


def test_build_model_target_only_dimensions_alias() -> None:
    model = build_model(
        ModelConfig(name="linear"),
        input_len=8,
        output_len=2,
        input_dim=3,
        target_dim=3,
    )

    assert isinstance(model, LinearForecastModel)
    assert model.input_dim == 3
    assert model.target_dim == 3
    assert model.num_features == 3


def test_existing_models_have_input_and_target_dim_attributes() -> None:
    models = [
        NaiveLastValueModel(input_len=6, output_len=2, num_features=2),
        MovingAverageForecastModel(input_len=6, output_len=2, num_features=2),
        SeasonalNaiveForecastModel(input_len=6, output_len=2, num_features=2, season_length=3),
        LinearForecastModel(input_len=6, output_len=2, num_features=2),
        MLPForecastModel(input_len=6, output_len=2, num_features=2, hidden_sizes=[8]),
        RNNForecastModel(input_len=6, output_len=2, num_features=2, hidden_size=8),
        GRUForecastModel(input_len=6, output_len=2, num_features=2, hidden_size=8),
        LSTMForecastModel(input_len=6, output_len=2, num_features=2, hidden_size=8),
        TCNForecastModel(input_len=6, output_len=2, num_features=2, hidden_channels=8),
    ]

    for model in models:
        assert model.input_dim == 2
        assert model.target_dim == 2
        assert model.num_features == 2


def test_model_forward_shapes_still_target_only() -> None:
    x = torch.randn(4, 6, 2)
    models = [
        NaiveLastValueModel(input_len=6, output_len=2, num_features=2),
        MovingAverageForecastModel(input_len=6, output_len=2, num_features=2),
        SeasonalNaiveForecastModel(input_len=6, output_len=2, num_features=2, season_length=3),
        LinearForecastModel(input_len=6, output_len=2, num_features=2),
        MLPForecastModel(input_len=6, output_len=2, num_features=2, hidden_sizes=[8]),
        RNNForecastModel(input_len=6, output_len=2, num_features=2, hidden_size=8),
        GRUForecastModel(input_len=6, output_len=2, num_features=2, hidden_size=8),
        LSTMForecastModel(input_len=6, output_len=2, num_features=2, hidden_size=8),
        TCNForecastModel(input_len=6, output_len=2, num_features=2, hidden_channels=8),
    ]

    for model in models:
        assert model(x).shape == (4, 2, 2)


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


def test_rnn_model_shape() -> None:
    model = RNNForecastModel(input_len=6, output_len=3, num_features=2, hidden_size=8)
    x = torch.randn(4, 6, 2)

    y = model(x)

    assert y.shape == (4, 3, 2)
    assert torch.isfinite(y).all()


def test_gru_model_shape() -> None:
    model = GRUForecastModel(input_len=6, output_len=3, num_features=2, hidden_size=8)
    x = torch.randn(4, 6, 2)

    y = model(x)

    assert y.shape == (4, 3, 2)
    assert torch.isfinite(y).all()


def test_lstm_model_shape() -> None:
    model = LSTMForecastModel(input_len=6, output_len=3, num_features=2, hidden_size=8)
    x = torch.randn(4, 6, 2)

    y = model(x)

    assert y.shape == (4, 3, 2)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("model_cls", (RNNForecastModel, GRUForecastModel, LSTMForecastModel))
def test_recurrent_models_support_bidirectional(model_cls) -> None:
    model = model_cls(
        input_len=6,
        output_len=2,
        num_features=1,
        hidden_size=5,
        bidirectional=True,
    )
    x = torch.randn(3, 6, 1)

    y = model(x)

    assert y.shape == (3, 2, 1)
    assert model.projection.in_features == 10


@pytest.mark.parametrize("model_cls", (RNNForecastModel, GRUForecastModel, LSTMForecastModel))
def test_recurrent_models_reject_invalid_hidden_size(model_cls) -> None:
    with pytest.raises(ValueError, match="hidden_size must be positive"):
        model_cls(input_len=4, output_len=2, num_features=1, hidden_size=0)


@pytest.mark.parametrize("model_cls", (RNNForecastModel, GRUForecastModel, LSTMForecastModel))
def test_recurrent_models_reject_invalid_num_layers(model_cls) -> None:
    with pytest.raises(ValueError, match="num_layers must be positive"):
        model_cls(input_len=4, output_len=2, num_features=1, num_layers=0)


@pytest.mark.parametrize("dropout", (-0.1, 1.0))
@pytest.mark.parametrize("model_cls", (RNNForecastModel, GRUForecastModel, LSTMForecastModel))
def test_recurrent_models_reject_invalid_dropout(model_cls, dropout: float) -> None:
    with pytest.raises(ValueError, match="dropout must be >= 0 and < 1"):
        model_cls(input_len=4, output_len=2, num_features=1, dropout=dropout)


def test_recurrent_models_registered() -> None:
    assert {"rnn", "gru", "lstm"}.issubset(set(MODEL_REGISTRY.names()))


def test_tcn_model_shape() -> None:
    model = TCNForecastModel(input_len=6, output_len=3, num_features=2, hidden_channels=8)
    x = torch.randn(4, 6, 2)

    y = model(x)

    assert y.shape == (4, 3, 2)
    assert torch.isfinite(y).all()


def test_tcn_model_handles_multivariate_input() -> None:
    model = TCNForecastModel(
        input_len=5,
        output_len=2,
        num_features=3,
        hidden_channels=6,
        num_layers=2,
    )
    x = torch.randn(7, 5, 3)

    y = model(x)

    assert y.shape == (7, 2, 3)
    assert torch.isfinite(y).all()


def test_tcn_rejects_invalid_hidden_channels() -> None:
    with pytest.raises(ValueError, match="hidden_channels must be positive"):
        TCNForecastModel(input_len=4, output_len=2, num_features=1, hidden_channels=0)


def test_tcn_rejects_invalid_num_layers() -> None:
    with pytest.raises(ValueError, match="num_layers must be positive"):
        TCNForecastModel(input_len=4, output_len=2, num_features=1, num_layers=0)


def test_tcn_rejects_invalid_kernel_size() -> None:
    with pytest.raises(ValueError, match="kernel_size must be positive"):
        TCNForecastModel(input_len=4, output_len=2, num_features=1, kernel_size=0)


@pytest.mark.parametrize("dropout", (-0.1, 1.0))
def test_tcn_rejects_invalid_dropout(dropout: float) -> None:
    with pytest.raises(ValueError, match="dropout must be >= 0 and < 1"):
        TCNForecastModel(input_len=4, output_len=2, num_features=1, dropout=dropout)


def test_tcn_model_registered() -> None:
    assert "tcn" in MODEL_REGISTRY.names()


def test_model_registry_includes_recurrent_and_tcn() -> None:
    assert {"rnn", "gru", "lstm", "tcn"}.issubset(set(MODEL_REGISTRY.names()))
