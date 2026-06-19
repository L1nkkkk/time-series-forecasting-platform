from __future__ import annotations

import torch

from ts_platform.models import MODEL_REGISTRY
from ts_platform.models.linear import LinearForecastModel
from ts_platform.models.mlp import MLPForecastModel
from ts_platform.models.naive import NaiveLastValueModel


def test_model_forward_shapes() -> None:
    x = torch.randn(4, 8, 2)
    models = [
        NaiveLastValueModel(input_len=8, output_len=3, num_features=2),
        LinearForecastModel(input_len=8, output_len=3, num_features=2),
        MLPForecastModel(input_len=8, output_len=3, num_features=2, hidden_sizes=[16]),
    ]

    for model in models:
        assert model(x).shape == (4, 3, 2)


def test_model_registry() -> None:
    assert {"naive", "linear", "mlp"}.issubset(set(MODEL_REGISTRY.names()))
