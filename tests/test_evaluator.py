from __future__ import annotations

import pytest
import torch
from torch import nn

from ts_platform.data.transforms import FeatureAwareScalerBundle
from ts_platform.runner.evaluator import evaluate
from ts_platform.runner.trainer import _target_scaler
from ts_platform.scaler.base import BaseScaler
from ts_platform.scaler.standard import StandardScaler


class ConstantForecastModel(nn.Module):
    """Tiny model that returns a fixed scaled prediction."""

    def __init__(self, prediction: torch.Tensor) -> None:
        super().__init__()
        self.prediction = prediction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the configured prediction for every batch."""

        return self.prediction.repeat(x.shape[0], 1, 1)


class ExplodingScaler(BaseScaler):
    def __init__(self) -> None:
        self.fitted = True

    def fit(self, values: torch.Tensor) -> BaseScaler:
        return self

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return values

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        msg = "feature scaler inverse_transform should not be used"
        raise AssertionError(msg)

    def state_dict(self) -> dict[str, object]:
        return {"fitted": self.fitted}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.fitted = bool(state.get("fitted", True))


def test_original_scale_metrics() -> None:
    scaler = StandardScaler()
    scaler.load_state_dict(
        {
            "fitted": True,
            "mean": torch.tensor([[10.0]]),
            "std": torch.tensor([[2.0]]),
            "eps": 1e-8,
        }
    )
    model = ConstantForecastModel(torch.tensor([[[1.0]]]))
    batches = [{"x": torch.zeros(1, 1, 1), "y": torch.tensor([[[0.0]]])}]

    metrics = evaluate(
        model,
        batches,
        ["mae"],
        torch.device("cpu"),
        scaler=scaler,
        include_scaled_metrics=True,
    )

    assert metrics["scaled"]["mae"] == pytest.approx(1.0)
    assert metrics["original"]["mae"] == pytest.approx(2.0)


def test_evaluator_can_report_scaled_and_original_metrics() -> None:
    scaler = StandardScaler()
    scaler.load_state_dict(
        {
            "fitted": True,
            "mean": torch.tensor([[5.0]]),
            "std": torch.tensor([[3.0]]),
            "eps": 1e-8,
        }
    )
    model = ConstantForecastModel(torch.tensor([[[2.0]]]))
    batches = [{"x": torch.zeros(1, 1, 1), "y": torch.tensor([[[1.0]]])}]

    metrics = evaluate(
        model,
        batches,
        ["mae", "mse"],
        torch.device("cpu"),
        inverse_transform=scaler.inverse_transform,
        include_scaled_metrics=True,
    )

    assert set(metrics) == {"original", "scaled"}
    assert metrics["scaled"]["mae"] == pytest.approx(1.0)
    assert metrics["original"]["mae"] == pytest.approx(3.0)


def test_evaluator_feature_aware_uses_target_inverse_transform_only() -> None:
    target_scaler = StandardScaler()
    target_scaler.load_state_dict(
        {
            "fitted": True,
            "mean": torch.tensor([[10.0]]),
            "std": torch.tensor([[2.0]]),
            "eps": 1e-8,
        }
    )
    bundle = FeatureAwareScalerBundle(target=target_scaler, features=ExplodingScaler())
    model = ConstantForecastModel(torch.tensor([[[1.0]]]))
    batches = [{"x": torch.zeros(1, 2, 3), "y": torch.tensor([[[0.0]]])}]

    metrics = evaluate(
        model,
        batches,
        ["mae"],
        torch.device("cpu"),
        scaler=_target_scaler(bundle),
        include_scaled_metrics=True,
    )

    assert metrics["scaled"]["mae"] == pytest.approx(1.0)
    assert metrics["original"]["mae"] == pytest.approx(2.0)
