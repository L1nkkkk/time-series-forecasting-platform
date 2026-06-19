"""Regression metrics for forecasting."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

MetricFn = Callable[[torch.Tensor, torch.Tensor], float]
METRIC_REGISTRY: dict[str, MetricFn] = {}


def _validate_inputs(y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
    if y_pred.shape != y_true.shape:
        msg = f"metric shape mismatch: y_pred={tuple(y_pred.shape)}, y_true={tuple(y_true.shape)}"
        raise ValueError(msg)
    if y_pred.numel() == 0:
        msg = "metrics cannot be computed on empty tensors"
        raise ValueError(msg)


def mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Mean absolute error."""

    _validate_inputs(y_pred, y_true)
    return float(torch.mean(torch.abs(y_pred - y_true)).item())


def mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Mean squared error."""

    _validate_inputs(y_pred, y_true)
    return float(torch.mean((y_pred - y_true) ** 2).item())


def rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Root mean squared error."""

    return math.sqrt(mse(y_pred, y_true))


def mape(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> float:
    """Mean absolute percentage error with epsilon protection."""

    _validate_inputs(y_pred, y_true)
    denominator = torch.clamp(torch.abs(y_true), min=eps)
    return float(torch.mean(torch.abs((y_true - y_pred) / denominator)).item())


def wape(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> float:
    """Weighted absolute percentage error with epsilon protection."""

    _validate_inputs(y_pred, y_true)
    numerator = torch.sum(torch.abs(y_true - y_pred))
    denominator = torch.clamp(torch.sum(torch.abs(y_true)), min=eps)
    return float((numerator / denominator).item())


def register_metric(name: str, metric: MetricFn) -> None:
    """Register a metric callable."""

    normalized = name.strip().lower()
    if not normalized:
        msg = "metric name must not be empty"
        raise ValueError(msg)
    METRIC_REGISTRY[normalized] = metric


def compute_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    metric_names: list[str],
) -> dict[str, float]:
    """Compute named metrics."""

    results: dict[str, float] = {}
    for name in metric_names:
        normalized = name.strip().lower()
        try:
            metric = METRIC_REGISTRY[normalized]
        except KeyError as exc:
            available = ", ".join(registered_metric_names()) or "<none>"
            msg = f"unknown metric {normalized!r}; available: {available}"
            raise KeyError(msg) from exc
        results[normalized] = metric(y_pred, y_true)
    return results


def registered_metric_names() -> list[str]:
    """Return sorted metric names."""

    return sorted(METRIC_REGISTRY)


for _name, _metric in {
    "mae": mae,
    "mse": mse,
    "rmse": rmse,
    "mape": mape,
    "wape": wape,
}.items():
    register_metric(_name, _metric)
