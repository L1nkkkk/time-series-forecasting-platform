"""Evaluation loop."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import nn

from ts_platform.metrics.regression import compute_metrics
from ts_platform.scaler.base import BaseScaler

MetricValues = dict[str, float]
EvaluationMetrics = dict[str, MetricValues]


def evaluate(
    model: nn.Module,
    batches: Iterable[dict[str, Any]],
    metric_names: list[str],
    device: torch.device,
    *,
    scaler: BaseScaler | None = None,
    inverse_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    include_scaled_metrics: bool = True,
) -> EvaluationMetrics:
    """Evaluate a model and report original-scale metrics by default."""

    if scaler is not None and inverse_transform is not None:
        msg = "Pass either scaler or inverse_transform, not both"
        raise ValueError(msg)

    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in batches:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            y_pred = model(x)
            predictions.append(y_pred.detach().cpu())
            targets.append(y.detach().cpu())
    if not predictions:
        msg = "cannot evaluate with no batches"
        raise ValueError(msg)

    y_pred_scaled = torch.cat(predictions, dim=0)
    y_true_scaled = torch.cat(targets, dim=0)
    transform = inverse_transform or (scaler.inverse_transform if scaler is not None else None)
    if transform is None:
        y_pred_original = y_pred_scaled
        y_true_original = y_true_scaled
    else:
        y_pred_original = transform(y_pred_scaled)
        y_true_original = transform(y_true_scaled)

    results: EvaluationMetrics = {
        "original": compute_metrics(y_pred_original, y_true_original, metric_names)
    }
    if include_scaled_metrics:
        results["scaled"] = compute_metrics(y_pred_scaled, y_true_scaled, metric_names)
    return results


def collect_forecast_samples(
    model: nn.Module,
    batches: Iterable[dict[str, Any]],
    device: torch.device,
    *,
    scaler: BaseScaler | None = None,
    inverse_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    target_cols: list[str] | None = None,
    max_samples: int = 16,
) -> dict[str, Any]:
    """Collect a small original-scale forecast payload for visual inspection."""

    if scaler is not None and inverse_transform is not None:
        msg = "Pass either scaler or inverse_transform, not both"
        raise ValueError(msg)
    if max_samples <= 0:
        msg = "max_samples must be positive"
        raise ValueError(msg)

    model.eval()
    samples: list[dict[str, Any]] = []
    seen = 0
    transform = inverse_transform or (scaler.inverse_transform if scaler is not None else None)
    with torch.no_grad():
        for batch in batches:
            x = batch["x"].to(device)
            y_scaled = batch["y"].detach().cpu()
            y_pred_scaled = model(x).detach().cpu()
            if transform is None:
                y_true = y_scaled
                y_pred = y_pred_scaled
            else:
                y_true = transform(y_scaled)
                y_pred = transform(y_pred_scaled)

            for sample_index in range(y_true.shape[0]):
                if len(samples) >= max_samples:
                    return _forecast_sample_payload(samples, target_cols, seen)
                samples.append(
                    {
                        "sample_index": seen,
                        "actual": _round_nested_floats(y_true[sample_index].tolist()),
                        "predicted": _round_nested_floats(y_pred[sample_index].tolist()),
                    }
                )
                seen += 1

    return _forecast_sample_payload(samples, target_cols, seen)


def _forecast_sample_payload(
    samples: list[dict[str, Any]],
    target_cols: list[str] | None,
    observed_samples: int,
) -> dict[str, Any]:
    horizon = len(samples[0]["actual"]) if samples else 0
    target_dim = len(samples[0]["actual"][0]) if samples and samples[0]["actual"] else 0
    labels = target_cols or [f"target_{index + 1}" for index in range(target_dim)]
    return {
        "split": "test",
        "target_cols": labels,
        "horizon": list(range(1, horizon + 1)),
        "sample_count": len(samples),
        "observed_samples": observed_samples,
        "samples": samples,
    }


def _round_nested_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return [_round_nested_floats(item) for item in value]
    return value
