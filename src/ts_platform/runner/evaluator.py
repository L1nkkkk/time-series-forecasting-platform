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
