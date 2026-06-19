"""Evaluation loop."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from ts_platform.metrics.regression import compute_metrics


def evaluate(
    model: nn.Module,
    batches: Iterable[dict[str, Any]],
    metric_names: list[str],
    device: torch.device,
) -> dict[str, float]:
    """Evaluate a model over batches."""

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
    return compute_metrics(torch.cat(predictions, dim=0), torch.cat(targets, dim=0), metric_names)
