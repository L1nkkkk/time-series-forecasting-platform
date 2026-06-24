"""Result payload types for experiments."""

from __future__ import annotations

from typing import Any

MetricValues = dict[str, float]
MetricGroups = dict[str, MetricValues]
OptionalMetricGroups = MetricGroups | None


def result_payload(
    *,
    run_metadata: dict[str, str],
    checkpoint_path: str,
    history: list[dict[str, Any]],
    validation_metrics: OptionalMetricGroups,
    test_metrics: MetricGroups,
    resumed_from: str | None,
    best_checkpoint_path: str | None = None,
    best_epoch: int | None = None,
    best_metric: dict[str, Any] | None = None,
    model_export_path: str | None = None,
    model_export_metadata_path: str | None = None,
    data_metadata: dict[str, Any] | None = None,
    forecast_samples: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical serializable training result payload."""

    payload: dict[str, Any] = {
        **run_metadata,
        "checkpoint_path": checkpoint_path,
        "history": history,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    if model_export_path is not None:
        payload["model_export_path"] = model_export_path
    if model_export_metadata_path is not None:
        payload["model_export_metadata_path"] = model_export_metadata_path
    if best_checkpoint_path is not None:
        payload["best_checkpoint_path"] = best_checkpoint_path
    if best_epoch is not None:
        payload["best_epoch"] = best_epoch
    if best_metric is not None:
        payload["best_metric"] = best_metric
    if data_metadata is not None:
        payload["data_metadata"] = data_metadata
    if forecast_samples is not None:
        payload["forecast_samples"] = forecast_samples
    if resumed_from is not None:
        payload["resumed_from"] = resumed_from
    return payload
