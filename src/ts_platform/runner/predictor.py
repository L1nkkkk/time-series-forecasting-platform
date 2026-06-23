"""Inference helpers for exported forecasting models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ts_platform.data.transforms import FeatureAwareScalerBundle
from ts_platform.runner.checkpoint import ScalerOrBundle
from ts_platform.runner.model_export import (
    load_model_export,
    restore_model_from_export,
    restore_scalers_from_export,
)
from ts_platform.scaler.base import BaseScaler


def predict_from_model_export(
    export_path: str | Path,
    *,
    values: list[Any],
) -> dict[str, Any]:
    """Load a model export and forecast one or more input windows."""

    payload = load_model_export(export_path)
    model = restore_model_from_export(payload)
    scaler = restore_scalers_from_export(payload)
    model.eval()

    raw_x = _input_tensor(values)
    _validate_input(raw_x, payload)
    scaled_x = _scale_input(raw_x, scaler, payload)

    with torch.no_grad():
        scaled_prediction = model(scaled_x)

    target_scaler = _target_scaler(scaler)
    prediction = target_scaler.inverse_transform(scaled_prediction)
    model_info = _require_mapping(payload, "model")
    data_info = _require_mapping(payload, "data")
    return {
        "format": "ts_platform_prediction",
        "model_export_path": str(export_path),
        "model": {
            "name": model_info.get("name"),
            "input_len": model_info.get("input_len"),
            "output_len": model_info.get("output_len"),
            "input_dim": model_info.get("input_dim"),
            "target_dim": model_info.get("target_dim"),
        },
        "data": {
            "target_cols": data_info.get("target_cols", []),
            "feature_cols": data_info.get("feature_cols", []),
            "feature_aware": data_info.get("feature_aware", False),
        },
        "input": raw_x.tolist(),
        "scaled_prediction": scaled_prediction.cpu().tolist(),
        "prediction": prediction.cpu().tolist(),
    }


def _input_tensor(values: list[Any]) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(values, dtype=torch.float32)
    except (TypeError, ValueError) as exc:
        msg = (
            "values must be numeric and shaped [input_len, input_dim] "
            "or [batch, input_len, input_dim]"
        )
        raise ValueError(msg) from exc
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        msg = "values must be shaped [input_len, input_dim] or [batch, input_len, input_dim]"
        raise ValueError(msg)
    return tensor


def _validate_input(values: torch.Tensor, payload: dict[str, Any]) -> None:
    model_info = _require_mapping(payload, "model")
    input_len = model_info.get("input_len")
    input_dim = model_info.get("input_dim")
    if values.shape[1] != input_len:
        msg = f"values input length must be {input_len}, got {values.shape[1]}"
        raise ValueError(msg)
    if values.shape[2] != input_dim:
        msg = f"values input dimension must be {input_dim}, got {values.shape[2]}"
        raise ValueError(msg)


def _scale_input(
    values: torch.Tensor,
    scaler: ScalerOrBundle,
    payload: dict[str, Any],
) -> torch.Tensor:
    data_info = _require_mapping(payload, "data")
    feature_aware = bool(data_info.get("feature_aware"))
    if not feature_aware:
        if isinstance(scaler, FeatureAwareScalerBundle):
            msg = "target-only export unexpectedly restored a feature-aware scaler"
            raise ValueError(msg)
        return scaler.transform(values)

    if not isinstance(scaler, FeatureAwareScalerBundle) or scaler.features is None:
        msg = "feature-aware export requires target and feature scalers"
        raise ValueError(msg)
    target_dim = int(_require_mapping(payload, "model")["target_dim"])
    target_x = values[..., :target_dim]
    feature_x = values[..., target_dim:]
    return torch.cat(
        [scaler.target.transform(target_x), scaler.features.transform(feature_x)],
        dim=-1,
    )


def _target_scaler(scaler: ScalerOrBundle) -> BaseScaler:
    if isinstance(scaler, FeatureAwareScalerBundle):
        return scaler.target
    return scaler


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        msg = f"model export field {key!r} must be a dictionary"
        raise ValueError(msg)
    return value
