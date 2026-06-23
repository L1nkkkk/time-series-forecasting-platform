"""Inference-focused model export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import torch

from ts_platform.config.schema import ModelConfig, PlatformConfig, ScalerConfig
from ts_platform.data.transforms import FeatureAwareScalerBundle
from ts_platform.models.base import BaseForecastModel
from ts_platform.models.registry import build_model
from ts_platform.runner.checkpoint import ScalerOrBundle
from ts_platform.scaler.base import BaseScaler
from ts_platform.scaler.registry import build_scaler

MODEL_EXPORT_SCHEMA_VERSION = 1
MODEL_EXPORT_FORMAT = "ts_platform_model_export"
ModelExportPayload = dict[str, Any]


def save_model_export(
    path: Path,
    metadata_path: Path,
    *,
    model: BaseForecastModel,
    config: PlatformConfig,
    scaler: ScalerOrBundle,
    metrics: dict[str, Any] | None,
    data_metadata: dict[str, Any],
) -> tuple[Path, Path]:
    """Save a lightweight inference export plus a JSON metadata companion."""

    payload = _model_export_payload(
        model=model,
        config=config,
        scaler=scaler,
        metrics=metrics,
        data_metadata=data_metadata,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(_metadata_payload(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path, metadata_path


def load_model_export(path: str | Path) -> ModelExportPayload:
    """Load and validate a model export payload."""

    export_path = Path(path)
    if not export_path.exists():
        msg = f"Model export file does not exist: {export_path}"
        raise FileNotFoundError(msg)
    payload = torch.load(export_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        msg = "Model export payload must be a dictionary"
        raise ValueError(msg)
    if payload.get("format") != MODEL_EXPORT_FORMAT:
        msg = f"Unsupported model export format: {payload.get('format')!r}"
        raise ValueError(msg)
    if payload.get("schema_version") != MODEL_EXPORT_SCHEMA_VERSION:
        msg = f"Unsupported model export schema_version: {payload.get('schema_version')!r}"
        raise ValueError(msg)
    _require_mapping(payload, "model")
    _require_mapping(payload, "data")
    _require_mapping(payload, "target_scaler")
    if _require_mapping(payload, "data").get("feature_dim", 0):
        _require_mapping(payload, "feature_scaler")
    return cast(ModelExportPayload, payload)


def restore_model_from_export(payload: ModelExportPayload) -> BaseForecastModel:
    """Build a model and load its exported weights."""

    model_info = _require_mapping(payload, "model")
    model_config = _model_config_from_export(model_info)
    model = build_model(
        model_config,
        input_len=_require_int(model_info, "input_len"),
        output_len=_require_int(model_info, "output_len"),
        input_dim=_require_int(model_info, "input_dim"),
        target_dim=_require_int(model_info, "target_dim"),
    )
    state_dict = model_info.get("state_dict")
    if not isinstance(state_dict, dict):
        msg = "Model export model.state_dict must be a dictionary"
        raise ValueError(msg)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        msg = "Model export state is incompatible with the exported model config"
        raise ValueError(msg) from exc
    return model


def restore_scalers_from_export(payload: ModelExportPayload) -> ScalerOrBundle:
    """Restore target-only or feature-aware scalers from a model export."""

    target_scaler = _restore_scaler(_require_mapping(payload, "target_scaler"))
    if "feature_scaler" not in payload:
        return target_scaler
    feature_scaler = _restore_scaler(_require_mapping(payload, "feature_scaler"))
    return FeatureAwareScalerBundle(target=target_scaler, features=feature_scaler)


def _model_export_payload(
    *,
    model: BaseForecastModel,
    config: PlatformConfig,
    scaler: ScalerOrBundle,
    metrics: dict[str, Any] | None,
    data_metadata: dict[str, Any],
) -> ModelExportPayload:
    target_scaler = scaler.target if isinstance(scaler, FeatureAwareScalerBundle) else scaler
    feature_scaler = scaler.features if isinstance(scaler, FeatureAwareScalerBundle) else None
    payload: ModelExportPayload = {
        "format": MODEL_EXPORT_FORMAT,
        "schema_version": MODEL_EXPORT_SCHEMA_VERSION,
        "model": {
            "name": config.model.name,
            "params": config.model.params,
            "input_len": model.input_len,
            "output_len": model.output_len,
            "input_dim": model.input_dim,
            "target_dim": model.target_dim,
            "num_features": model.num_features,
            "state_dict": _cpu_state_dict(model.state_dict()),
        },
        "data": dict(data_metadata),
        "target_scaler": _scaler_payload(config.data.scaler, target_scaler),
        "metrics": metrics,
    }
    if feature_scaler is not None:
        payload["feature_scaler"] = _scaler_payload(config.data.scaler, feature_scaler)
    return payload


def _metadata_payload(payload: ModelExportPayload) -> dict[str, Any]:
    model_info = _require_mapping(payload, "model")
    data_info = _require_mapping(payload, "data")
    metadata: dict[str, Any] = {
        "format": payload["format"],
        "schema_version": payload["schema_version"],
        "model": {
            "name": model_info["name"],
            "params": model_info["params"],
            "input_len": model_info["input_len"],
            "output_len": model_info["output_len"],
            "input_dim": model_info["input_dim"],
            "target_dim": model_info["target_dim"],
            "num_features": model_info["num_features"],
        },
        "data": data_info,
        "target_scaler": _scaler_metadata(_require_mapping(payload, "target_scaler")),
        "metrics": payload.get("metrics"),
    }
    if "feature_scaler" in payload:
        metadata["feature_scaler"] = _scaler_metadata(_require_mapping(payload, "feature_scaler"))
    return metadata


def _scaler_payload(config: ScalerConfig, scaler: BaseScaler) -> dict[str, Any]:
    return {
        "name": config.name,
        "params": config.params,
        "state": _cpu_state_dict(scaler.state_dict()),
    }


def _scaler_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": payload.get("name"),
        "params": payload.get("params", {}),
        "fitted": bool(payload.get("state")),
    }


def _restore_scaler(payload: dict[str, Any]) -> BaseScaler:
    scaler_config = _scaler_config_from_export(payload)
    scaler = build_scaler(scaler_config)
    state = payload.get("state")
    if not isinstance(state, dict):
        msg = "Model export scaler.state must be a dictionary"
        raise ValueError(msg)
    scaler.load_state_dict(state)
    return scaler


def _model_config_from_export(model_info: dict[str, Any]) -> ModelConfig:
    name = model_info.get("name")
    params = model_info.get("params", {})
    if not isinstance(name, str):
        msg = "Model export model.name must be a string"
        raise ValueError(msg)
    if not isinstance(params, dict):
        msg = "Model export model.params must be a dictionary"
        raise ValueError(msg)
    return ModelConfig(name=name, params=params)


def _scaler_config_from_export(scaler_info: dict[str, Any]) -> ScalerConfig:
    name = scaler_info.get("name")
    params = scaler_info.get("params", {})
    if not isinstance(name, str):
        msg = "Model export scaler.name must be a string"
        raise ValueError(msg)
    if not isinstance(params, dict):
        msg = "Model export scaler.params must be a dictionary"
        raise ValueError(msg)
    return ScalerConfig(name=name, params=params)


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        msg = f"Model export field {key!r} must be a dictionary"
        raise ValueError(msg)
    return value


def _require_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        msg = f"Model export field {key!r} must be an int"
        raise ValueError(msg)
    return value


def _cpu_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    cpu_state: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            cpu_state[key] = value.detach().cpu()
        else:
            cpu_state[key] = value
    return cpu_state
