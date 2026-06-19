"""Checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch

from ts_platform.config.schema import ModelConfig, PlatformConfig, ScalerConfig
from ts_platform.models.base import BaseForecastModel
from ts_platform.models.registry import build_model
from ts_platform.scaler.base import BaseScaler
from ts_platform.scaler.registry import build_scaler

CHECKPOINT_SCHEMA_VERSION = 1
CheckpointPayload = dict[str, Any]


def save_checkpoint(
    path: Path,
    *,
    model: BaseForecastModel,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict[str, Any] | None,
    config: PlatformConfig,
    scaler: BaseScaler,
    environment: dict[str, Any],
) -> Path:
    """Save a model checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": epoch,
        "config": config.model_dump(mode="json"),
        "model": {
            "name": config.model.name,
            "params": config.model.params,
            "input_len": model.input_len,
            "output_len": model.output_len,
            "num_features": model.num_features,
            "state_dict": model.state_dict(),
        },
        "optimizer": {
            "name": config.training.optimizer,
            "state_dict": optimizer.state_dict() if optimizer is not None else None,
        },
        "scaler": {
            "name": config.data.scaler.name,
            "params": config.data.scaler.params,
            "state": scaler.state_dict(),
        },
        "metrics": metrics,
        "environment": environment,
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path) -> CheckpointPayload:
    """Load and validate a checkpoint payload."""

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        msg = f"Checkpoint file does not exist: {checkpoint_path}"
        raise FileNotFoundError(msg)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        msg = "Checkpoint payload must be a dictionary"
        raise ValueError(msg)
    schema_version = payload.get("schema_version")
    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        msg = (
            f"Unsupported checkpoint schema_version {schema_version!r}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}"
        )
        raise ValueError(msg)
    _require_mapping(payload, "model")
    _require_mapping(payload, "optimizer")
    _require_mapping(payload, "scaler")
    _require_mapping(payload, "config")
    if not isinstance(payload.get("epoch"), int):
        msg = "Checkpoint field 'epoch' must be an int"
        raise ValueError(msg)
    return cast(CheckpointPayload, payload)


def restore_model_from_checkpoint(
    checkpoint: CheckpointPayload,
    config: ModelConfig | None = None,
) -> BaseForecastModel:
    """Build a model and load its state from a checkpoint."""

    model_info = _require_mapping(checkpoint, "model")
    checkpoint_config = _model_config_from_checkpoint(model_info)
    if config is not None:
        _assert_model_config_compatible(checkpoint_config, config)
        model_config = config
    else:
        model_config = checkpoint_config

    input_len = _require_int(model_info, "input_len")
    output_len = _require_int(model_info, "output_len")
    num_features = _require_int(model_info, "num_features")
    model = build_model(
        model_config,
        input_len=input_len,
        output_len=output_len,
        num_features=num_features,
    )
    state_dict = model_info.get("state_dict")
    if not isinstance(state_dict, dict):
        msg = "Checkpoint model.state_dict must be a dictionary"
        raise ValueError(msg)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        msg = "Checkpoint model state is incompatible with the requested model config"
        raise ValueError(msg) from exc
    return model


def restore_scaler_from_checkpoint(
    checkpoint: CheckpointPayload,
    config: ScalerConfig | None = None,
) -> BaseScaler:
    """Build a scaler and load its state from a checkpoint."""

    scaler_info = _require_mapping(checkpoint, "scaler")
    checkpoint_config = _scaler_config_from_checkpoint(scaler_info)
    if config is not None:
        _assert_scaler_config_compatible(checkpoint_config, config)
        scaler_config = config
    else:
        scaler_config = checkpoint_config

    scaler = build_scaler(scaler_config)
    state = scaler_info.get("state")
    if not isinstance(state, dict):
        msg = "Checkpoint scaler.state must be a dictionary"
        raise ValueError(msg)
    scaler.load_state_dict(state)
    return scaler


def validate_checkpoint_for_training(
    checkpoint: CheckpointPayload,
    config: PlatformConfig,
    *,
    num_features: int,
) -> None:
    """Validate that a checkpoint can resume the current training config."""

    model_info = _require_mapping(checkpoint, "model")
    _assert_model_config_compatible(_model_config_from_checkpoint(model_info), config.model)
    _assert_scaler_config_compatible(
        _scaler_config_from_checkpoint(_require_mapping(checkpoint, "scaler")),
        config.data.scaler,
    )
    expected = {
        "input_len": config.data.input_len,
        "output_len": config.data.output_len,
        "num_features": num_features,
    }
    for key, expected_value in expected.items():
        actual_value = _require_int(model_info, key)
        if actual_value != expected_value:
            msg = (
                f"Checkpoint model {key}={actual_value} is incompatible with "
                f"current config {key}={expected_value}"
            )
            raise ValueError(msg)
    optimizer_info = _require_mapping(checkpoint, "optimizer")
    optimizer_name = optimizer_info.get("name")
    if optimizer_name != config.training.optimizer:
        msg = (
            f"Checkpoint optimizer {optimizer_name!r} is incompatible with "
            f"current optimizer {config.training.optimizer!r}"
        )
        raise ValueError(msg)


def load_optimizer_state_from_checkpoint(
    checkpoint: CheckpointPayload,
    optimizer: torch.optim.Optimizer | None,
) -> None:
    """Load optimizer state if both checkpoint and current model are trainable."""

    optimizer_info = _require_mapping(checkpoint, "optimizer")
    state_dict = optimizer_info.get("state_dict")
    if optimizer is None:
        if state_dict is not None:
            msg = "Checkpoint has optimizer state but current model has no trainable parameters"
            raise ValueError(msg)
        return
    if state_dict is None:
        return
    if not isinstance(state_dict, dict):
        msg = "Checkpoint optimizer.state_dict must be a dictionary or null"
        raise ValueError(msg)
    try:
        optimizer.load_state_dict(state_dict)
    except ValueError as exc:
        msg = "Checkpoint optimizer state is incompatible with current optimizer"
        raise ValueError(msg) from exc


def _model_config_from_checkpoint(model_info: dict[str, Any]) -> ModelConfig:
    name = model_info.get("name")
    params = model_info.get("params", {})
    if not isinstance(name, str):
        msg = "Checkpoint model.name must be a string"
        raise ValueError(msg)
    if not isinstance(params, dict):
        msg = "Checkpoint model.params must be a dictionary"
        raise ValueError(msg)
    return ModelConfig(name=name, params=params)


def _scaler_config_from_checkpoint(scaler_info: dict[str, Any]) -> ScalerConfig:
    name = scaler_info.get("name")
    params = scaler_info.get("params", {})
    if not isinstance(name, str):
        msg = "Checkpoint scaler.name must be a string"
        raise ValueError(msg)
    if not isinstance(params, dict):
        msg = "Checkpoint scaler.params must be a dictionary"
        raise ValueError(msg)
    return ScalerConfig(name=name, params=params)


def _assert_model_config_compatible(
    checkpoint_config: ModelConfig, current_config: ModelConfig
) -> None:
    if checkpoint_config != current_config:
        msg = (
            "Checkpoint model config is incompatible with current config: "
            f"checkpoint={checkpoint_config.model_dump(mode='json')}, "
            f"current={current_config.model_dump(mode='json')}"
        )
        raise ValueError(msg)


def _assert_scaler_config_compatible(
    checkpoint_config: ScalerConfig,
    current_config: ScalerConfig,
) -> None:
    if checkpoint_config != current_config:
        msg = (
            "Checkpoint scaler config is incompatible with current config: "
            f"checkpoint={checkpoint_config.model_dump(mode='json')}, "
            f"current={current_config.model_dump(mode='json')}"
        )
        raise ValueError(msg)


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        msg = f"Checkpoint field {key!r} must be a dictionary"
        raise ValueError(msg)
    return value


def _require_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        msg = f"Checkpoint field {key!r} must be an int"
        raise ValueError(msg)
    return value
