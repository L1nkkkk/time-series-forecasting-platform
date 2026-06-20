"""Checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch

from ts_platform.config.schema import ModelConfig, PlatformConfig, ScalerConfig
from ts_platform.data.transforms import FeatureAwareScalerBundle
from ts_platform.models.base import BaseForecastModel
from ts_platform.models.registry import build_model
from ts_platform.scaler.base import BaseScaler
from ts_platform.scaler.registry import build_scaler

CHECKPOINT_SCHEMA_VERSION = 2
SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS = {1, 2}
CheckpointPayload = dict[str, Any]
ScalerOrBundle = BaseScaler | FeatureAwareScalerBundle


def save_checkpoint(
    path: Path,
    *,
    model: BaseForecastModel,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict[str, Any] | None,
    config: PlatformConfig,
    scaler: ScalerOrBundle,
    environment: dict[str, Any],
) -> Path:
    """Save a model checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    target_scaler = scaler.target if isinstance(scaler, FeatureAwareScalerBundle) else scaler
    feature_scaler = scaler.features if isinstance(scaler, FeatureAwareScalerBundle) else None
    target_scaler_payload = _scaler_payload(config.data.scaler, target_scaler)
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": epoch,
        "config": config.model_dump(mode="json"),
        "model": {
            "name": config.model.name,
            "params": config.model.params,
            "input_len": model.input_len,
            "output_len": model.output_len,
            "input_dim": model.input_dim,
            "target_dim": model.target_dim,
            "num_features": model.num_features,
            "state_dict": model.state_dict(),
        },
        "data": _data_metadata_from_config(config, model),
        "optimizer": {
            "name": config.training.optimizer,
            "state_dict": optimizer.state_dict() if optimizer is not None else None,
        },
        "target_scaler": target_scaler_payload,
        "metrics": metrics,
        "environment": environment,
    }
    if feature_scaler is None:
        payload["scaler"] = target_scaler_payload
    else:
        payload["feature_scaler"] = _scaler_payload(config.data.scaler, feature_scaler)
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
    if schema_version not in SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS:
        msg = (
            f"Unsupported checkpoint schema_version {schema_version!r}; "
            f"expected one of {sorted(SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS)}"
        )
        raise ValueError(msg)
    _require_mapping(payload, "model")
    _require_mapping(payload, "optimizer")
    if schema_version == 1:
        _require_mapping(payload, "scaler")
    else:
        data_info = _require_mapping(payload, "data")
        _require_mapping(payload, "target_scaler")
        if isinstance(data_info.get("feature_dim"), int) and data_info["feature_dim"] > 0:
            _require_mapping(payload, "feature_scaler")
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

    if _schema_version(checkpoint) == 1:
        model = build_model(
            model_config,
            input_len=_require_int(model_info, "input_len"),
            output_len=_require_int(model_info, "output_len"),
            num_features=_require_int(model_info, "num_features"),
        )
    else:
        model = build_model(
            model_config,
            input_len=_require_int(model_info, "input_len"),
            output_len=_require_int(model_info, "output_len"),
            input_dim=_require_int(model_info, "input_dim"),
            target_dim=_require_int(model_info, "target_dim"),
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
    """Build the target scaler and load its state from a checkpoint."""

    scaler_info = _target_scaler_info(checkpoint)
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


def restore_scalers_from_checkpoint(
    checkpoint: CheckpointPayload,
    config: ScalerConfig | None = None,
) -> ScalerOrBundle:
    """Restore either a target-only scaler or a feature-aware scaler bundle."""

    target_scaler = restore_scaler_from_checkpoint(checkpoint, config)
    if _schema_version(checkpoint) == 1 or "feature_scaler" not in checkpoint:
        return target_scaler

    feature_scaler_info = _require_mapping(checkpoint, "feature_scaler")
    checkpoint_config = _scaler_config_from_checkpoint(feature_scaler_info)
    if config is not None:
        _assert_scaler_config_compatible(checkpoint_config, config)
        scaler_config = config
    else:
        scaler_config = checkpoint_config
    feature_scaler = build_scaler(scaler_config)
    state = feature_scaler_info.get("state")
    if not isinstance(state, dict):
        msg = "Checkpoint feature_scaler.state must be a dictionary"
        raise ValueError(msg)
    feature_scaler.load_state_dict(state)
    return FeatureAwareScalerBundle(target=target_scaler, features=feature_scaler)


def validate_checkpoint_for_training(
    checkpoint: CheckpointPayload,
    config: PlatformConfig,
    *,
    num_features: int | None = None,
    input_dim: int | None = None,
    target_dim: int | None = None,
    target_cols: list[str] | None = None,
    feature_cols: list[str] | None = None,
) -> None:
    """Validate that a checkpoint can resume the current training config."""

    expected_target_dim = target_dim if target_dim is not None else num_features
    expected_input_dim = input_dim if input_dim is not None else num_features
    if expected_input_dim is None or expected_target_dim is None:
        msg = "validate_checkpoint_for_training requires num_features or input_dim/target_dim"
        raise ValueError(msg)

    model_info = _require_mapping(checkpoint, "model")
    _assert_model_config_compatible(_model_config_from_checkpoint(model_info), config.model)
    _assert_scaler_config_compatible(
        _scaler_config_from_checkpoint(_target_scaler_info(checkpoint)),
        config.data.scaler,
    )
    if "feature_scaler" in checkpoint:
        _assert_scaler_config_compatible(
            _scaler_config_from_checkpoint(_require_mapping(checkpoint, "feature_scaler")),
            config.data.scaler,
        )

    if _schema_version(checkpoint) == 1:
        if expected_input_dim != expected_target_dim:
            msg = "Checkpoint schema version 1 is target-only and cannot resume feature-aware data"
            raise ValueError(msg)
        _assert_checkpoint_ints(
            model_info,
            {
                "input_len": config.data.input_len,
                "output_len": config.data.output_len,
                "num_features": expected_target_dim,
            },
            label="model",
        )
    else:
        _assert_checkpoint_ints(
            model_info,
            {
                "input_len": config.data.input_len,
                "output_len": config.data.output_len,
                "input_dim": expected_input_dim,
                "target_dim": expected_target_dim,
                "num_features": expected_target_dim,
            },
            label="model",
        )
        data_info = _require_mapping(checkpoint, "data")
        _assert_checkpoint_ints(
            data_info,
            {
                "input_dim": expected_input_dim,
                "target_dim": expected_target_dim,
                "feature_dim": expected_input_dim - expected_target_dim,
            },
            label="data",
        )
        _assert_column_list_compatible(data_info, "target_cols", target_cols)
        _assert_column_list_compatible(data_info, "feature_cols", feature_cols)
        if expected_input_dim > expected_target_dim and "feature_scaler" not in checkpoint:
            msg = "Checkpoint feature-aware data requires feature_scaler metadata"
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


def _schema_version(checkpoint: CheckpointPayload) -> int:
    schema_version = checkpoint.get("schema_version")
    if not isinstance(schema_version, int):
        msg = "Checkpoint field 'schema_version' must be an int"
        raise ValueError(msg)
    return schema_version


def _target_scaler_info(checkpoint: CheckpointPayload) -> dict[str, Any]:
    if _schema_version(checkpoint) == 1:
        return _require_mapping(checkpoint, "scaler")
    return _require_mapping(checkpoint, "target_scaler")


def _scaler_payload(config: ScalerConfig, scaler: BaseScaler) -> dict[str, Any]:
    return {
        "name": config.name,
        "params": config.params,
        "state": scaler.state_dict(),
    }


def _data_metadata_from_config(
    config: PlatformConfig,
    model: BaseForecastModel,
) -> dict[str, Any]:
    target_cols = _list_param(config.data.params.get("target_cols"))
    feature_cols = _list_param(config.data.params.get("feature_cols"))
    return {
        "target_cols": target_cols,
        "feature_cols": feature_cols,
        "input_dim": model.input_dim,
        "target_dim": model.target_dim,
        "feature_dim": model.input_dim - model.target_dim,
    }


def _list_param(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _assert_checkpoint_ints(
    payload: dict[str, Any],
    expected: dict[str, int],
    *,
    label: str,
) -> None:
    for key, expected_value in expected.items():
        actual_value = _require_int(payload, key)
        if actual_value != expected_value:
            msg = (
                f"Checkpoint {label} {key}={actual_value} is incompatible with "
                f"current config {key}={expected_value}"
            )
            raise ValueError(msg)


def _assert_column_list_compatible(
    data_info: dict[str, Any],
    key: str,
    expected: list[str] | None,
) -> None:
    if expected is None or key not in data_info:
        return
    actual = data_info.get(key)
    if actual != expected:
        msg = (
            f"Checkpoint data {key}={actual!r} is incompatible with "
            f"current config {key}={expected!r}"
        )
        raise ValueError(msg)


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
