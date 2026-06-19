"""Scaler registry."""

from __future__ import annotations

from ts_platform.config.schema import ScalerConfig
from ts_platform.data.registry import Registry
from ts_platform.scaler.base import BaseScaler, IdentityScaler
from ts_platform.scaler.minmax import MinMaxScaler
from ts_platform.scaler.standard import StandardScaler

SCALER_REGISTRY: Registry[type[BaseScaler]] = Registry()


def register_builtin_scalers() -> None:
    """Register built-in scalers once."""

    existing = set(SCALER_REGISTRY.names())
    if "identity" not in existing:
        SCALER_REGISTRY.register("identity", IdentityScaler)
    if "standard" not in existing:
        SCALER_REGISTRY.register("standard", StandardScaler)
    if "minmax" not in existing:
        SCALER_REGISTRY.register("minmax", MinMaxScaler)


def build_scaler(config: ScalerConfig) -> BaseScaler:
    """Build a scaler from config."""

    register_builtin_scalers()
    scaler_cls = SCALER_REGISTRY.get(config.name)
    params = dict(config.params)
    if scaler_cls is MinMaxScaler and "feature_range" in params:
        params["feature_range"] = tuple(params["feature_range"])
    return scaler_cls(**params)


def registered_scaler_names() -> list[str]:
    """Return registered scaler names."""

    register_builtin_scalers()
    return SCALER_REGISTRY.names()


register_builtin_scalers()
