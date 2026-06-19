"""Model registry and builders."""

from __future__ import annotations

from typing import Any

from ts_platform.config.schema import ModelConfig
from ts_platform.data.registry import Registry
from ts_platform.models.base import BaseForecastModel

MODEL_REGISTRY: Registry[type[BaseForecastModel]] = Registry()


def build_model(
    config: ModelConfig,
    *,
    input_len: int,
    output_len: int,
    num_features: int,
) -> BaseForecastModel:
    """Build a model from config."""

    model_cls = MODEL_REGISTRY.get(config.name)
    params: dict[str, Any] = dict(config.params)
    return model_cls(
        input_len=input_len,
        output_len=output_len,
        num_features=num_features,
        **params,
    )


def registered_model_names() -> list[str]:
    """Return registered model names."""

    return MODEL_REGISTRY.names()
