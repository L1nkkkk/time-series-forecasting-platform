"""Model registry and builders."""

from __future__ import annotations

from typing import Any

from ts_platform.config.schema import ModelConfig
from ts_platform.data.base import ForecastDimensions
from ts_platform.data.registry import Registry
from ts_platform.models.base import BaseForecastModel

MODEL_REGISTRY: Registry[type[BaseForecastModel]] = Registry()


def build_model(
    config: ModelConfig,
    *,
    input_len: int,
    output_len: int,
    num_features: int | None = None,
    input_dim: int | None = None,
    target_dim: int | None = None,
) -> BaseForecastModel:
    """Build a model from config."""

    if num_features is not None and (input_dim is not None or target_dim is not None):
        msg = "Pass either num_features or input_dim/target_dim, not both"
        raise ValueError(msg)

    model_cls = MODEL_REGISTRY.get(config.name)
    params: dict[str, Any] = dict(config.params)
    if num_features is not None:
        return model_cls(
            input_len=input_len,
            output_len=output_len,
            num_features=num_features,
            **params,
        )
    if input_dim is None or target_dim is None:
        msg = "Pass num_features or both input_dim and target_dim"
        raise ValueError(msg)

    dimensions = ForecastDimensions(
        input_len=input_len,
        output_len=output_len,
        input_dim=input_dim,
        target_dim=target_dim,
    )
    return model_cls(
        input_len=dimensions.input_len,
        output_len=dimensions.output_len,
        input_dim=dimensions.input_dim,
        target_dim=dimensions.target_dim,
        **params,
    )


def registered_model_names() -> list[str]:
    """Return registered model names."""

    return MODEL_REGISTRY.names()
