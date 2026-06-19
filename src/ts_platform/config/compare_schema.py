"""Pydantic schemas for compare runs."""

from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator, model_validator

from ts_platform.config.schema import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    StrictConfigModel,
    TrainingConfig,
    validate_safe_path_component,
)


class CompareModelConfig(StrictConfigModel):
    """One model entry inside a compare config."""

    name: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    alias: str | None = None

    @field_validator("alias")
    @classmethod
    def validate_alias(cls, value: str | None) -> str | None:
        """Require optional aliases to be safe path components."""

        if value is None:
            return None
        return validate_safe_path_component(value, field_name="model alias")


class CompareConfig(StrictConfigModel):
    """Root config for a multi-model compare run."""

    experiment: ExperimentConfig
    data: DataConfig
    models: list[CompareModelConfig] = Field(min_length=2)
    training: TrainingConfig
    evaluation: EvaluationConfig
    primary_metric: str | None = None
    continue_on_error: bool = True

    @model_validator(mode="after")
    def validate_primary_metric(self) -> CompareConfig:
        """Default and validate the primary metric used for ranking."""

        if self.primary_metric is None:
            self.primary_metric = self.evaluation.metrics[0]
        if self.primary_metric not in self.evaluation.metrics:
            msg = "primary_metric must be one of evaluation.metrics"
            raise ValueError(msg)
        return self
