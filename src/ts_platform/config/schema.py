"""Pydantic schemas for platform configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictConfigModel(BaseModel):
    """Base class that rejects unknown fields in user configs."""

    model_config = ConfigDict(extra="forbid")


class ExperimentConfig(StrictConfigModel):
    """Experiment output and reproducibility settings."""

    name: str = Field(min_length=1)
    output_dir: Path = Path("runs")
    seed: int = 42
    overwrite: bool = False


class ScalerConfig(StrictConfigModel):
    """Scaler selection and optional parameters."""

    name: str = "standard"
    params: dict[str, Any] = Field(default_factory=dict)


class DataConfig(StrictConfigModel):
    """Dataset and dataloader settings."""

    name: str = Field(min_length=1)
    input_len: int = Field(gt=0)
    output_len: int = Field(gt=0)
    batch_size: int = Field(default=32, gt=0)
    train_ratio: float = Field(default=0.7, gt=0.0, lt=1.0)
    val_ratio: float = Field(default=0.15, ge=0.0, lt=1.0)
    test_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)
    cache_dir: Path = Path("data/cache")
    scaler: ScalerConfig = Field(default_factory=ScalerConfig)
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_split_ratios(self) -> DataConfig:
        """Ensure train/validation/test ratios define a full split."""

        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            msg = "train_ratio + val_ratio + test_ratio must equal 1.0"
            raise ValueError(msg)
        return self


class ModelConfig(StrictConfigModel):
    """Model registry name and model-specific settings."""

    name: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class TrainingConfig(StrictConfigModel):
    """Training loop settings."""

    epochs: int = Field(default=1, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    device: str = "cpu"
    optimizer: Literal["adam", "sgd"] = "adam"
    loss: Literal["mse", "mae"] = "mse"
    checkpoint_every: int = Field(default=1, ge=0)
    num_workers: int = Field(default=0, ge=0)
    resume_from: Path | None = None


class EvaluationConfig(StrictConfigModel):
    """Evaluation metric settings."""

    metrics: list[str] = Field(default_factory=lambda: ["mae", "mse", "rmse", "mape", "wape"])

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, value: list[str]) -> list[str]:
        """Require at least one metric name."""

        if not value:
            msg = "evaluation.metrics must contain at least one metric"
            raise ValueError(msg)
        return value


class PlatformConfig(StrictConfigModel):
    """Root config for a training run."""

    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
