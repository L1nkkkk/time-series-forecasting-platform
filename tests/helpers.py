from __future__ import annotations

from pathlib import Path

from ts_platform.config.schema import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PlatformConfig,
    ScalerConfig,
    TrainingConfig,
)


def tiny_config(
    output_dir: Path,
    *,
    name: str = "tiny",
    epochs: int = 1,
    overwrite: bool = True,
    seed: int = 7,
    val_ratio: float = 0.15,
    resume_from: Path | None = None,
) -> PlatformConfig:
    test_ratio = 0.15 if val_ratio > 0 else 0.2
    train_ratio = 1.0 - val_ratio - test_ratio
    return PlatformConfig(
        experiment=ExperimentConfig(
            name=name,
            output_dir=output_dir,
            seed=seed,
            overwrite=overwrite,
        ),
        data=DataConfig(
            name="synthetic",
            input_len=6,
            output_len=2,
            batch_size=8,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            scaler=ScalerConfig(name="standard"),
            params={"length": 80, "num_features": 1, "noise_std": 0.0},
        ),
        model=ModelConfig(name="linear"),
        training=TrainingConfig(
            epochs=epochs,
            learning_rate=0.01,
            device="cpu",
            resume_from=resume_from,
        ),
    )


def tiny_feature_config(
    output_dir: Path,
    *,
    name: str = "feature_tiny",
    model_name: str = "linear",
    model_params: dict[str, object] | None = None,
    epochs: int = 1,
    overwrite: bool = True,
    seed: int = 7,
    resume_from: Path | None = None,
    target_cols: list[str] | None = None,
    feature_cols: list[str] | None = None,
) -> PlatformConfig:
    return PlatformConfig(
        experiment=ExperimentConfig(
            name=name,
            output_dir=output_dir,
            seed=seed,
            overwrite=overwrite,
        ),
        data=DataConfig(
            name="csv",
            input_len=4,
            output_len=2,
            batch_size=4,
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            scaler=ScalerConfig(name="standard"),
            params={
                "path": "tests/fixtures/tiny_series_with_features.csv",
                "timestamp_col": "timestamp",
                "target_cols": target_cols or ["value"],
                "feature_cols": feature_cols or ["temperature", "holiday"],
                "missing_policy": "error",
                "sort_by_time": True,
            },
        ),
        model=ModelConfig(name=model_name, params=model_params or {}),
        training=TrainingConfig(
            epochs=epochs,
            learning_rate=0.01,
            device="cpu",
            resume_from=resume_from,
        ),
    )
