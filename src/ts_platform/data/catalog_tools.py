"""Shared helpers for catalog profiling and config generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ts_platform.config.schema import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    PlatformConfig,
    ScalerConfig,
    TrainingConfig,
)
from ts_platform.data.catalog import DatasetMetadata
from ts_platform.data.catalog_loader import load_dataset_catalog
from ts_platform.data.profile import DatasetProfile, profile_csv_dataset


def profile_catalog(
    catalog_path: str | Path,
    *,
    input_len: int | None,
    output_len: int | None,
) -> list[dict[str, Any]]:
    """Profile CSV entries from a local dataset catalog."""

    return [
        profile_catalog_entry(
            metadata,
            input_len=input_len,
            output_len=output_len,
        )
        for metadata in load_dataset_catalog(catalog_path)
    ]


def profile_catalog_entry(
    metadata: DatasetMetadata,
    *,
    input_len: int | None,
    output_len: int | None,
) -> dict[str, Any]:
    """Profile one catalog entry or return a warning stub."""

    if metadata.dataset_type != "csv":
        return profile_stub(metadata, warning=f"unsupported dataset_type: {metadata.dataset_type}")
    if metadata.path is None:
        return profile_stub(metadata, warning="missing path")
    if metadata.target_cols is None:
        return profile_stub(metadata, warning="missing target_cols")
    return profile_csv_dataset(
        path=metadata.path,
        target_cols=metadata.target_cols,
        timestamp_col=metadata.timestamp_col,
        input_len=input_len,
        output_len=output_len,
        name=metadata.name,
    ).to_dict()


def profile_stub(metadata: DatasetMetadata, *, warning: str) -> dict[str, Any]:
    """Return a serializable profile payload for unsupported catalog entries."""

    return DatasetProfile(
        name=metadata.name,
        dataset_type=metadata.dataset_type,
        path=metadata.path or "",
        exists=False,
        row_count=0,
        column_count=0,
        columns=[],
        timestamp_col=metadata.timestamp_col,
        start_timestamp=None,
        end_timestamp=None,
        target_cols=metadata.target_cols or [],
        target_missing_counts={},
        target_dtypes={},
        duplicate_timestamp_count=None,
        inferred_frequency=None,
        min_required_rows=None,
        can_build_windows=False,
        warnings=[warning],
    ).to_dict()


def config_from_catalog(
    catalog_path: str | Path,
    *,
    dataset_name: str,
    input_len: int,
    output_len: int,
    model_name: str,
    epochs: int,
    batch_size: int,
) -> PlatformConfig:
    """Generate a training config from one catalog entry."""

    metadata = find_catalog_metadata(load_dataset_catalog(catalog_path), dataset_name)
    return config_from_catalog_metadata(
        metadata,
        input_len=input_len,
        output_len=output_len,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
    )


def save_generated_config(config: PlatformConfig, output_path: str | Path) -> Path:
    """Write a generated config as YAML."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    return path


def find_catalog_metadata(metadata: list[DatasetMetadata], name: str) -> DatasetMetadata:
    """Find the most recent catalog entry with a matching name."""

    normalized = name.strip().lower()
    for item in reversed(metadata):
        if item.name.strip().lower() == normalized:
            return item
    msg = f"unknown dataset metadata: {normalized}"
    raise KeyError(msg)


def config_from_catalog_metadata(
    metadata: DatasetMetadata,
    *,
    input_len: int,
    output_len: int,
    model_name: str,
    epochs: int,
    batch_size: int,
) -> PlatformConfig:
    """Build a train config from one validated catalog metadata row."""

    if metadata.dataset_type != "csv":
        msg = "make-config-from-catalog only supports csv datasets"
        raise ValueError(msg)
    if metadata.path is None:
        msg = "csv catalog entry must include path"
        raise ValueError(msg)
    if metadata.target_cols is None:
        msg = "csv catalog entry must include target_cols"
        raise ValueError(msg)
    data_params: dict[str, Any] = {
        "path": metadata.path,
        "target_cols": metadata.target_cols,
        "missing_policy": "error",
        "sort_by_time": True,
    }
    if metadata.timestamp_col is not None:
        data_params["timestamp_col"] = metadata.timestamp_col
    return PlatformConfig(
        experiment=ExperimentConfig(
            name=f"train_{metadata.name}_{model_name}",
            output_dir=Path("runs"),
            overwrite=True,
        ),
        data=DataConfig(
            name="csv",
            input_len=input_len,
            output_len=output_len,
            batch_size=batch_size,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            scaler=ScalerConfig(name="standard"),
            params=data_params,
        ),
        model=ModelConfig(name=model_name),
        training=TrainingConfig(
            epochs=epochs,
            learning_rate=0.01,
            device="cpu",
            optimizer="adam",
            loss="mse",
            checkpoint_every=1,
        ),
        evaluation=EvaluationConfig(
            metrics=["mae", "mse", "rmse", "wape"],
            include_scaled_metrics=False,
        ),
    )
