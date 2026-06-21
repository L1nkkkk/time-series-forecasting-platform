"""Dataset discovery, profiling, and config generation commands."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from ts_platform.cli.utils import print_json
from ts_platform.config.schema import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    PlatformConfig,
    ScalerConfig,
    TrainingConfig,
)
from ts_platform.data import (
    DATASET_CATALOG,
    DATASET_REGISTRY,
    DatasetMetadata,
    load_dataset_catalog,
    profile_csv_dataset,
    register_dataset_catalog,
)
from ts_platform.data.profile import DatasetProfile


def register(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register dataset commands."""

    datasets_parser = subparsers.add_parser("list-datasets", help="List registered datasets")
    datasets_parser.add_argument(
        "--catalog",
        action="append",
        default=[],
        help="Optional local dataset catalog YAML to load before listing",
    )
    datasets_parser.set_defaults(handler=handle_list_datasets)

    profile_dataset_parser = subparsers.add_parser(
        "profile-dataset",
        help="Profile one local CSV dataset",
    )
    profile_dataset_parser.add_argument("--path", required=True, help="Local CSV path")
    profile_dataset_parser.add_argument(
        "--target-cols",
        nargs="+",
        required=True,
        help="One or more target column names",
    )
    profile_dataset_parser.add_argument("--timestamp-col", help="Optional timestamp column")
    profile_dataset_parser.add_argument("--input-len", type=int, help="Optional input length")
    profile_dataset_parser.add_argument("--output-len", type=int, help="Optional output length")
    profile_dataset_parser.add_argument("--name", help="Optional dataset name")
    profile_dataset_parser.set_defaults(handler=handle_profile_dataset)

    profile_catalog_parser = subparsers.add_parser(
        "profile-catalog",
        help="Profile CSV entries in a local dataset catalog",
    )
    profile_catalog_parser.add_argument("--catalog", required=True, help="Catalog YAML path")
    profile_catalog_parser.add_argument("--input-len", type=int, help="Optional input length")
    profile_catalog_parser.add_argument("--output-len", type=int, help="Optional output length")
    profile_catalog_parser.set_defaults(handler=handle_profile_catalog)

    make_config_parser = subparsers.add_parser(
        "make-config-from-catalog",
        help="Generate a training config from one catalog entry",
    )
    make_config_parser.add_argument("--catalog", required=True, help="Catalog YAML path")
    make_config_parser.add_argument("--dataset", required=True, help="Catalog dataset name")
    make_config_parser.add_argument("--output", required=True, help="Output YAML path")
    make_config_parser.add_argument("--input-len", required=True, type=int, help="Input length")
    make_config_parser.add_argument("--output-len", required=True, type=int, help="Output length")
    make_config_parser.add_argument("--model", required=True, help="Registered model name")
    make_config_parser.add_argument("--epochs", required=True, type=int, help="Training epochs")
    make_config_parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    make_config_parser.set_defaults(handler=handle_make_config_from_catalog)


def handle_list_datasets(args: argparse.Namespace) -> int:
    """List registered datasets."""

    for catalog_path in args.catalog:
        register_dataset_catalog(catalog_path)
    payload = {"datasets": DATASET_CATALOG.list(), "names": DATASET_REGISTRY.names()}
    print_json(payload)
    return 0


def handle_profile_dataset(args: argparse.Namespace) -> int:
    """Profile one local CSV dataset."""

    profile = profile_csv_dataset(
        path=args.path,
        target_cols=args.target_cols,
        timestamp_col=args.timestamp_col,
        input_len=args.input_len,
        output_len=args.output_len,
        name=args.name,
    )
    print_json(profile.to_dict())
    return 0


def handle_profile_catalog(args: argparse.Namespace) -> int:
    """Profile CSV entries from a catalog."""

    profiles = [
        _profile_catalog_entry(
            metadata,
            input_len=args.input_len,
            output_len=args.output_len,
        )
        for metadata in load_dataset_catalog(args.catalog)
    ]
    print_json({"profiles": profiles})
    return 0


def handle_make_config_from_catalog(args: argparse.Namespace) -> int:
    """Generate a training config from one catalog entry."""

    metadata = _find_catalog_metadata(load_dataset_catalog(args.catalog), args.dataset)
    config = _config_from_catalog_metadata(
        metadata,
        input_len=args.input_len,
        output_len=args.output_len,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    payload = {
        "output": str(output_path),
        "config": config.model_dump(mode="json"),
    }
    print_json(payload)
    return 0


def _profile_catalog_entry(
    metadata: DatasetMetadata,
    *,
    input_len: int | None,
    output_len: int | None,
) -> dict[str, Any]:
    if metadata.dataset_type != "csv":
        return _profile_stub(metadata, warning=f"unsupported dataset_type: {metadata.dataset_type}")
    if metadata.path is None:
        return _profile_stub(metadata, warning="missing path")
    if metadata.target_cols is None:
        return _profile_stub(metadata, warning="missing target_cols")
    return profile_csv_dataset(
        path=metadata.path,
        target_cols=metadata.target_cols,
        timestamp_col=metadata.timestamp_col,
        input_len=input_len,
        output_len=output_len,
        name=metadata.name,
    ).to_dict()


def _profile_stub(metadata: DatasetMetadata, *, warning: str) -> dict[str, Any]:
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


def _find_catalog_metadata(metadata: list[DatasetMetadata], name: str) -> DatasetMetadata:
    normalized = name.strip().lower()
    for item in reversed(metadata):
        if item.name.strip().lower() == normalized:
            return item
    msg = f"unknown dataset metadata: {normalized}"
    raise KeyError(msg)


def _config_from_catalog_metadata(
    metadata: DatasetMetadata,
    *,
    input_len: int,
    output_len: int,
    model_name: str,
    epochs: int,
    batch_size: int,
) -> PlatformConfig:
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
