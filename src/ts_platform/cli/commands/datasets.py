"""Dataset discovery, profiling, and config generation commands."""

from __future__ import annotations

import argparse

from ts_platform.cli.utils import print_json
from ts_platform.data import (
    DATASET_CATALOG,
    DATASET_REGISTRY,
    profile_csv_dataset,
    register_dataset_catalog,
)
from ts_platform.data.catalog_loader import load_dataset_catalog
from ts_platform.data.catalog_tools import (
    config_from_catalog_metadata,
    find_catalog_metadata,
    profile_catalog_entry,
    save_generated_config,
)


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
        profile_catalog_entry(
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

    metadata = find_catalog_metadata(load_dataset_catalog(args.catalog), args.dataset)
    config = config_from_catalog_metadata(
        metadata,
        input_len=args.input_len,
        output_len=args.output_len,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    output_path = save_generated_config(config, args.output)
    payload = {
        "output": str(output_path),
        "config": config.model_dump(mode="json"),
    }
    print_json(payload)
    return 0
