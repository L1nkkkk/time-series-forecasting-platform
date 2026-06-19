"""CLI entry point."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ts_platform.api.services.experiment_store import ExperimentStore
from ts_platform.data import DATASET_CATALOG, DATASET_REGISTRY, register_dataset_catalog
from ts_platform.models.registry import registered_model_names
from ts_platform.runner.comparer import CompareRunner
from ts_platform.runner.trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(prog="ts-platform")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="Run a training config")
    train_parser.add_argument("--config", required=True, help="Path to YAML or JSON config")

    compare_parser = subparsers.add_parser("compare", help="Run a multi-model compare config")
    compare_parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML or JSON compare config",
    )

    datasets_parser = subparsers.add_parser("list-datasets", help="List registered datasets")
    datasets_parser.add_argument(
        "--catalog",
        action="append",
        default=[],
        help="Optional local dataset catalog YAML to load before listing",
    )

    subparsers.add_parser("list-models", help="List registered models")

    show_results_parser = subparsers.add_parser("show-results", help="Show run results JSON")
    show_results_parser.add_argument("--experiment", required=True, help="Experiment name")
    show_results_parser.add_argument("--run", default="latest", help="Run id or latest")
    show_results_parser.add_argument(
        "--runs-root",
        default="runs",
        help="Runs root to read from",
    )

    show_leaderboard_parser = subparsers.add_parser(
        "show-leaderboard",
        help="Show compare leaderboard JSON",
    )
    show_leaderboard_parser.add_argument("--experiment", required=True, help="Experiment name")
    show_leaderboard_parser.add_argument("--run", default="latest", help="Run id or latest")
    show_leaderboard_parser.add_argument(
        "--runs-root",
        default="runs",
        help="Runs root to read from",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        training_result = Trainer.from_config_path(args.config).run()
        print(json.dumps(training_result.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "compare":
        compare_result = CompareRunner.from_config_path(args.config).run()
        print(json.dumps(compare_result.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "list-datasets":
        for catalog_path in args.catalog:
            register_dataset_catalog(catalog_path)
        payload = {"datasets": DATASET_CATALOG.list(), "names": DATASET_REGISTRY.names()}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "list-models":
        print(json.dumps({"models": registered_model_names()}, indent=2, sort_keys=True))
        return 0
    if args.command == "show-results":
        results_payload = ExperimentStore(Path(args.runs_root)).read_results(
            args.experiment,
            args.run,
        )
        print(json.dumps(results_payload, indent=2, sort_keys=True))
        return 0
    if args.command == "show-leaderboard":
        leaderboard_payload = ExperimentStore(Path(args.runs_root)).read_leaderboard(
            args.experiment,
            args.run,
        )
        print(json.dumps(leaderboard_payload, indent=2, sort_keys=True))
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
