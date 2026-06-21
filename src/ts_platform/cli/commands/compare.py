"""Compare command."""

from __future__ import annotations

import argparse

from ts_platform.cli.utils import print_json
from ts_platform.runner.comparer import CompareRunner


def register(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the compare command."""

    parser = subparsers.add_parser("compare", help="Run a multi-model compare config")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML or JSON compare config",
    )
    parser.set_defaults(handler=handle_compare)


def handle_compare(args: argparse.Namespace) -> int:
    """Run a multi-model compare config."""

    compare_result = CompareRunner.from_config_path(args.config).run()
    print_json(compare_result.to_dict())
    return 0
