"""Experiment result commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from ts_platform.api.services.experiment_store import ExperimentStore
from ts_platform.cli.utils import print_json


def register(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register experiment result commands."""

    show_results_parser = subparsers.add_parser("show-results", help="Show run results JSON")
    _add_run_lookup_args(show_results_parser)
    show_results_parser.set_defaults(handler=handle_show_results)

    show_leaderboard_parser = subparsers.add_parser(
        "show-leaderboard",
        help="Show compare leaderboard JSON",
    )
    _add_run_lookup_args(show_leaderboard_parser)
    show_leaderboard_parser.set_defaults(handler=handle_show_leaderboard)


def handle_show_results(args: argparse.Namespace) -> int:
    """Show run results JSON."""

    results_payload = ExperimentStore(Path(args.runs_root)).read_results(
        args.experiment,
        args.run,
    )
    print_json(results_payload)
    return 0


def handle_show_leaderboard(args: argparse.Namespace) -> int:
    """Show compare leaderboard JSON."""

    leaderboard_payload = ExperimentStore(Path(args.runs_root)).read_leaderboard(
        args.experiment,
        args.run,
    )
    print_json(leaderboard_payload)
    return 0


def _add_run_lookup_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--run", default="latest", help="Run id or latest")
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Runs root to read from",
    )
