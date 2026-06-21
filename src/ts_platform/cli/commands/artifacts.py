"""Artifact inspection commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from ts_platform.api.services.artifact_service import ArtifactService
from ts_platform.api.services.experiment_store import ExperimentStore
from ts_platform.cli.utils import print_json, read_text_artifact


def register(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register artifact commands."""

    show_artifacts_parser = subparsers.add_parser(
        "show-artifacts",
        help="Show run artifact manifest JSON",
    )
    _add_run_lookup_args(show_artifacts_parser)
    show_artifacts_parser.set_defaults(handler=handle_show_artifacts)

    show_artifact_parser = subparsers.add_parser(
        "show-artifact",
        help="Show one run artifact file",
    )
    _add_run_lookup_args(show_artifact_parser)
    show_artifact_parser.add_argument("--artifact", required=True, help="Artifact name")
    show_artifact_parser.add_argument(
        "--output",
        help="Optional file path to write the artifact content to",
    )
    show_artifact_parser.set_defaults(handler=handle_show_artifact)


def handle_show_artifacts(args: argparse.Namespace) -> int:
    """Show run artifact manifest JSON."""

    artifacts_payload = ExperimentStore(Path(args.runs_root)).read_artifacts(
        args.experiment,
        args.run,
    )
    print_json(artifacts_payload)
    return 0


def handle_show_artifact(args: argparse.Namespace) -> int:
    """Show one run artifact file."""

    artifact = ArtifactService(Path(args.runs_root)).resolve_artifact(
        args.experiment,
        args.run,
        args.artifact,
    )
    content = read_text_artifact(artifact.path)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
    else:
        print(content, end="")
    return 0


def _add_run_lookup_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--run", default="latest", help="Run id or latest")
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Runs root to read from",
    )
