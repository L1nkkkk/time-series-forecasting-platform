"""Training command."""

from __future__ import annotations

import argparse

from ts_platform.cli.utils import print_json
from ts_platform.runner.trainer import Trainer


def register(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the train command."""

    parser = subparsers.add_parser("train", help="Run a training config")
    parser.add_argument("--config", required=True, help="Path to YAML or JSON config")
    parser.set_defaults(handler=handle_train)


def handle_train(args: argparse.Namespace) -> int:
    """Run a training config."""

    training_result = Trainer.from_config_path(args.config).run()
    print_json(training_result.to_dict())
    return 0
