"""Model discovery command."""

from __future__ import annotations

import argparse

from ts_platform.cli.utils import print_json
from ts_platform.models.registry import registered_model_names


def register(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register model discovery commands."""

    parser = subparsers.add_parser("list-models", help="List registered models")
    parser.set_defaults(handler=handle_list_models)


def handle_list_models(args: argparse.Namespace) -> int:
    """List registered models."""

    del args
    print_json({"models": registered_model_names()})
    return 0
