"""CLI command registration."""

from __future__ import annotations

import argparse

from ts_platform.cli.commands import (
    artifacts,
    compare,
    datasets,
    experiments,
    jobs,
    models,
    retry,
    train,
    worker,
)


def register_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register all CLI commands."""

    train.register(subparsers)
    compare.register(subparsers)
    datasets.register(subparsers)
    models.register(subparsers)
    experiments.register(subparsers)
    artifacts.register(subparsers)
    jobs.register(subparsers)
    retry.register(subparsers)
    worker.register(subparsers)
