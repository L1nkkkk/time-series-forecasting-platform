"""CLI entry point."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from ts_platform.runner.trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(prog="ts-platform")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="Run a training config")
    train_parser.add_argument("--config", required=True, help="Path to YAML or JSON config")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        result = Trainer.from_config_path(args.config).run()
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
