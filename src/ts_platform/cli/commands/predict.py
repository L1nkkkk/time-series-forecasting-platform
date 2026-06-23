"""Prediction command for model exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ts_platform.cli.utils import print_json
from ts_platform.runner.predictor import predict_from_model_export


def register(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register model export prediction commands."""

    parser = subparsers.add_parser("predict", help="Run inference from a model export")
    parser.add_argument("--model-export", required=True, help="Path to model_export.pt")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--values-json",
        help="JSON array shaped [input_len, input_dim] or [batch, input_len, input_dim]",
    )
    input_group.add_argument(
        "--values-file",
        help="Path to a JSON file containing values for prediction",
    )
    parser.set_defaults(handler=handle_predict)


def handle_predict(args: argparse.Namespace) -> int:
    """Run prediction from a model export."""

    values = _load_values(args)
    print_json(
        predict_from_model_export(
            args.model_export,
            values=values,
        )
    )
    return 0


def _load_values(args: argparse.Namespace) -> list[Any]:
    raw_text = args.values_json or Path(args.values_file).read_text(encoding="utf-8")
    payload = json.loads(raw_text)
    if not isinstance(payload, list):
        msg = "prediction values must be a JSON array"
        raise ValueError(msg)
    return payload
