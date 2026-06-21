"""Shared CLI helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def print_json(payload: Any) -> None:
    """Print a stable JSON payload to stdout."""

    print(json.dumps(payload, indent=2, sort_keys=True))


def read_text_artifact(path: Path) -> str:
    """Read a resolved text artifact as UTF-8."""

    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        msg = f"artifact is not UTF-8 text: {path}"
        raise ValueError(msg) from exc
