"""Compare configuration loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import yaml

from ts_platform.config.compare_schema import CompareConfig


def load_compare_config(path: str | Path) -> CompareConfig:
    """Load a YAML or JSON compare config and validate it."""

    config_path = Path(path)
    if not config_path.exists():
        msg = f"Compare config file does not exist: {config_path}"
        raise FileNotFoundError(msg)

    suffix = config_path.suffix.lower()
    raw_text = config_path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        raw_data = yaml.safe_load(raw_text)
    elif suffix == ".json":
        raw_data = json.loads(raw_text)
    else:
        msg = f"Unsupported compare config extension: {suffix}"
        raise ValueError(msg)

    if not isinstance(raw_data, dict):
        msg = "Compare config root must be a mapping"
        raise ValueError(msg)
    return CompareConfig.model_validate(cast(dict[str, Any], raw_data))


def save_compare_config_snapshot(config: CompareConfig, path: str | Path) -> None:
    """Save a validated compare config as YAML."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="json")
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
