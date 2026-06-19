"""API runtime settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class APISettings:
    """Small settings object for the synchronous demo API."""

    runs_root: Path = Path("runs")
    dataset_catalog_glob: str = "configs/datasets/*.yaml"
