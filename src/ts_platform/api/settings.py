"""API runtime settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

JobBackend = Literal["json", "sqlite"]


@dataclass(frozen=True)
class APISettings:
    """Small settings object for the synchronous demo API."""

    runs_root: Path = Path("runs")
    dataset_catalog_glob: str = "configs/datasets/*.yaml"
    artifact_max_bytes: int = 5 * 1024 * 1024
    allow_checkpoint_download: bool = False
    artifact_allowed_kinds: tuple[str, ...] = ("json", "yaml", "csv", "log")
    job_backend: JobBackend = "json"
    sqlite_jobs_db_path: Path = Path("runs/jobs.sqlite3")
    jobs_root: Path = Path("runs/jobs")
