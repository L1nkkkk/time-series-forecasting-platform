"""API runtime settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

JobBackend = Literal["json", "sqlite"]
JobExecutionMode = Literal["in_process", "external_worker"]


@dataclass(frozen=True)
class APISettings:
    """Small settings object for the synchronous demo API."""

    runs_root: Path = Path("runs")
    api_key: str | None = None
    auth_exempt_paths: tuple[str, ...] = ("/health", "/ui", "/ui/", "/ui/static")
    rate_limit_requests_per_minute: int | None = None
    max_request_body_bytes: int | None = 10 * 1024 * 1024
    audit_log_path: Path | None = None
    dataset_catalog_glob: str = "configs/datasets/*.yaml"
    artifact_max_bytes: int = 5 * 1024 * 1024
    allow_checkpoint_download: bool = False
    artifact_allowed_kinds: tuple[str, ...] = ("json", "yaml", "csv", "log", "model")
    job_backend: JobBackend = "json"
    job_execution_mode: JobExecutionMode = "in_process"
    sqlite_jobs_db_path: Path = Path("runs/jobs.sqlite3")
    jobs_root: Path = Path("runs/jobs")

    @classmethod
    def from_env(cls) -> APISettings:
        """Build settings from environment variables with demo-safe defaults."""

        return cls(
            api_key=_optional_env("TS_PLATFORM_API_KEY"),
            rate_limit_requests_per_minute=_optional_int_env("TS_PLATFORM_RATE_LIMIT_PER_MINUTE"),
            max_request_body_bytes=_optional_int_env(
                "TS_PLATFORM_MAX_REQUEST_BODY_BYTES",
                default=10 * 1024 * 1024,
            ),
            audit_log_path=_optional_path_env("TS_PLATFORM_AUDIT_LOG_PATH"),
        )


def _optional_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _optional_int_env(name: str, *, default: int | None = None) -> int | None:
    value = _optional_env(name)
    if value is None:
        return default
    return int(value)


def _optional_path_env(name: str) -> Path | None:
    value = _optional_env(name)
    return Path(value) if value is not None else None
