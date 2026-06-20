"""Factories for job store and runner backends."""

from __future__ import annotations

from ts_platform.api.jobs.base import JobStoreProtocol
from ts_platform.api.jobs.runner import JobRunner
from ts_platform.api.jobs.sqlite_store import SQLiteJobStore
from ts_platform.api.jobs.store import JsonJobStore
from ts_platform.api.settings import APISettings


def build_job_store(settings: APISettings) -> JobStoreProtocol:
    """Build a job store backend from API settings."""

    validate_job_execution_settings(settings)
    if settings.job_backend == "json":
        return JsonJobStore(settings.jobs_root)
    if settings.job_backend == "sqlite":
        return SQLiteJobStore(settings.jobs_root, settings.sqlite_jobs_db_path)
    msg = f"unsupported job backend: {settings.job_backend}"
    raise ValueError(msg)


def build_job_runner(settings: APISettings) -> JobRunner:
    """Build a local job runner with the configured store backend."""

    validate_job_execution_settings(settings)
    return JobRunner(store=build_job_store(settings), runs_root=settings.runs_root)


def validate_job_execution_settings(settings: APISettings) -> None:
    """Validate supported job backend and execution mode combinations."""

    if settings.job_execution_mode == "external_worker" and settings.job_backend != "sqlite":
        msg = "external_worker job execution mode requires sqlite job backend"
        raise ValueError(msg)
    if settings.job_execution_mode not in {"in_process", "external_worker"}:
        msg = f"unsupported job execution mode: {settings.job_execution_mode}"
        raise ValueError(msg)
