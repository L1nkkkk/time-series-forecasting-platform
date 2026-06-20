"""Job API routes."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import Any, NoReturn

from fastapi import APIRouter, HTTPException

from ts_platform.api.jobs.factory import build_job_runner, validate_job_execution_settings
from ts_platform.api.jobs.runner import JobRunner
from ts_platform.api.jobs.store import (
    JobNotFoundError,
    JobStoreError,
    UnsafeJobIdError,
)
from ts_platform.api.settings import APISettings
from ts_platform.config.compare_schema import CompareConfig
from ts_platform.config.schema import PlatformConfig

router = APIRouter()
API_SETTINGS = APISettings()
RUNS_ROOT = API_SETTINGS.runs_root
JOBS_ROOT = API_SETTINGS.jobs_root
SQLITE_JOBS_DB_PATH = API_SETTINGS.sqlite_jobs_db_path
_JOB_RUNNER: JobRunner | None = None
_JOB_RUNNER_LOCK = RLock()


@router.post("/jobs/train")
def submit_train_job(config: PlatformConfig) -> dict[str, Any]:
    """Submit a training job to the local runner."""

    try:
        settings = _runner_settings()
        validate_job_execution_settings(settings)
        runner = get_job_runner()
        if settings.job_execution_mode == "external_worker":
            return runner.enqueue_train(config).to_dict()
        return runner.submit_train(config).to_dict()
    except (UnsafeJobIdError, JobStoreError, ValueError) as exc:
        _raise_job_error(exc)


@router.post("/jobs/compare")
def submit_compare_job(config: CompareConfig) -> dict[str, Any]:
    """Submit a compare job to the local runner."""

    try:
        settings = _runner_settings()
        validate_job_execution_settings(settings)
        runner = get_job_runner()
        if settings.job_execution_mode == "external_worker":
            return runner.enqueue_compare(config).to_dict()
        return runner.submit_compare(config).to_dict()
    except (UnsafeJobIdError, JobStoreError, ValueError) as exc:
        _raise_job_error(exc)


@router.get("/jobs")
def list_jobs() -> dict[str, list[dict[str, Any]]]:
    """List persisted local jobs newest first."""

    try:
        return {"jobs": [job.to_dict() for job in get_job_runner().store.list_jobs()]}
    except JobStoreError as exc:
        _raise_job_error(exc)


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    """Read one persisted local job."""

    try:
        return get_job_runner().store.get_job(job_id).to_dict()
    except (UnsafeJobIdError, JobNotFoundError, JobStoreError) as exc:
        _raise_job_error(exc)


@router.get("/jobs/{job_id}/result")
def get_job_result(job_id: str) -> dict[str, Any]:
    """Read the result JSON for a succeeded job."""

    runner = get_job_runner()
    try:
        job = runner.store.get_job(job_id)
    except (UnsafeJobIdError, JobNotFoundError, JobStoreError) as exc:
        _raise_job_error(exc)
    if job.status != "succeeded":
        raise HTTPException(
            status_code=409,
            detail={"status": job.status, "error": job.error},
        )
    if job.result_path is None:
        raise HTTPException(status_code=404, detail="job result path is not recorded")
    return _read_json_file(Path(job.result_path), runs_root=runner.runs_root)


@router.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: str) -> dict[str, Any]:
    """Return a JSON wrapper containing the job error and train log when available."""

    runner = get_job_runner()
    try:
        job = runner.store.get_job(job_id)
    except (UnsafeJobIdError, JobNotFoundError, JobStoreError) as exc:
        _raise_job_error(exc)

    log_text: str | None = None
    log_path: str | None = None
    if job.result_path is not None:
        candidate = Path(job.result_path).parent / "train.log"
        _assert_inside_root(candidate, runner.runs_root)
        if candidate.exists():
            log_path = str(candidate)
            log_text = candidate.read_text(encoding="utf-8")
    return {
        "job_id": job.job_id,
        "status": job.status,
        "error": job.error,
        "log_path": log_path,
        "log": log_text,
    }


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    """Request cancellation for a queued or running job."""

    try:
        return get_job_runner().store.request_cancel(job_id).to_dict()
    except (UnsafeJobIdError, JobNotFoundError, JobStoreError) as exc:
        _raise_job_error(exc)


def get_job_runner() -> JobRunner:
    """Return the app-level lazy local job runner."""

    global _JOB_RUNNER
    with _JOB_RUNNER_LOCK:
        if _JOB_RUNNER is None:
            _JOB_RUNNER = build_job_runner(_runner_settings())
        return _JOB_RUNNER


def shutdown_job_runner() -> None:
    """Shut down and clear the app-level local job runner if it exists."""

    global _JOB_RUNNER
    with _JOB_RUNNER_LOCK:
        if _JOB_RUNNER is not None:
            _JOB_RUNNER.shutdown(wait=False)
            _JOB_RUNNER = None


def _read_json_file(path: Path, *, runs_root: Path) -> dict[str, Any]:
    _assert_inside_root(path, runs_root)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"job result does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        detail = f"job result is not valid JSON: {path}"
        raise HTTPException(status_code=500, detail=detail) from exc
    except OSError as exc:
        raise HTTPException(status_code=404, detail=f"job result cannot be read: {path}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail=f"job result is not a JSON object: {path}")
    return payload


def _assert_inside_root(path: Path, runs_root: Path) -> None:
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(Path(runs_root).resolve()):
        raise HTTPException(status_code=500, detail=f"job path escapes runs root: {path}")


def _runner_settings() -> APISettings:
    return replace(
        API_SETTINGS,
        runs_root=RUNS_ROOT,
        jobs_root=JOBS_ROOT,
        sqlite_jobs_db_path=SQLITE_JOBS_DB_PATH,
    )


def _raise_job_error(
    exc: UnsafeJobIdError | JobNotFoundError | JobStoreError | ValueError,
) -> NoReturn:
    if isinstance(exc, UnsafeJobIdError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, JobNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail=str(exc)) from exc
