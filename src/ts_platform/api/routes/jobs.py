"""Job API routes."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import Any, NoReturn

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ts_platform.api.jobs.base import JobStoreProtocol
from ts_platform.api.jobs.factory import (
    build_job_runner,
    build_job_store,
    validate_job_execution_settings,
)
from ts_platform.api.jobs.models import JobRecord
from ts_platform.api.jobs.retry import RetryPolicy
from ts_platform.api.jobs.runner import JobRunner
from ts_platform.api.jobs.sqlite_store import SQLiteJobStore
from ts_platform.api.jobs.store import (
    JobNotFoundError,
    JobStateConflictError,
    JobStoreError,
    UnsafeJobIdError,
)
from ts_platform.api.jobs.worker import JobWorker
from ts_platform.api.settings import APISettings, JobBackend
from ts_platform.config.compare_loader import load_compare_config
from ts_platform.config.compare_schema import CompareConfig
from ts_platform.config.loader import load_config
from ts_platform.config.schema import PlatformConfig, validate_safe_path_component

router = APIRouter()
API_SETTINGS = APISettings()
RUNS_ROOT = API_SETTINGS.runs_root
JOBS_ROOT = API_SETTINGS.jobs_root
SQLITE_JOBS_DB_PATH = API_SETTINGS.sqlite_jobs_db_path
_JOB_RUNNER: JobRunner | None = None
_JOB_RUNNER_LOCK = RLock()


class ConfigPathRequest(BaseModel):
    """Request body for config-path job submission."""

    config_path: str = Field(min_length=1)


class WorkerLoopRequest(BaseModel):
    """Bounded local worker loop settings."""

    max_jobs: int = Field(default=1, ge=1)
    max_idle_cycles: int = Field(default=1, ge=1)
    sleep_seconds: float = Field(default=1.0, ge=0.0)
    worker_id: str = Field(default="api_worker", min_length=1)
    sqlite_db: str | None = None
    jobs_root: str | None = None
    runs_root: str | None = None


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


@router.post("/jobs/train-config")
def submit_train_config_job(payload: ConfigPathRequest) -> dict[str, Any]:
    """Submit a training job from a local YAML or JSON config path."""

    try:
        return submit_train_job(load_config(payload.config_path))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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


@router.post("/jobs/compare-config")
def submit_compare_config_job(payload: ConfigPathRequest) -> dict[str, Any]:
    """Submit a compare job from a local YAML or JSON config path."""

    try:
        return submit_compare_job(load_compare_config(payload.config_path))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/jobs")
def list_jobs(
    job_backend: JobBackend | None = None,
    jobs_root: str | None = None,
    sqlite_db: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """List persisted local jobs newest first."""

    try:
        store = _job_store_from_overrides(
            job_backend=job_backend,
            jobs_root=jobs_root,
            sqlite_db=sqlite_db,
        )
        return {"jobs": [job.to_dict() for job in store.list_jobs()]}
    except JobStoreError as exc:
        _raise_job_error(exc)


@router.get("/jobs/stale")
def get_stale_jobs(
    older_than_seconds: int = 3600,
    jobs_root: str | None = None,
    sqlite_db: str | None = None,
) -> list[dict[str, Any]]:
    """Read stale SQLite running jobs without changing their status."""

    if older_than_seconds <= 0:
        raise HTTPException(status_code=400, detail="older_than_seconds must be > 0")
    try:
        store = _sqlite_store_from_overrides(
            jobs_root=jobs_root,
            sqlite_db=sqlite_db,
            detail="stale jobs require sqlite backend",
        )
        return [
            job.to_dict()
            for job in store.list_stale_running_jobs(older_than_seconds=older_than_seconds)
        ]
    except JobStoreError as exc:
        _raise_job_error(exc)


@router.post("/jobs/stale/timeout")
def timeout_stale_jobs(
    older_than_seconds: int = 3600,
    reason: str | None = None,
    jobs_root: str | None = None,
    sqlite_db: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Mark stale running SQLite jobs timed out."""

    if older_than_seconds <= 0:
        raise HTTPException(status_code=400, detail="older_than_seconds must be > 0")
    try:
        store = _sqlite_store_from_overrides(
            jobs_root=jobs_root,
            sqlite_db=sqlite_db,
            detail="job stale timeout require sqlite backend",
        )
        return {
            "timed_out": [
                job.to_dict()
                for job in store.mark_stale_running_jobs_timed_out(
                    older_than_seconds=older_than_seconds,
                    reason=reason,
                )
            ]
        }
    except JobStoreError as exc:
        _raise_job_error(exc)


@router.get("/jobs/{job_id}/events")
def get_job_events(
    job_id: str,
    jobs_root: str | None = None,
    sqlite_db: str | None = None,
) -> list[dict[str, Any]]:
    """Read SQLite audit events for one job."""

    try:
        store = _sqlite_store_from_overrides(
            jobs_root=jobs_root,
            sqlite_db=sqlite_db,
            detail="job events require sqlite backend",
        )
        store.get_job(job_id)
        return store.list_events(job_id)
    except (UnsafeJobIdError, JobNotFoundError, JobStoreError) as exc:
        _raise_job_error(exc)


@router.get("/jobs/{job_id}/attempts")
def get_job_attempts(
    job_id: str,
    jobs_root: str | None = None,
    sqlite_db: str | None = None,
) -> list[dict[str, Any]]:
    """Read SQLite worker attempts for one job."""

    try:
        store = _sqlite_store_from_overrides(
            jobs_root=jobs_root,
            sqlite_db=sqlite_db,
            detail="job attempts require sqlite backend",
        )
        store.get_job(job_id)
        return store.list_attempts(job_id)
    except (UnsafeJobIdError, JobNotFoundError, JobStoreError) as exc:
        _raise_job_error(exc)


@router.post("/jobs/{job_id}/timeout")
def timeout_job(
    job_id: str,
    reason: str | None = None,
    jobs_root: str | None = None,
    sqlite_db: str | None = None,
) -> dict[str, Any]:
    """Explicitly mark one SQLite job timed out."""

    try:
        store = _sqlite_store_from_overrides(
            jobs_root=jobs_root,
            sqlite_db=sqlite_db,
            detail="job timeout requires sqlite backend",
        )
        return store.mark_timed_out(job_id, reason=reason).to_dict()
    except (UnsafeJobIdError, JobNotFoundError, JobStoreError) as exc:
        _raise_job_error(exc)


@router.post("/jobs/{job_id}/retry")
def retry_job(
    job_id: str,
    max_attempts: int = 3,
    jobs_root: str | None = None,
    sqlite_db: str | None = None,
) -> dict[str, Any]:
    """Explicitly requeue one failed, timed out, or cancelled SQLite job."""

    if max_attempts < 1:
        raise HTTPException(status_code=400, detail="max_attempts must be >= 1")
    try:
        store = _sqlite_store_from_overrides(
            jobs_root=jobs_root,
            sqlite_db=sqlite_db,
            detail="job retry requires sqlite backend",
        )
        policy = RetryPolicy(max_attempts=max_attempts)
        return store.retry_job(job_id, policy=policy).to_dict()
    except JobStateConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (UnsafeJobIdError, JobNotFoundError, JobStoreError, ValueError) as exc:
        _raise_job_error(exc)


@router.get("/jobs/{job_id}")
def get_job(
    job_id: str,
    job_backend: JobBackend | None = None,
    jobs_root: str | None = None,
    sqlite_db: str | None = None,
) -> dict[str, Any]:
    """Read one persisted local job."""

    try:
        store = _job_store_from_overrides(
            job_backend=job_backend,
            jobs_root=jobs_root,
            sqlite_db=sqlite_db,
        )
        return store.get_job(job_id).to_dict()
    except (UnsafeJobIdError, JobNotFoundError, JobStoreError) as exc:
        _raise_job_error(exc)


@router.get("/jobs/{job_id}/result")
def get_job_result(
    job_id: str,
    job_backend: JobBackend | None = None,
    jobs_root: str | None = None,
    sqlite_db: str | None = None,
    runs_root: str | None = None,
) -> dict[str, Any]:
    """Read the result JSON for a succeeded job."""

    if _has_job_overrides(
        job_backend=job_backend,
        jobs_root=jobs_root,
        sqlite_db=sqlite_db,
        runs_root=runs_root,
    ):
        settings = _settings_with_overrides(
            job_backend=job_backend,
            jobs_root=jobs_root,
            sqlite_db=sqlite_db,
            runs_root=runs_root,
        )
        store = build_job_store(settings)
        resolved_runs_root = settings.runs_root
    else:
        runner = get_job_runner()
        store = runner.store
        resolved_runs_root = runner.runs_root
    try:
        job = store.get_job(job_id)
    except (UnsafeJobIdError, JobNotFoundError, JobStoreError) as exc:
        _raise_job_error(exc)
    if job.status != "succeeded":
        raise HTTPException(
            status_code=409,
            detail={"status": job.status, "error": job.error},
        )
    if job.result_path is None:
        raise HTTPException(status_code=404, detail="job result path is not recorded")
    return _read_json_file(Path(job.result_path), runs_root=resolved_runs_root)


@router.get("/jobs/{job_id}/logs")
def get_job_logs(
    job_id: str,
    job_backend: JobBackend | None = None,
    jobs_root: str | None = None,
    sqlite_db: str | None = None,
    runs_root: str | None = None,
) -> dict[str, Any]:
    """Return a JSON wrapper containing the job error and train log when available."""

    if _has_job_overrides(
        job_backend=job_backend,
        jobs_root=jobs_root,
        sqlite_db=sqlite_db,
        runs_root=runs_root,
    ):
        settings = _settings_with_overrides(
            job_backend=job_backend,
            jobs_root=jobs_root,
            sqlite_db=sqlite_db,
            runs_root=runs_root,
        )
        store = build_job_store(settings)
        resolved_runs_root = settings.runs_root
        max_bytes = settings.artifact_max_bytes
    else:
        runner = get_job_runner()
        store = runner.store
        resolved_runs_root = runner.runs_root
        max_bytes = _runner_settings().artifact_max_bytes
    try:
        job = store.get_job(job_id)
    except (UnsafeJobIdError, JobNotFoundError, JobStoreError) as exc:
        _raise_job_error(exc)

    log_text: str | None = None
    log_path: str | None = None
    if job.result_path is not None:
        candidate = Path(job.result_path).parent / "train.log"
        _assert_inside_root(candidate, resolved_runs_root)
        if candidate.exists():
            log_path = str(candidate)
            log_text = _read_log_file(candidate, max_bytes=max_bytes)
    return {
        "job_id": job.job_id,
        "status": job.status,
        "error": job.error,
        "log_path": log_path,
        "log": log_text,
    }


@router.get("/jobs/{job_id}/progress")
def get_job_progress(
    job_id: str,
    job_backend: JobBackend | None = None,
    jobs_root: str | None = None,
    sqlite_db: str | None = None,
    runs_root: str | None = None,
) -> dict[str, Any]:
    """Read live training progress for a local job when a run directory exists."""

    store, resolved_runs_root, max_bytes = _store_and_runs_root_from_overrides(
        job_backend=job_backend,
        jobs_root=jobs_root,
        sqlite_db=sqlite_db,
        runs_root=runs_root,
    )
    try:
        job = store.get_job(job_id)
    except (UnsafeJobIdError, JobNotFoundError, JobStoreError) as exc:
        _raise_job_error(exc)

    run_dir = _progress_run_dir(job, runs_root=resolved_runs_root)
    progress_payload: dict[str, Any] | None = None
    log_tail: str | None = None
    log_path: str | None = None
    if run_dir is not None:
        progress_path = run_dir / "progress.json"
        if progress_path.exists():
            progress_payload = _read_json_file(progress_path, runs_root=resolved_runs_root)
            child_progress, child_log_path, child_log_tail = _read_compare_child_progress(
                progress_payload,
                runs_root=resolved_runs_root,
                max_bytes=max_bytes,
            )
            if child_progress is not None:
                progress_payload["current_model_progress"] = child_progress
                _merge_compare_progress_percent(progress_payload, child_progress)
            if child_log_path is not None:
                log_path = child_log_path
                log_tail = child_log_tail
        candidate_log_path = run_dir / "train.log"
        if log_path is None and candidate_log_path.exists():
            log_path = str(candidate_log_path)
            log_tail = _read_log_tail(candidate_log_path, max_bytes=max_bytes)
    return {
        "job": job.to_dict(),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "progress": progress_payload,
        "log_path": log_path,
        "log_tail": log_tail,
    }


@router.post("/jobs/{job_id}/cancel")
def cancel_job(
    job_id: str,
    job_backend: JobBackend | None = None,
    jobs_root: str | None = None,
    sqlite_db: str | None = None,
) -> dict[str, Any]:
    """Request cancellation for a queued or running job."""

    try:
        store = _job_store_from_overrides(
            job_backend=job_backend,
            jobs_root=jobs_root,
            sqlite_db=sqlite_db,
        )
        return store.request_cancel(job_id).to_dict()
    except (UnsafeJobIdError, JobNotFoundError, JobStoreError) as exc:
        _raise_job_error(exc)


@router.post("/jobs/worker/once")
def run_worker_once(
    worker_id: str = "api_worker",
    jobs_root: str | None = None,
    sqlite_db: str | None = None,
    runs_root: str | None = None,
) -> dict[str, Any]:
    """Claim and run one queued SQLite job."""

    try:
        settings = (
            _settings_with_overrides(
                job_backend="sqlite",
                jobs_root=jobs_root,
                sqlite_db=sqlite_db,
                runs_root=runs_root,
            )
            if _has_job_overrides(jobs_root=jobs_root, sqlite_db=sqlite_db, runs_root=runs_root)
            else None
        )
        worker = _job_worker(
            worker_id=worker_id,
            settings=settings,
        )
        job = worker.run_once()
        return {"status": "idle"} if job is None else job.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except JobStoreError as exc:
        _raise_job_error(exc)


@router.post("/jobs/worker/loop")
def run_worker_loop(payload: WorkerLoopRequest) -> dict[str, Any]:
    """Claim and run queued SQLite jobs with finite local bounds."""

    try:
        settings = (
            _settings_with_overrides(
                job_backend="sqlite",
                jobs_root=payload.jobs_root,
                sqlite_db=payload.sqlite_db,
                runs_root=payload.runs_root,
            )
            if _has_job_overrides(
                jobs_root=payload.jobs_root,
                sqlite_db=payload.sqlite_db,
                runs_root=payload.runs_root,
            )
            else None
        )
        worker = _job_worker(
            worker_id=payload.worker_id,
            settings=settings,
        )
        jobs_payload: list[dict[str, Any]] = []
        processed = 0
        idle_cycles = 0
        while processed < payload.max_jobs and idle_cycles < payload.max_idle_cycles:
            job = worker.run_once()
            if job is None:
                idle_cycles += 1
                if idle_cycles < payload.max_idle_cycles:
                    time.sleep(payload.sleep_seconds)
                continue
            processed += 1
            jobs_payload.append(job.to_dict())
        return {
            "worker_id": worker.worker_id,
            "processed": processed,
            "idle_cycles": idle_cycles,
            "jobs": jobs_payload,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except JobStoreError as exc:
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


def _settings_with_overrides(
    *,
    job_backend: JobBackend | None = None,
    jobs_root: str | None = None,
    sqlite_db: str | None = None,
    runs_root: str | None = None,
) -> APISettings:
    settings = _runner_settings()
    return replace(
        settings,
        job_backend=job_backend or settings.job_backend,
        jobs_root=Path(jobs_root) if jobs_root else settings.jobs_root,
        sqlite_jobs_db_path=Path(sqlite_db) if sqlite_db else settings.sqlite_jobs_db_path,
        runs_root=Path(runs_root) if runs_root else settings.runs_root,
    )


def _store_and_runs_root_from_overrides(
    *,
    job_backend: JobBackend | None = None,
    jobs_root: str | None = None,
    sqlite_db: str | None = None,
    runs_root: str | None = None,
) -> tuple[JobStoreProtocol, Path, int]:
    if _has_job_overrides(
        job_backend=job_backend,
        jobs_root=jobs_root,
        sqlite_db=sqlite_db,
        runs_root=runs_root,
    ):
        settings = _settings_with_overrides(
            job_backend=job_backend,
            jobs_root=jobs_root,
            sqlite_db=sqlite_db,
            runs_root=runs_root,
        )
        return build_job_store(settings), settings.runs_root, settings.artifact_max_bytes
    runner = get_job_runner()
    return runner.store, runner.runs_root, _runner_settings().artifact_max_bytes


def _job_store_from_overrides(
    *,
    job_backend: JobBackend | None = None,
    jobs_root: str | None = None,
    sqlite_db: str | None = None,
) -> JobStoreProtocol:
    if not _has_job_overrides(
        job_backend=job_backend,
        jobs_root=jobs_root,
        sqlite_db=sqlite_db,
    ):
        return get_job_runner().store
    return build_job_store(
        _settings_with_overrides(
            job_backend=job_backend,
            jobs_root=jobs_root,
            sqlite_db=sqlite_db,
        )
    )


def _sqlite_store_from_overrides(
    *,
    jobs_root: str | None = None,
    sqlite_db: str | None = None,
    detail: str = "sqlite job store is required",
) -> SQLiteJobStore:
    if not _has_job_overrides(jobs_root=jobs_root, sqlite_db=sqlite_db):
        return _sqlite_store_or_raise(detail)
    settings = _settings_with_overrides(
        job_backend="sqlite",
        jobs_root=jobs_root,
        sqlite_db=sqlite_db,
    )
    store = build_job_store(settings)
    if not isinstance(store, SQLiteJobStore):
        raise HTTPException(status_code=400, detail="sqlite job store is required")
    return store


def _has_job_overrides(
    *,
    job_backend: JobBackend | None = None,
    jobs_root: str | None = None,
    sqlite_db: str | None = None,
    runs_root: str | None = None,
) -> bool:
    return any(value is not None for value in (job_backend, jobs_root, sqlite_db, runs_root))


def _read_log_file(path: Path, *, max_bytes: int) -> str:
    try:
        size_bytes = path.stat().st_size
        if size_bytes > max_bytes:
            detail = (
                f"job log exceeds maximum size: {path} ({size_bytes} bytes > {max_bytes} bytes)"
            )
            raise HTTPException(status_code=413, detail=detail)
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"job log is not valid UTF-8: {path}") from exc
    except HTTPException:
        raise
    except OSError as exc:
        raise HTTPException(status_code=404, detail=f"job log cannot be read: {path}") from exc


def _read_log_tail(path: Path, *, max_bytes: int) -> str:
    try:
        size_bytes = path.stat().st_size
        with path.open("rb") as handle:
            if size_bytes > max_bytes:
                handle.seek(-max_bytes, 2)
            content = handle.read().decode("utf-8", errors="replace")
        return content
    except OSError as exc:
        raise HTTPException(status_code=404, detail=f"job log cannot be read: {path}") from exc


def _read_compare_child_progress(
    progress_payload: dict[str, Any],
    *,
    runs_root: Path,
    max_bytes: int,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    if progress_payload.get("run_type") != "compare":
        return None, None, None
    run_dir_value = progress_payload.get("current_model_run_dir")
    if not isinstance(run_dir_value, str) or not run_dir_value:
        return None, None, None
    run_dir = Path(run_dir_value)
    _assert_inside_root(run_dir, runs_root)

    child_progress = None
    progress_path = run_dir / "progress.json"
    if progress_path.exists():
        child_progress = _read_json_file(progress_path, runs_root=runs_root)

    log_path = None
    log_tail = None
    candidate_log_path = run_dir / "train.log"
    if candidate_log_path.exists():
        log_path = str(candidate_log_path)
        log_tail = _read_log_tail(candidate_log_path, max_bytes=max_bytes)
    return child_progress, log_path, log_tail


def _merge_compare_progress_percent(
    progress_payload: dict[str, Any],
    child_progress: dict[str, Any],
) -> None:
    total_models = progress_payload.get("total_models")
    completed_models = progress_payload.get("completed_models")
    child_percent = child_progress.get("progress_percent")
    if not isinstance(total_models, int) or total_models <= 0:
        return
    if not isinstance(completed_models, int):
        return
    if not isinstance(child_percent, int | float):
        return
    combined = ((completed_models + (float(child_percent) / 100.0)) / total_models) * 100.0
    progress_payload["progress_percent"] = round(combined, 2)


def _progress_run_dir(job: JobRecord, *, runs_root: Path) -> Path | None:
    if job.result_path is not None:
        run_dir = Path(job.result_path).parent
        _assert_inside_root(run_dir, runs_root)
        return run_dir
    experiment_name = validate_safe_path_component(
        job.experiment_name,
        field_name="experiment_name",
    )
    experiment_root = Path(runs_root) / experiment_name
    _assert_inside_root(experiment_root, runs_root)
    latest_dir = experiment_root / "latest"
    if (latest_dir / "progress.json").exists() or (latest_dir / "train.log").exists():
        return latest_dir
    if not experiment_root.exists():
        return None
    candidates = [
        path
        for path in experiment_root.iterdir()
        if path.is_dir() and ((path / "progress.json").exists() or (path / "train.log").exists())
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _job_worker(*, worker_id: str, settings: APISettings | None = None) -> JobWorker:
    if settings is None:
        store = _sqlite_store_or_raise("worker operations require sqlite backend")
        runs_root = RUNS_ROOT
    else:
        built_store = build_job_store(settings)
        if not isinstance(built_store, SQLiteJobStore):
            raise HTTPException(status_code=400, detail="worker operations require sqlite backend")
        store = built_store
        runs_root = settings.runs_root
    return JobWorker(
        store=store,
        runs_root=runs_root,
        worker_id=worker_id,
    )


def _sqlite_store_or_raise(detail: str) -> SQLiteJobStore:
    store = get_job_runner().store
    if not isinstance(store, SQLiteJobStore):
        raise HTTPException(status_code=400, detail=detail)
    return store


def _raise_job_error(
    exc: UnsafeJobIdError | JobNotFoundError | JobStoreError | ValueError,
) -> NoReturn:
    if isinstance(exc, UnsafeJobIdError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, JobNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail=str(exc)) from exc
