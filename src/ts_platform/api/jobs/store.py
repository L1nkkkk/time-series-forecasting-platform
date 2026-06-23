"""Persistent local job store."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import Any, cast

from ts_platform.api.jobs.models import JobRecord, JobType, make_job_id, utc_now, validate_job_id


class UnsafeJobIdError(ValueError):
    """Raised when a job id is not safe for local path lookup."""


class JobNotFoundError(FileNotFoundError):
    """Raised when a requested job does not exist."""


class JobStoreError(ValueError):
    """Raised when job metadata cannot be read or written safely."""


class JobStateConflictError(JobStoreError):
    """Raised when a requested job state transition is not allowed."""


class JsonJobStore:
    """Persist job metadata as JSON under one fixed jobs root."""

    def __init__(self, jobs_root: Path) -> None:
        self.jobs_root = Path(jobs_root)
        self._resolved_root = self.jobs_root.resolve()
        self._lock = RLock()

    def create_job(
        self,
        job_type: JobType,
        experiment_name: str,
        config_payload: dict[str, Any],
    ) -> JobRecord:
        """Create a queued job and persist its request payload."""

        with self._lock:
            job_id = self._new_job_id()
            job_dir = self._job_dir(job_id)
            try:
                job_dir.mkdir(parents=True, exist_ok=False)
            except OSError as exc:
                msg = f"job directory cannot be created: {job_dir}"
                raise JobStoreError(msg) from exc
            config_snapshot_path = job_dir / "request_config.json"
            self._write_json(config_snapshot_path, config_payload)
            now = utc_now()
            job = JobRecord(
                job_id=job_id,
                job_type=job_type,
                status="queued",
                created_at=now,
                updated_at=now,
                started_at=None,
                finished_at=None,
                experiment_name=experiment_name,
                run_id=None,
                compare_run_id=None,
                result_path=None,
                leaderboard_json_path=None,
                artifacts_path=None,
                error=None,
                config_snapshot_path=str(config_snapshot_path),
            )
            self._write_job(job)
            return job

    def get_job(self, job_id: str) -> JobRecord:
        """Read one job by id."""

        with self._lock:
            path = self._job_dir(job_id) / "job.json"
            if not path.exists():
                msg = f"job does not exist: {job_id}"
                raise JobNotFoundError(msg)
            return self._read_job(path)

    def list_jobs(self, *, skip_corrupt: bool = True) -> list[JobRecord]:
        """List jobs newest first by ``created_at``.

        Corrupt job metadata is skipped by default so one damaged job does not
        prevent API callers from seeing other jobs. Set ``skip_corrupt=False``
        for strict debugging behavior.
        """

        with self._lock:
            if not self.jobs_root.exists():
                return []
            jobs: list[JobRecord] = []
            for path in self.jobs_root.iterdir():
                job_path = path / "job.json"
                if not path.is_dir() or not job_path.exists():
                    continue
                try:
                    jobs.append(self._read_job(job_path))
                except JobStoreError:
                    if not skip_corrupt:
                        raise
            return sorted(jobs, key=lambda job: job.created_at, reverse=True)

    def update_job(self, job: JobRecord, *, touch: bool = True) -> JobRecord:
        """Persist an updated job record."""

        with self._lock:
            stored_job = replace(job, updated_at=utc_now()) if touch else job
            self._write_job(stored_job)
            return stored_job

    def mark_running(self, job_id: str) -> JobRecord:
        """Move a queued job to running unless it was cancelled first."""

        job = self.get_job(job_id)
        if job.status == "cancelled":
            return job
        if job.status != "queued":
            return job
        now = utc_now()
        return self.update_job(
            replace(job, status="running", started_at=now, updated_at=now),
            touch=False,
        )

    def mark_succeeded(
        self,
        job_id: str,
        *,
        run_id: str | None,
        compare_run_id: str | None,
        result_path: str,
        artifacts_path: str | None,
        leaderboard_json_path: str | None = None,
    ) -> JobRecord:
        """Mark a job succeeded and record result artifact pointers."""

        job = self.get_job(job_id)
        if job.status == "cancelled":
            return job
        now = utc_now()
        return self.update_job(
            replace(
                job,
                status="succeeded",
                updated_at=now,
                finished_at=now,
                run_id=run_id,
                compare_run_id=compare_run_id,
                result_path=result_path,
                artifacts_path=artifacts_path,
                leaderboard_json_path=leaderboard_json_path,
                error=None,
            ),
            touch=False,
        )

    def mark_failed(self, job_id: str, error: str) -> JobRecord:
        """Mark a job failed and record a concise error."""

        job = self.get_job(job_id)
        if job.status == "cancelled":
            return job
        now = utc_now()
        return self.update_job(
            replace(
                job,
                status="failed",
                updated_at=now,
                finished_at=now,
                error=error,
            ),
            touch=False,
        )

    def request_cancel(self, job_id: str) -> JobRecord:
        """Request cancellation for queued or running jobs."""

        job = self.get_job(job_id)
        now = utc_now()
        if job.status == "queued":
            return self.update_job(
                replace(job, status="cancelled", updated_at=now, finished_at=now),
                touch=False,
            )
        if job.status == "running":
            return self.update_job(
                replace(job, status="cancel_requested", updated_at=now),
                touch=False,
            )
        return job

    def _new_job_id(self) -> str:
        for _ in range(10):
            job_id = make_job_id()
            if not self._job_dir(job_id).exists():
                return job_id
        msg = "could not create unique job id"
        raise JobStoreError(msg)

    def _job_dir(self, job_id: str) -> Path:
        try:
            safe_job_id = validate_job_id(job_id)
        except ValueError as exc:
            raise UnsafeJobIdError(str(exc)) from exc
        path = self.jobs_root / safe_job_id
        self._assert_inside_root(path)
        return path

    def _write_job(self, job: JobRecord) -> None:
        path = self._job_dir(job.job_id) / "job.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json(path, job.to_dict())

    def _read_job(self, path: Path) -> JobRecord:
        self._assert_inside_root(path)
        payload = self._read_json_object(path)
        try:
            return JobRecord.from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            msg = f"job metadata is invalid: {path}"
            raise JobStoreError(msg) from exc

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        self._assert_inside_root(path)
        tmp_path = path.with_name(f"{path.name}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            tmp_path.replace(path)
        except OSError as exc:
            msg = f"job metadata cannot be written: {path}"
            raise JobStoreError(msg) from exc

    def _read_json_object(self, path: Path) -> dict[str, Any]:
        payload = self._read_json(path)
        if not isinstance(payload, dict):
            msg = f"job metadata is not a JSON object: {path}"
            raise JobStoreError(msg)
        return cast(dict[str, Any], payload)

    def _read_json(self, path: Path) -> Any:
        self._assert_inside_root(path)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            msg = f"job metadata is not valid JSON: {path}"
            raise JobStoreError(msg) from exc
        except OSError as exc:
            msg = f"job metadata cannot be read: {path}"
            raise JobStoreError(msg) from exc

    def _assert_inside_root(self, path: Path) -> None:
        resolved_path = path.resolve()
        if not resolved_path.is_relative_to(self._resolved_root):
            msg = f"job path escapes jobs root: {path}"
            raise UnsafeJobIdError(msg)


JobStore = JsonJobStore
