"""SQLite-backed durable job store prototype."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import Any, cast

from ts_platform.api.jobs.models import JobRecord, JobType, make_job_id, utc_now, validate_job_id
from ts_platform.api.jobs.store import JobNotFoundError, JobStoreError, UnsafeJobIdError

_JOB_COLUMNS = (
    "job_id",
    "job_type",
    "status",
    "created_at",
    "updated_at",
    "started_at",
    "finished_at",
    "experiment_name",
    "run_id",
    "compare_run_id",
    "result_path",
    "leaderboard_json_path",
    "artifacts_path",
    "error",
    "config_snapshot_path",
)


class SQLiteJobStore:
    """Persist job metadata in SQLite while keeping config snapshots on disk."""

    def __init__(self, jobs_root: Path, db_path: Path) -> None:
        self.jobs_root = Path(jobs_root)
        self.db_path = Path(db_path)
        self._resolved_root = self.jobs_root.resolve()
        self._lock = RLock()
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

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
            job_dir.mkdir(parents=True, exist_ok=False)
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
            with self._connect() as conn:
                try:
                    self._insert_job(conn, job)
                    self._append_event(
                        conn,
                        job.job_id,
                        "job_created",
                        payload={"job_type": job.job_type, "experiment_name": job.experiment_name},
                    )
                except sqlite3.Error as exc:
                    msg = "job metadata cannot be written to SQLite"
                    raise JobStoreError(msg) from exc
            return job

    def get_job(self, job_id: str) -> JobRecord:
        """Read one job by id."""

        with self._lock:
            safe_job_id = self._safe_job_id(job_id)
            with self._connect() as conn:
                try:
                    row = conn.execute(
                        f"SELECT {', '.join(_JOB_COLUMNS)} FROM jobs WHERE job_id = ?",
                        (safe_job_id,),
                    ).fetchone()
                except sqlite3.Error as exc:
                    msg = "job metadata cannot be read from SQLite"
                    raise JobStoreError(msg) from exc
            if row is None:
                msg = f"job does not exist: {safe_job_id}"
                raise JobNotFoundError(msg)
            return self._row_to_job(row)

    def list_jobs(self, *, skip_corrupt: bool = True) -> list[JobRecord]:
        """List jobs newest first by ``created_at``."""

        with self._lock:
            with self._connect() as conn:
                try:
                    rows = conn.execute(
                        f"SELECT {', '.join(_JOB_COLUMNS)} FROM jobs ORDER BY created_at DESC"
                    ).fetchall()
                except sqlite3.Error as exc:
                    msg = "job metadata cannot be listed from SQLite"
                    raise JobStoreError(msg) from exc
            jobs: list[JobRecord] = []
            for row in rows:
                try:
                    jobs.append(self._row_to_job(row))
                except JobStoreError:
                    if not skip_corrupt:
                        raise
            return jobs

    def update_job(self, job: JobRecord, *, touch: bool = True) -> JobRecord:
        """Persist an updated job record."""

        with self._lock:
            stored_job = replace(job, updated_at=utc_now()) if touch else job
            self._safe_job_id(stored_job.job_id)
            with self._connect() as conn:
                try:
                    rowcount = self._update_job(conn, stored_job)
                except sqlite3.Error as exc:
                    msg = "job metadata cannot be updated in SQLite"
                    raise JobStoreError(msg) from exc
            if rowcount == 0:
                msg = f"job does not exist: {stored_job.job_id}"
                raise JobNotFoundError(msg)
            return stored_job

    def mark_running(self, job_id: str) -> JobRecord:
        """Move a queued job to running unless it was cancelled first."""

        with self._lock:
            job = self.get_job(job_id)
            if job.status == "cancelled":
                return job
            if job.status != "queued":
                return job
            now = utc_now()
            updated = replace(job, status="running", started_at=now, updated_at=now)
            return self._update_with_event(updated, "job_running")

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

        with self._lock:
            job = self.get_job(job_id)
            if job.status == "cancelled":
                return job
            now = utc_now()
            updated = replace(
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
            )
            return self._update_with_event(
                updated,
                "job_succeeded",
                payload={
                    "run_id": run_id,
                    "compare_run_id": compare_run_id,
                    "result_path": result_path,
                    "artifacts_path": artifacts_path,
                    "leaderboard_json_path": leaderboard_json_path,
                },
            )

    def mark_failed(self, job_id: str, error: str) -> JobRecord:
        """Mark a job failed and record a concise error."""

        with self._lock:
            job = self.get_job(job_id)
            if job.status == "cancelled":
                return job
            now = utc_now()
            updated = replace(job, status="failed", updated_at=now, finished_at=now, error=error)
            return self._update_with_event(updated, "job_failed", message=error)

    def request_cancel(self, job_id: str) -> JobRecord:
        """Request cancellation for queued or running jobs."""

        with self._lock:
            job = self.get_job(job_id)
            now = utc_now()
            if job.status == "queued":
                updated = replace(job, status="cancelled", updated_at=now, finished_at=now)
                return self._update_with_event(updated, "job_cancelled")
            if job.status == "running":
                updated = replace(job, status="cancel_requested", updated_at=now)
                return self._update_with_event(updated, "cancel_requested")
            return job

    def list_events(self, job_id: str) -> list[dict[str, Any]]:
        """List audit events for one job in insertion order."""

        with self._lock:
            safe_job_id = self._safe_job_id(job_id)
            with self._connect() as conn:
                try:
                    rows = conn.execute(
                        """
                        SELECT event_id, job_id, event_type, created_at, message, payload_json
                        FROM job_events
                        WHERE job_id = ?
                        ORDER BY event_id ASC
                        """,
                        (safe_job_id,),
                    ).fetchall()
                except sqlite3.Error as exc:
                    msg = "job events cannot be read from SQLite"
                    raise JobStoreError(msg) from exc
            return [self._row_to_event(row) for row in rows]

    def append_event(
        self,
        job_id: str,
        event_type: str,
        *,
        message: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Append one audit event for a job."""

        with self._lock:
            safe_job_id = self._safe_job_id(job_id)
            with self._connect() as conn:
                try:
                    self._append_event(
                        conn,
                        safe_job_id,
                        event_type,
                        message=message,
                        payload=payload,
                    )
                except sqlite3.Error as exc:
                    msg = "job event cannot be written to SQLite"
                    raise JobStoreError(msg) from exc

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS jobs (
                        job_id TEXT PRIMARY KEY,
                        job_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        started_at TEXT,
                        finished_at TEXT,
                        experiment_name TEXT NOT NULL,
                        run_id TEXT,
                        compare_run_id TEXT,
                        result_path TEXT,
                        leaderboard_json_path TEXT,
                        artifacts_path TEXT,
                        error TEXT,
                        config_snapshot_path TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS job_events (
                        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        job_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        message TEXT,
                        payload_json TEXT,
                        FOREIGN KEY(job_id) REFERENCES jobs(job_id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_job_events_job_id
                    ON job_events(job_id, event_id);
                    """
                )
            except sqlite3.Error as exc:
                msg = "SQLite job store cannot be initialized"
                raise JobStoreError(msg) from exc

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
        except sqlite3.Error as exc:
            msg = "SQLite job store cannot be opened"
            raise JobStoreError(msg) from exc
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            msg = "SQLite job store operation failed"
            raise JobStoreError(msg) from exc
        finally:
            conn.close()

    def _new_job_id(self) -> str:
        for _ in range(10):
            job_id = make_job_id()
            if not self._job_dir(job_id).exists() and not self._job_exists(job_id):
                return job_id
        msg = "could not create unique job id"
        raise JobStoreError(msg)

    def _job_exists(self, job_id: str) -> bool:
        with self._connect() as conn:
            try:
                row = conn.execute("SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            except sqlite3.Error as exc:
                msg = "job metadata cannot be checked in SQLite"
                raise JobStoreError(msg) from exc
        return row is not None

    def _job_dir(self, job_id: str) -> Path:
        safe_job_id = self._safe_job_id(job_id)
        path = self.jobs_root / safe_job_id
        self._assert_inside_root(path)
        return path

    def _safe_job_id(self, job_id: str) -> str:
        try:
            return validate_job_id(job_id)
        except ValueError as exc:
            raise UnsafeJobIdError(str(exc)) from exc

    def _insert_job(self, conn: sqlite3.Connection, job: JobRecord) -> None:
        conn.execute(
            """
            INSERT INTO jobs (
                job_id, job_type, status, created_at, updated_at, started_at, finished_at,
                experiment_name, run_id, compare_run_id, result_path, leaderboard_json_path,
                artifacts_path, error, config_snapshot_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            self._job_values(job),
        )

    def _update_job(self, conn: sqlite3.Connection, job: JobRecord) -> int:
        cursor = conn.execute(
            """
            UPDATE jobs
            SET job_type = ?,
                status = ?,
                created_at = ?,
                updated_at = ?,
                started_at = ?,
                finished_at = ?,
                experiment_name = ?,
                run_id = ?,
                compare_run_id = ?,
                result_path = ?,
                leaderboard_json_path = ?,
                artifacts_path = ?,
                error = ?,
                config_snapshot_path = ?
            WHERE job_id = ?
            """,
            (
                job.job_type,
                job.status,
                job.created_at,
                job.updated_at,
                job.started_at,
                job.finished_at,
                job.experiment_name,
                job.run_id,
                job.compare_run_id,
                job.result_path,
                job.leaderboard_json_path,
                job.artifacts_path,
                job.error,
                job.config_snapshot_path,
                job.job_id,
            ),
        )
        return cursor.rowcount

    def _update_with_event(
        self,
        job: JobRecord,
        event_type: str,
        *,
        message: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> JobRecord:
        with self._connect() as conn:
            try:
                rowcount = self._update_job(conn, job)
                if rowcount == 0:
                    msg = f"job does not exist: {job.job_id}"
                    raise JobNotFoundError(msg)
                self._append_event(conn, job.job_id, event_type, message=message, payload=payload)
            except sqlite3.Error as exc:
                msg = "job metadata cannot be updated in SQLite"
                raise JobStoreError(msg) from exc
        return job

    def _append_event(
        self,
        conn: sqlite3.Connection,
        job_id: str,
        event_type: str,
        *,
        message: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        payload_json = None if payload is None else json.dumps(payload, sort_keys=True)
        conn.execute(
            """
            INSERT INTO job_events (job_id, event_type, created_at, message, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (job_id, event_type, utc_now(), message, payload_json),
        )

    def _row_to_job(self, row: sqlite3.Row) -> JobRecord:
        payload = {column: row[column] for column in _JOB_COLUMNS}
        try:
            return JobRecord.from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            msg = f"job metadata is invalid in SQLite for job_id={payload.get('job_id')}"
            raise JobStoreError(msg) from exc

    def _row_to_event(self, row: sqlite3.Row) -> dict[str, Any]:
        payload_json = cast(str | None, row["payload_json"])
        payload: dict[str, Any] | None = None
        if payload_json is not None:
            try:
                decoded_payload = json.loads(payload_json)
            except json.JSONDecodeError as exc:
                msg = f"job event payload is not valid JSON: {row['event_id']}"
                raise JobStoreError(msg) from exc
            if not isinstance(decoded_payload, dict):
                msg = f"job event payload is not a JSON object: {row['event_id']}"
                raise JobStoreError(msg)
            payload = cast(dict[str, Any], decoded_payload)
        return {
            "event_id": row["event_id"],
            "job_id": row["job_id"],
            "event_type": row["event_type"],
            "created_at": row["created_at"],
            "message": row["message"],
            "payload_json": payload_json,
            "payload": payload,
        }

    def _job_values(self, job: JobRecord) -> tuple[str | None, ...]:
        return (
            job.job_id,
            job.job_type,
            job.status,
            job.created_at,
            job.updated_at,
            job.started_at,
            job.finished_at,
            job.experiment_name,
            job.run_id,
            job.compare_run_id,
            job.result_path,
            job.leaderboard_json_path,
            job.artifacts_path,
            job.error,
            job.config_snapshot_path,
        )

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        self._assert_inside_root(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp")
        try:
            tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            tmp_path.replace(path)
        except OSError as exc:
            msg = f"job request config cannot be written: {path}"
            raise JobStoreError(msg) from exc

    def _assert_inside_root(self, path: Path) -> None:
        resolved_path = path.resolve()
        if not resolved_path.is_relative_to(self._resolved_root):
            msg = f"job path escapes jobs root: {path}"
            raise UnsafeJobIdError(msg)
