"""Shared job store protocol."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ts_platform.api.jobs.models import JobRecord, JobType


@runtime_checkable
class JobStoreProtocol(Protocol):
    """Storage operations required by the local job runner and API routes."""

    def create_job(
        self,
        job_type: JobType,
        experiment_name: str,
        config_payload: dict[str, Any],
    ) -> JobRecord:
        """Create a queued job and persist its request payload."""

    def get_job(self, job_id: str) -> JobRecord:
        """Read one job by id."""

    def list_jobs(self, *, skip_corrupt: bool = True) -> list[JobRecord]:
        """List jobs newest first by creation time."""

    def update_job(self, job: JobRecord, *, touch: bool = True) -> JobRecord:
        """Persist an updated job record."""

    def mark_running(self, job_id: str) -> JobRecord:
        """Move a queued job to running unless it was cancelled first."""

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

    def mark_failed(self, job_id: str, error: str) -> JobRecord:
        """Mark a job failed and record a concise error."""

    def request_cancel(self, job_id: str) -> JobRecord:
        """Request cancellation for queued or running jobs."""
