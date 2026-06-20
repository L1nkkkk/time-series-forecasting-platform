"""Job models for the local API runner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, cast
from uuid import uuid4

from ts_platform.config.schema import validate_safe_path_component

JobStatus = Literal[
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancel_requested",
    "cancelled",
    "retrying",
    "timed_out",
]
JobType = Literal["train", "compare"]

JOB_STATUS_VALUES: tuple[JobStatus, ...] = (
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancel_requested",
    "cancelled",
    "retrying",
    "timed_out",
)
JOB_TYPE_VALUES: tuple[JobType, ...] = ("train", "compare")


@dataclass(frozen=True)
class JobRecord:
    """Serializable metadata for one submitted local job."""

    job_id: str
    job_type: JobType
    status: JobStatus
    created_at: str
    updated_at: str
    started_at: str | None
    finished_at: str | None
    experiment_name: str
    run_id: str | None
    compare_run_id: str | None
    result_path: str | None
    leaderboard_json_path: str | None
    artifacts_path: str | None
    error: str | None
    config_snapshot_path: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable job payload."""

        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "experiment_name": self.experiment_name,
            "run_id": self.run_id,
            "compare_run_id": self.compare_run_id,
            "result_path": self.result_path,
            "leaderboard_json_path": self.leaderboard_json_path,
            "artifacts_path": self.artifacts_path,
            "error": self.error,
            "config_snapshot_path": self.config_snapshot_path,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> JobRecord:
        """Build a job record from a decoded JSON object."""

        job_type = payload["job_type"]
        status = payload["status"]
        if job_type not in JOB_TYPE_VALUES:
            msg = f"unsupported job_type: {job_type}"
            raise ValueError(msg)
        if status not in JOB_STATUS_VALUES:
            msg = f"unsupported job status: {status}"
            raise ValueError(msg)
        return cls(
            job_id=str(payload["job_id"]),
            job_type=cast(JobType, job_type),
            status=cast(JobStatus, status),
            created_at=str(payload["created_at"]),
            updated_at=str(payload["updated_at"]),
            started_at=_optional_str(payload.get("started_at")),
            finished_at=_optional_str(payload.get("finished_at")),
            experiment_name=str(payload["experiment_name"]),
            run_id=_optional_str(payload.get("run_id")),
            compare_run_id=_optional_str(payload.get("compare_run_id")),
            result_path=_optional_str(payload.get("result_path")),
            leaderboard_json_path=_optional_str(payload.get("leaderboard_json_path")),
            artifacts_path=_optional_str(payload.get("artifacts_path")),
            error=_optional_str(payload.get("error")),
            config_snapshot_path=str(payload["config_snapshot_path"]),
        )


def make_job_id() -> str:
    """Create a safe job id with the same shape as run ids."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return validate_job_id(f"{timestamp}_{uuid4().hex[:6]}")


def utc_now() -> str:
    """Return an ISO timestamp suitable for job metadata."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def validate_job_id(value: str) -> str:
    """Validate a job id as one safe path component."""

    return validate_safe_path_component(value, field_name="job_id")


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
