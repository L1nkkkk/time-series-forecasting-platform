"""Single-process SQLite job worker prototype."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ts_platform.api.jobs.models import JobRecord
from ts_platform.api.jobs.sqlite_store import SQLiteJobStore
from ts_platform.api.services.compare_service import compare_with_safe_output_dir
from ts_platform.api.services.training_service import train_with_safe_output_dir
from ts_platform.config.compare_schema import CompareConfig
from ts_platform.config.schema import PlatformConfig, validate_safe_path_component

TrainJobFunc = Callable[[PlatformConfig, Path], dict[str, Any]]
CompareJobFunc = Callable[[CompareConfig, Path], dict[str, Any]]


class JobWorker:
    """Claim and execute one SQLite-backed job at a time."""

    def __init__(
        self,
        *,
        store: SQLiteJobStore,
        runs_root: Path,
        worker_id: str,
        train_func: TrainJobFunc | None = None,
        compare_func: CompareJobFunc | None = None,
    ) -> None:
        self.store = store
        self.runs_root = Path(runs_root)
        self.worker_id = validate_safe_path_component(worker_id, field_name="worker_id")
        self._resolved_runs_root = self.runs_root.resolve()
        self._resolved_jobs_root = self.store.jobs_root.resolve()
        self._train_func = train_func or _run_train_job
        self._compare_func = compare_func or _run_compare_job

    def run_once(self) -> JobRecord | None:
        """Claim and run at most one queued job."""

        claimed = self.store.claim_next_queued_job(worker_id=self.worker_id)
        if claimed is None:
            return None

        job = claimed.job
        try:
            if job.job_type == "train":
                finished = self._run_train(job)
            elif job.job_type == "compare":
                finished = self._run_compare(job)
            else:  # pragma: no cover - JobRecord validation prevents this.
                msg = f"unsupported job_type: {job.job_type}"
                raise ValueError(msg)
            self.store.mark_attempt_succeeded(claimed.attempt_id)
            return finished
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            failed = self.store.mark_failed(job.job_id, error)
            self.store.mark_attempt_failed(claimed.attempt_id, error)
            return failed

    def _run_train(self, job: JobRecord) -> JobRecord:
        config = PlatformConfig.model_validate(self._read_config_snapshot(job))
        payload = self._train_func(config, self.runs_root)
        run_dir = self._payload_path(payload, "run_dir")
        return self.store.mark_succeeded(
            job.job_id,
            run_id=_optional_str(payload.get("run_id")),
            compare_run_id=None,
            result_path=str(run_dir / "results.json"),
            artifacts_path=str(run_dir / "artifacts.json"),
        )

    def _run_compare(self, job: JobRecord) -> JobRecord:
        config = CompareConfig.model_validate(self._read_config_snapshot(job))
        payload = self._compare_func(config, self.runs_root)
        compare_run_dir = self._payload_path(payload, "compare_run_dir")
        leaderboard_json_path = _optional_str(payload.get("leaderboard_json_path"))
        if leaderboard_json_path is not None:
            self._assert_inside_runs_root(Path(leaderboard_json_path))
        return self.store.mark_succeeded(
            job.job_id,
            run_id=None,
            compare_run_id=_optional_str(payload.get("compare_run_id")),
            result_path=str(compare_run_dir / "results.json"),
            artifacts_path=str(compare_run_dir / "artifacts.json"),
            leaderboard_json_path=leaderboard_json_path,
        )

    def _read_config_snapshot(self, job: JobRecord) -> dict[str, Any]:
        path = Path(job.config_snapshot_path)
        self._assert_inside_jobs_root(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            msg = f"job request config is not valid JSON: {path}"
            raise ValueError(msg) from exc
        except OSError as exc:
            msg = f"job request config cannot be read: {path}"
            raise ValueError(msg) from exc
        if not isinstance(payload, dict):
            msg = f"job request config is not a JSON object: {path}"
            raise ValueError(msg)
        return payload

    def _payload_path(self, payload: dict[str, Any], key: str) -> Path:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            msg = f"job result payload is missing {key}"
            raise ValueError(msg)
        path = Path(value)
        self._assert_inside_runs_root(path)
        return path

    def _assert_inside_runs_root(self, path: Path) -> None:
        resolved_path = path.resolve()
        if not resolved_path.is_relative_to(self._resolved_runs_root):
            msg = f"job result path escapes runs root: {path}"
            raise ValueError(msg)

    def _assert_inside_jobs_root(self, path: Path) -> None:
        resolved_path = path.resolve()
        if not resolved_path.is_relative_to(self._resolved_jobs_root):
            msg = f"job config snapshot escapes jobs root: {path}"
            raise ValueError(msg)


def _run_train_job(config: PlatformConfig, runs_root: Path) -> dict[str, Any]:
    return train_with_safe_output_dir(config, runs_root=runs_root)


def _run_compare_job(config: CompareConfig, runs_root: Path) -> dict[str, Any]:
    return compare_with_safe_output_dir(config, runs_root=runs_root)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
