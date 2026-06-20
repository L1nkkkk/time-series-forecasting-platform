"""ThreadPoolExecutor-backed local job runner."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import RLock
from typing import Any

from ts_platform.api.jobs.base import JobStoreProtocol
from ts_platform.api.jobs.models import JobRecord
from ts_platform.api.jobs.store import JobStore
from ts_platform.api.services.compare_service import compare_with_safe_output_dir
from ts_platform.api.services.training_service import train_with_safe_output_dir
from ts_platform.config.compare_schema import CompareConfig
from ts_platform.config.schema import PlatformConfig

TrainJobFunc = Callable[[PlatformConfig, Path], dict[str, Any]]
CompareJobFunc = Callable[[CompareConfig, Path], dict[str, Any]]


class JobRunner:
    """Submit train and compare jobs to a local thread pool."""

    def __init__(
        self,
        *,
        runs_root: Path,
        jobs_root: Path | None = None,
        store: JobStoreProtocol | None = None,
        max_workers: int = 1,
        train_func: TrainJobFunc | None = None,
        compare_func: CompareJobFunc | None = None,
    ) -> None:
        if store is None:
            if jobs_root is None:
                msg = "jobs_root is required when store is not provided"
                raise ValueError(msg)
            store = JobStore(jobs_root)
        self.store = store
        self.runs_root = Path(runs_root)
        self._resolved_runs_root = self.runs_root.resolve()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="ts-platform-job",
        )
        self._train_func = train_func or _run_train_job
        self._compare_func = compare_func or _run_compare_job
        self._futures: dict[str, Future[None]] = {}
        self._lock = RLock()

    def submit_train(self, config: PlatformConfig) -> JobRecord:
        """Create and submit a training job without waiting for completion."""

        job = self.enqueue_train(config)
        future = self._executor.submit(self._run_train, job.job_id, config.model_copy(deep=True))
        self._remember_future(job.job_id, future)
        return job

    def submit_compare(self, config: CompareConfig) -> JobRecord:
        """Create and submit a compare job without waiting for completion."""

        job = self.enqueue_compare(config)
        future = self._executor.submit(self._run_compare, job.job_id, config.model_copy(deep=True))
        self._remember_future(job.job_id, future)
        return job

    def enqueue_train(self, config: PlatformConfig) -> JobRecord:
        """Create a queued training job without submitting in-process execution."""

        return self.store.create_job(
            "train",
            config.experiment.name,
            config.model_dump(mode="json"),
        )

    def enqueue_compare(self, config: CompareConfig) -> JobRecord:
        """Create a queued compare job without submitting in-process execution."""

        job = self.store.create_job(
            "compare",
            config.experiment.name,
            config.model_dump(mode="json"),
        )
        return job

    def wait(self, job_id: str, timeout: float | None = None) -> JobRecord:
        """Wait for an in-process job future, then return the stored job record."""

        future = self._future_for(job_id)
        if future is not None:
            future.result(timeout=timeout)
        return self.store.get_job(job_id)

    def shutdown(self, *, wait: bool = True) -> None:
        """Shut down the local executor."""

        self._executor.shutdown(wait=wait)

    def _run_train(self, job_id: str, config: PlatformConfig) -> None:
        try:
            if not self._mark_started(job_id):
                return
            payload = self._train_func(config, self.runs_root)
            run_dir = self._payload_path(payload, "run_dir")
            self.store.mark_succeeded(
                job_id,
                run_id=_optional_str(payload.get("run_id")),
                compare_run_id=None,
                result_path=str(run_dir / "results.json"),
                artifacts_path=str(run_dir / "artifacts.json"),
            )
        except Exception as exc:  # pragma: no cover - defensive guard around thread entrypoint
            self._mark_failed_safely(job_id, exc)

    def _run_compare(self, job_id: str, config: CompareConfig) -> None:
        try:
            if not self._mark_started(job_id):
                return
            payload = self._compare_func(config, self.runs_root)
            compare_run_dir = self._payload_path(payload, "compare_run_dir")
            leaderboard_json_path = _optional_str(payload.get("leaderboard_json_path"))
            if leaderboard_json_path is not None:
                self._assert_inside_runs_root(Path(leaderboard_json_path))
            self.store.mark_succeeded(
                job_id,
                run_id=None,
                compare_run_id=_optional_str(payload.get("compare_run_id")),
                result_path=str(compare_run_dir / "results.json"),
                artifacts_path=str(compare_run_dir / "artifacts.json"),
                leaderboard_json_path=leaderboard_json_path,
            )
        except Exception as exc:  # pragma: no cover - defensive guard around thread entrypoint
            self._mark_failed_safely(job_id, exc)

    def _mark_started(self, job_id: str) -> bool:
        job = self.store.get_job(job_id)
        if job.status == "cancelled":
            return False
        job = self.store.mark_running(job_id)
        return job.status != "cancelled"

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

    def _mark_failed_safely(self, job_id: str, exc: Exception) -> None:
        try:
            self.store.mark_failed(job_id, f"{type(exc).__name__}: {exc}")
        except Exception:
            return

    def _remember_future(self, job_id: str, future: Future[None]) -> None:
        with self._lock:
            self._futures[job_id] = future

    def _future_for(self, job_id: str) -> Future[None] | None:
        with self._lock:
            return self._futures.get(job_id)


def _run_train_job(config: PlatformConfig, runs_root: Path) -> dict[str, Any]:
    return train_with_safe_output_dir(config, runs_root=runs_root)


def _run_compare_job(config: CompareConfig, runs_root: Path) -> dict[str, Any]:
    return compare_with_safe_output_dir(config, runs_root=runs_root)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
