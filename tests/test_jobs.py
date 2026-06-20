from __future__ import annotations

import json
import re
import time
from pathlib import Path
from threading import Event
from typing import Any

import pytest

from tests.helpers import tiny_config
from ts_platform.api.jobs.models import JOB_STATUS_VALUES, JobRecord, make_job_id, utc_now
from ts_platform.api.jobs.runner import JobRunner
from ts_platform.api.jobs.store import JobNotFoundError, JobStore, UnsafeJobIdError
from ts_platform.config.compare_schema import CompareConfig, CompareModelConfig
from ts_platform.config.schema import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    PlatformConfig,
    ScalerConfig,
    TrainingConfig,
    validate_safe_path_component,
)


def _compare_config(tmp_path: Path, *, name: str = "job_compare") -> CompareConfig:
    return CompareConfig(
        experiment=ExperimentConfig(name=name, output_dir=tmp_path, overwrite=True, seed=11),
        data=DataConfig(
            name="synthetic",
            input_len=4,
            output_len=2,
            batch_size=4,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            scaler=ScalerConfig(name="standard"),
            params={"length": 48, "num_features": 1, "noise_std": 0.0},
        ),
        models=[
            CompareModelConfig(name="naive"),
            CompareModelConfig(name="moving_average", params={"window_size": 2}),
        ],
        training=TrainingConfig(epochs=1, learning_rate=0.01, device="cpu"),
        evaluation=EvaluationConfig(metrics=["mae", "mse"], include_scaled_metrics=False),
        primary_metric="mae",
        continue_on_error=True,
    )


def test_job_record_serializes_roundtrip() -> None:
    now = utc_now()
    record = JobRecord(
        job_id=make_job_id(),
        job_type="train",
        status="queued",
        created_at=now,
        updated_at=now,
        started_at=None,
        finished_at=None,
        experiment_name="roundtrip",
        run_id=None,
        compare_run_id=None,
        result_path=None,
        leaderboard_json_path=None,
        artifacts_path=None,
        error=None,
        config_snapshot_path="runs/jobs/request_config.json",
    )

    assert JobRecord.from_dict(record.to_dict()) == record


def test_job_id_is_safe_path_component() -> None:
    job_id = make_job_id()

    assert re.fullmatch(r"\d{8}T\d{6}Z_[0-9a-f]{6}", job_id)
    assert validate_safe_path_component(job_id, field_name="job_id") == job_id


def test_job_status_values() -> None:
    assert set(JOB_STATUS_VALUES) == {
        "queued",
        "running",
        "succeeded",
        "failed",
        "cancel_requested",
        "cancelled",
    }


def test_job_store_creates_job_files(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")

    job = store.create_job("train", "store_create", {"config": True})

    job_dir = tmp_path / "jobs" / job.job_id
    assert (job_dir / "job.json").is_file()
    assert (job_dir / "request_config.json").is_file()
    assert json.loads((job_dir / "request_config.json").read_text(encoding="utf-8")) == {
        "config": True
    }


def test_job_store_reads_job(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    created = store.create_job("train", "store_read", {"config": True})

    read = store.get_job(created.job_id)

    assert read == created


def test_job_store_lists_jobs(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    first = store.create_job("train", "store_first", {"config": 1})
    second = store.create_job("compare", "store_second", {"config": 2})

    jobs = store.list_jobs()

    assert len(jobs) == 2
    assert {job.job_id for job in jobs} == {first.job_id, second.job_id}


def test_job_store_rejects_unsafe_job_id(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")

    with pytest.raises(UnsafeJobIdError, match="job_id"):
        store.get_job("../escape")


def test_job_store_missing_job_is_clear_error(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")

    with pytest.raises(JobNotFoundError, match="does not exist"):
        store.get_job("20260619T120000Z_a1b2c3")


def test_job_store_request_cancel_for_queued_job(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    job = store.create_job("train", "store_cancel", {"config": True})

    cancelled = store.request_cancel(job.job_id)

    assert cancelled.status == "cancelled"
    assert cancelled.finished_at is not None


def test_job_store_request_cancel_for_running_job(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    job = store.create_job("train", "store_cancel_running", {"config": True})
    store.mark_running(job.job_id)

    cancelled = store.request_cancel(job.job_id)

    assert cancelled.status == "cancel_requested"
    assert cancelled.finished_at is None


def test_job_runner_train_succeeds(tmp_path: Path) -> None:
    runner = JobRunner(jobs_root=tmp_path / "jobs", runs_root=tmp_path / "runs")
    try:
        job = runner.submit_train(tiny_config(tmp_path / "requested", name="runner_train"))

        finished = runner.wait(job.job_id, timeout=30)

        assert finished.status == "succeeded"
        assert finished.run_id
        assert finished.result_path is not None
        assert Path(finished.result_path).is_file()
        assert finished.artifacts_path is not None
        assert Path(finished.artifacts_path).is_file()
    finally:
        runner.shutdown()


def test_job_runner_compare_succeeds(tmp_path: Path) -> None:
    runner = JobRunner(jobs_root=tmp_path / "jobs", runs_root=tmp_path / "runs")
    try:
        job = runner.submit_compare(_compare_config(tmp_path / "requested"))

        finished = runner.wait(job.job_id, timeout=30)

        assert finished.status == "succeeded"
        assert finished.compare_run_id
        assert finished.result_path is not None
        assert Path(finished.result_path).is_file()
        assert finished.leaderboard_json_path is not None
        assert Path(finished.leaderboard_json_path).is_file()
    finally:
        runner.shutdown()


def test_job_runner_records_failure(tmp_path: Path) -> None:
    def fail_train(config: PlatformConfig, runs_root: Path) -> dict[str, Any]:
        raise RuntimeError("boom")

    runner = JobRunner(
        jobs_root=tmp_path / "jobs",
        runs_root=tmp_path / "runs",
        train_func=fail_train,
    )
    try:
        job = runner.submit_train(tiny_config(tmp_path / "requested", name="runner_failure"))

        finished = runner.wait(job.job_id, timeout=5)

        assert finished.status == "failed"
        assert finished.error is not None
        assert "boom" in finished.error
    finally:
        runner.shutdown()


def test_job_runner_uses_safe_runs_root(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe_runs"
    requested_root = tmp_path / "requested_runs"
    runner = JobRunner(jobs_root=tmp_path / "jobs", runs_root=safe_root)
    try:
        job = runner.submit_train(tiny_config(requested_root, name="runner_safe"))

        finished = runner.wait(job.job_id, timeout=30)

        assert finished.status == "succeeded"
        assert finished.result_path is not None
        assert Path(finished.result_path).is_relative_to(safe_root)
        assert not (requested_root / "runner_safe" / "latest" / "results.json").exists()
    finally:
        runner.shutdown()


def test_job_runner_does_not_block_submission(tmp_path: Path) -> None:
    started = Event()
    release = Event()

    def slow_train(config: PlatformConfig, runs_root: Path) -> dict[str, Any]:
        started.set()
        release.wait(timeout=5)
        run_dir = runs_root / config.experiment.name / "latest"
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_type": "train",
            "experiment_name": config.experiment.name,
            "run_id": "fake_run",
            "run_dir": str(run_dir),
        }
        (run_dir / "results.json").write_text(json.dumps(payload), encoding="utf-8")
        (run_dir / "artifacts.json").write_text("{}", encoding="utf-8")
        return payload

    runner = JobRunner(
        jobs_root=tmp_path / "jobs",
        runs_root=tmp_path / "runs",
        train_func=slow_train,
    )
    try:
        start = time.perf_counter()
        job = runner.submit_train(tiny_config(tmp_path / "requested", name="runner_fast_submit"))
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5
        assert job.status == "queued"
        assert started.wait(timeout=5)
        release.set()
        assert runner.wait(job.job_id, timeout=5).status == "succeeded"
    finally:
        release.set()
        runner.shutdown()
