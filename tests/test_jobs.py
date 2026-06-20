from __future__ import annotations

import json
import re
import time
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import Any

import pytest

from tests.helpers import tiny_config
from ts_platform.api.jobs.base import JobStoreProtocol
from ts_platform.api.jobs.factory import build_job_store
from ts_platform.api.jobs.models import JOB_STATUS_VALUES, JobRecord, make_job_id, utc_now
from ts_platform.api.jobs.runner import JobRunner
from ts_platform.api.jobs.sqlite_store import SQLiteJobStore
from ts_platform.api.jobs.store import (
    JobNotFoundError,
    JobStore,
    JobStoreError,
    JsonJobStore,
    UnsafeJobIdError,
)
from ts_platform.api.jobs.worker import JobWorker
from ts_platform.api.settings import APISettings
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


def _sqlite_store(tmp_path: Path) -> SQLiteJobStore:
    return SQLiteJobStore(tmp_path / "jobs", tmp_path / "jobs.sqlite3")


def _write_train_result(config: PlatformConfig, runs_root: Path) -> dict[str, Any]:
    run_dir = runs_root / config.experiment.name / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_type": "train",
        "experiment_name": config.experiment.name,
        "run_id": "worker_train_run",
        "run_dir": str(run_dir),
    }
    (run_dir / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    (run_dir / "artifacts.json").write_text("{}", encoding="utf-8")
    return payload


def _write_compare_result(config: CompareConfig, runs_root: Path) -> dict[str, Any]:
    run_dir = runs_root / config.experiment.name / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = run_dir / "leaderboard.json"
    leaderboard_path.write_text("[]", encoding="utf-8")
    payload = {
        "run_type": "compare",
        "experiment_name": config.experiment.name,
        "compare_run_id": "worker_compare_run",
        "compare_run_dir": str(run_dir),
        "leaderboard_json_path": str(leaderboard_path),
    }
    (run_dir / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    (run_dir / "artifacts.json").write_text("{}", encoding="utf-8")
    return payload


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


def test_json_job_store_implements_protocol(tmp_path: Path) -> None:
    assert isinstance(JsonJobStore(tmp_path / "jobs"), JobStoreProtocol)


def test_sqlite_job_store_implements_protocol(tmp_path: Path) -> None:
    assert isinstance(_sqlite_store(tmp_path), JobStoreProtocol)


def test_job_store_factory_builds_json_store(tmp_path: Path) -> None:
    store = build_job_store(APISettings(job_backend="json", jobs_root=tmp_path / "jobs"))

    assert isinstance(store, JsonJobStore)


def test_job_store_factory_builds_sqlite_store(tmp_path: Path) -> None:
    store = build_job_store(
        APISettings(
            job_backend="sqlite",
            jobs_root=tmp_path / "jobs",
            sqlite_jobs_db_path=tmp_path / "jobs.sqlite3",
        )
    )

    assert isinstance(store, SQLiteJobStore)


def test_job_store_factory_rejects_unknown_backend(tmp_path: Path) -> None:
    settings = APISettings(
        job_backend="memory",  # type: ignore[arg-type]
        jobs_root=tmp_path / "jobs",
    )

    with pytest.raises(ValueError, match="unsupported job backend"):
        build_job_store(settings)


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


def test_job_store_list_jobs_skips_corrupt_metadata(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    good = store.create_job("train", "store_good", {"config": 1})
    corrupt = store.create_job("compare", "store_corrupt", {"config": 2})
    (tmp_path / "jobs" / corrupt.job_id / "job.json").write_text("{", encoding="utf-8")

    jobs = store.list_jobs()

    assert [job.job_id for job in jobs] == [good.job_id]


def test_job_store_list_jobs_strict_corrupt_metadata_is_error(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    corrupt = store.create_job("train", "store_corrupt", {"config": True})
    (tmp_path / "jobs" / corrupt.job_id / "job.json").write_text("{", encoding="utf-8")

    with pytest.raises(JobStoreError, match="not valid JSON"):
        store.list_jobs(skip_corrupt=False)


def test_job_store_get_job_corrupt_metadata_is_error(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    corrupt = store.create_job("train", "store_corrupt_get", {"config": True})
    (tmp_path / "jobs" / corrupt.job_id / "job.json").write_text("{", encoding="utf-8")

    with pytest.raises(JobStoreError, match="not valid JSON"):
        store.get_job(corrupt.job_id)


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


def test_sqlite_job_store_creates_db(tmp_path: Path) -> None:
    db_path = tmp_path / "jobs.sqlite3"

    SQLiteJobStore(tmp_path / "jobs", db_path)

    assert db_path.is_file()


def test_sqlite_job_store_creates_job(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)

    job = store.create_job("train", "sqlite_create", {"config": True})

    assert job.status == "queued"
    assert (tmp_path / "jobs" / job.job_id / "request_config.json").is_file()
    assert not (tmp_path / "jobs" / job.job_id / "job.json").exists()


def test_sqlite_job_store_reads_job(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    created = store.create_job("train", "sqlite_read", {"config": True})

    read = store.get_job(created.job_id)

    assert read == created


def test_sqlite_job_store_lists_jobs(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    first = store.create_job("train", "sqlite_first", {"config": 1})
    second = store.create_job("compare", "sqlite_second", {"config": 2})

    jobs = store.list_jobs()

    assert len(jobs) == 2
    assert {job.job_id for job in jobs} == {first.job_id, second.job_id}


def test_sqlite_job_store_updates_status(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    job = store.create_job("train", "sqlite_update", {"config": True})

    updated = store.update_job(replace(job, status="running", started_at=utc_now()))

    assert updated.status == "running"
    assert store.get_job(job.job_id).status == "running"


def test_sqlite_job_store_mark_succeeded(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    job = store.create_job("train", "sqlite_success", {"config": True})

    succeeded = store.mark_succeeded(
        job.job_id,
        run_id="run_1",
        compare_run_id=None,
        result_path=str(tmp_path / "runs" / "results.json"),
        artifacts_path=str(tmp_path / "runs" / "artifacts.json"),
    )

    assert succeeded.status == "succeeded"
    assert succeeded.run_id == "run_1"
    assert succeeded.finished_at is not None


def test_sqlite_job_store_mark_failed(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    job = store.create_job("train", "sqlite_failed", {"config": True})

    failed = store.mark_failed(job.job_id, "boom")

    assert failed.status == "failed"
    assert failed.error == "boom"
    assert failed.finished_at is not None


def test_sqlite_job_store_request_cancel_queued(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    job = store.create_job("train", "sqlite_cancel", {"config": True})

    cancelled = store.request_cancel(job.job_id)

    assert cancelled.status == "cancelled"
    assert cancelled.finished_at is not None
    assert [event["event_type"] for event in store.list_events(job.job_id)] == [
        "job_created",
        "job_cancelled",
    ]


def test_sqlite_job_store_request_cancel_running(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    job = store.create_job("train", "sqlite_cancel_running", {"config": True})
    store.mark_running(job.job_id)

    cancelled = store.request_cancel(job.job_id)

    assert cancelled.status == "cancel_requested"
    assert cancelled.finished_at is None
    assert [event["event_type"] for event in store.list_events(job.job_id)] == [
        "job_created",
        "job_running",
        "cancel_requested",
    ]


def test_sqlite_job_store_rejects_unsafe_job_id(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)

    with pytest.raises(UnsafeJobIdError, match="job_id"):
        store.get_job("../escape")


def test_sqlite_job_store_missing_job_is_clear_error(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)

    with pytest.raises(JobNotFoundError, match="does not exist"):
        store.get_job("20260619T120000Z_a1b2c3")


def test_sqlite_job_store_persists_across_instances(tmp_path: Path) -> None:
    first_store = _sqlite_store(tmp_path)
    job = first_store.create_job("train", "sqlite_persist", {"config": True})
    second_store = _sqlite_store(tmp_path)

    assert second_store.get_job(job.job_id) == job


def test_sqlite_job_store_writes_request_config_snapshot(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    job = store.create_job("compare", "sqlite_snapshot", {"config": {"nested": True}})

    snapshot_path = Path(job.config_snapshot_path)

    assert snapshot_path.is_file()
    assert json.loads(snapshot_path.read_text(encoding="utf-8")) == {"config": {"nested": True}}


def test_sqlite_job_store_records_created_event(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    job = store.create_job("train", "sqlite_events_create", {"config": True})

    events = store.list_events(job.job_id)

    assert [event["event_type"] for event in events] == ["job_created"]
    assert events[0]["payload"]["job_type"] == "train"


def test_sqlite_job_store_records_status_transition_events(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    job = store.create_job("train", "sqlite_events", {"config": True})

    store.mark_running(job.job_id)
    store.mark_succeeded(
        job.job_id,
        run_id="run_1",
        compare_run_id=None,
        result_path=str(tmp_path / "runs" / "results.json"),
        artifacts_path=str(tmp_path / "runs" / "artifacts.json"),
    )

    assert [event["event_type"] for event in store.list_events(job.job_id)] == [
        "job_created",
        "job_running",
        "job_succeeded",
    ]


def test_sqlite_job_store_lists_events(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    job = store.create_job("train", "sqlite_events_list", {"config": True})
    store.append_event(job.job_id, "custom_event", message="hello", payload={"ok": True})

    events = store.list_events(job.job_id)

    assert [event["event_type"] for event in events] == ["job_created", "custom_event"]
    assert events[1]["message"] == "hello"
    assert events[1]["payload"] == {"ok": True}


def test_sqlite_job_store_creates_attempts_table(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)

    with store._connect() as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'job_attempts'"
        ).fetchone()

    assert row is not None


def test_sqlite_job_store_creates_attempt_on_claim(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    job = store.create_job("train", "sqlite_attempt", {"config": True})

    claimed = store.claim_next_queued_job(worker_id="worker_1")

    assert claimed is not None
    attempts = store.list_attempts(job.job_id)
    assert len(attempts) == 1
    assert attempts[0]["attempt_id"] == claimed.attempt_id
    assert attempts[0]["status"] == "running"
    assert attempts[0]["worker_id"] == "worker_1"


def test_sqlite_job_store_updates_attempt_on_success(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    job = store.create_job("train", "sqlite_attempt_success", {"config": True})
    claimed = store.claim_next_queued_job(worker_id="worker_1")
    assert claimed is not None

    attempt = store.mark_attempt_succeeded(claimed.attempt_id)

    assert attempt["status"] == "succeeded"
    assert attempt["finished_at"] is not None
    assert [event["event_type"] for event in store.list_events(job.job_id)] == [
        "job_created",
        "job_claimed",
        "attempt_succeeded",
    ]


def test_sqlite_job_store_updates_attempt_on_failure(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    job = store.create_job("train", "sqlite_attempt_failure", {"config": True})
    claimed = store.claim_next_queued_job(worker_id="worker_1")
    assert claimed is not None

    attempt = store.mark_attempt_failed(claimed.attempt_id, "boom")

    assert attempt["status"] == "failed"
    assert attempt["error"] == "boom"
    assert attempt["finished_at"] is not None
    assert [event["event_type"] for event in store.list_events(job.job_id)] == [
        "job_created",
        "job_claimed",
        "attempt_failed",
    ]


def test_sqlite_job_store_records_heartbeat_event(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    job = store.create_job("train", "sqlite_heartbeat", {"config": True})
    claimed = store.claim_next_queued_job(worker_id="worker_1")
    assert claimed is not None

    attempt = store.record_heartbeat(claimed.attempt_id)

    assert attempt["heartbeat_at"] is not None
    assert [event["event_type"] for event in store.list_events(job.job_id)] == [
        "job_created",
        "job_claimed",
        "heartbeat",
    ]


def test_sqlite_list_stale_running_jobs(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    job = store.create_job("train", "sqlite_stale", {"config": True})
    old = "2000-01-01T00:00:00+00:00"
    store.update_job(
        replace(job, status="running", started_at=old, updated_at=old),
        touch=False,
    )

    stale = store.list_stale_running_jobs(older_than_seconds=60)

    assert [item.job_id for item in stale] == [job.job_id]


def test_sqlite_list_stale_running_jobs_uses_heartbeat(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    job = store.create_job("train", "sqlite_stale_heartbeat", {"config": True})
    claimed = store.claim_next_queued_job(worker_id="worker_1")
    assert claimed is not None
    old = "2000-01-01T00:00:00+00:00"
    store.update_job(replace(claimed.job, updated_at=old), touch=False)

    assert store.list_stale_running_jobs(older_than_seconds=60) == []

    with store._connect() as conn:
        conn.execute(
            "UPDATE job_attempts SET heartbeat_at = ? WHERE attempt_id = ?",
            (old, claimed.attempt_id),
        )

    stale = store.list_stale_running_jobs(older_than_seconds=60)

    assert [item.job_id for item in stale] == [job.job_id]


def test_sqlite_list_stale_running_jobs_rejects_invalid_threshold(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)

    with pytest.raises(ValueError, match="older_than_seconds must be > 0"):
        store.list_stale_running_jobs(older_than_seconds=0)


def test_sqlite_list_stale_running_jobs_does_not_mutate_status(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    job = store.create_job("train", "sqlite_stale_no_mutate", {"config": True})
    old = "2000-01-01T00:00:00+00:00"
    store.update_job(
        replace(job, status="running", started_at=old, updated_at=old),
        touch=False,
    )

    stale = store.list_stale_running_jobs(older_than_seconds=60)

    assert [item.job_id for item in stale] == [job.job_id]
    assert store.get_job(job.job_id).status == "running"


def test_sqlite_claim_next_queued_job_returns_oldest(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    first = store.create_job("train", "sqlite_oldest_first", {"config": 1})
    second = store.create_job("train", "sqlite_oldest_second", {"config": 2})
    first = store.update_job(
        replace(first, created_at="2026-01-01T00:00:00+00:00"),
        touch=False,
    )
    store.update_job(
        replace(second, created_at="2026-01-02T00:00:00+00:00"),
        touch=False,
    )

    claimed = store.claim_next_queued_job(worker_id="worker_1")

    assert claimed is not None
    assert claimed.job.job_id == first.job_id
    assert store.get_job(second.job_id).status == "queued"


def test_sqlite_claim_next_queued_job_marks_running(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    job = store.create_job("train", "sqlite_claim_running", {"config": True})

    claimed = store.claim_next_queued_job(worker_id="worker_1")

    assert claimed is not None
    assert claimed.job.status == "running"
    assert claimed.job.started_at is not None
    assert store.get_job(job.job_id).status == "running"


def test_sqlite_claim_next_queued_job_ignores_cancelled(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    cancelled = store.create_job("train", "sqlite_cancelled_claim", {"config": 1})
    queued = store.create_job("train", "sqlite_queued_claim", {"config": 2})
    store.request_cancel(cancelled.job_id)

    claimed = store.claim_next_queued_job(worker_id="worker_1")

    assert claimed is not None
    assert claimed.job.job_id == queued.job_id


def test_sqlite_claim_next_queued_job_returns_none_when_empty(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)

    assert store.claim_next_queued_job(worker_id="worker_1") is None


def test_sqlite_claim_next_queued_job_records_event(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    job = store.create_job("train", "sqlite_claim_event", {"config": True})

    claimed = store.claim_next_queued_job(worker_id="worker_1")

    assert claimed is not None
    events = store.list_events(job.job_id)
    assert [event["event_type"] for event in events] == ["job_created", "job_claimed"]
    assert events[1]["payload"] == {"attempt_id": claimed.attempt_id, "worker_id": "worker_1"}


def test_sqlite_claim_next_queued_job_rejects_unsafe_worker_id(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    store.create_job("train", "sqlite_bad_worker", {"config": True})

    with pytest.raises(UnsafeJobIdError, match="worker_id"):
        store.claim_next_queued_job(worker_id="bad worker")


def test_job_worker_run_once_returns_none_when_empty(tmp_path: Path) -> None:
    worker = JobWorker(
        store=_sqlite_store(tmp_path),
        runs_root=tmp_path / "runs",
        worker_id="worker_1",
    )

    assert worker.run_once() is None


def test_job_worker_run_once_executes_train_job(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    config = tiny_config(tmp_path / "requested", name="worker_train")
    job = store.create_job("train", config.experiment.name, config.model_dump(mode="json"))
    worker = JobWorker(
        store=store,
        runs_root=tmp_path / "runs",
        worker_id="worker_1",
        train_func=_write_train_result,
    )

    finished = worker.run_once()

    assert finished is not None
    assert finished.job_id == job.job_id
    assert finished.status == "succeeded"
    assert finished.run_id == "worker_train_run"
    assert Path(finished.result_path or "").is_file()
    assert store.list_attempts(job.job_id)[0]["status"] == "succeeded"


def test_job_worker_run_once_executes_compare_job(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    config = _compare_config(tmp_path / "requested", name="worker_compare")
    job = store.create_job("compare", config.experiment.name, config.model_dump(mode="json"))
    worker = JobWorker(
        store=store,
        runs_root=tmp_path / "runs",
        worker_id="worker_1",
        compare_func=_write_compare_result,
    )

    finished = worker.run_once()

    assert finished is not None
    assert finished.job_id == job.job_id
    assert finished.status == "succeeded"
    assert finished.compare_run_id == "worker_compare_run"
    assert finished.leaderboard_json_path is not None
    assert Path(finished.leaderboard_json_path).is_file()


def test_job_worker_records_failure(tmp_path: Path) -> None:
    def fail_train(config: PlatformConfig, runs_root: Path) -> dict[str, Any]:
        raise RuntimeError("worker boom")

    store = _sqlite_store(tmp_path)
    config = tiny_config(tmp_path / "requested", name="worker_failure")
    job = store.create_job("train", config.experiment.name, config.model_dump(mode="json"))
    worker = JobWorker(
        store=store,
        runs_root=tmp_path / "runs",
        worker_id="worker_1",
        train_func=fail_train,
    )

    finished = worker.run_once()

    assert finished is not None
    assert finished.status == "failed"
    assert finished.error is not None
    assert "worker boom" in finished.error
    assert store.list_attempts(job.job_id)[0]["status"] == "failed"


def test_job_worker_records_heartbeat_event(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    config = tiny_config(tmp_path / "requested", name="worker_heartbeat")
    job = store.create_job("train", config.experiment.name, config.model_dump(mode="json"))
    worker = JobWorker(
        store=store,
        runs_root=tmp_path / "runs",
        worker_id="worker_1",
        train_func=_write_train_result,
    )

    finished = worker.run_once()

    assert finished is not None
    assert finished.status == "succeeded"
    event_types = [event["event_type"] for event in store.list_events(job.job_id)]
    assert event_types.count("heartbeat") == 2
    assert "attempt_succeeded" in event_types


def test_job_worker_records_heartbeat_before_success(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    config = tiny_config(tmp_path / "requested", name="worker_heartbeat_order")
    job = store.create_job("train", config.experiment.name, config.model_dump(mode="json"))
    worker = JobWorker(
        store=store,
        runs_root=tmp_path / "runs",
        worker_id="worker_1",
        train_func=_write_train_result,
    )

    worker.run_once()

    event_types = [event["event_type"] for event in store.list_events(job.job_id)]
    success_index = event_types.index("job_succeeded")
    assert event_types[success_index - 1] == "heartbeat"
    assert event_types.index("attempt_succeeded") > success_index


def test_job_worker_failure_preserves_original_error_when_heartbeat_fails(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    def fail_train(config: PlatformConfig, runs_root: Path) -> dict[str, Any]:
        raise RuntimeError("worker boom")

    store = _sqlite_store(tmp_path)
    config = tiny_config(tmp_path / "requested", name="worker_heartbeat_failure")
    job = store.create_job("train", config.experiment.name, config.model_dump(mode="json"))
    original_record_heartbeat = store.record_heartbeat
    heartbeat_calls = 0

    def flaky_record_heartbeat(attempt_id: int) -> dict[str, Any]:
        nonlocal heartbeat_calls
        heartbeat_calls += 1
        if heartbeat_calls > 1:
            raise RuntimeError("heartbeat down")
        return original_record_heartbeat(attempt_id)

    monkeypatch.setattr(store, "record_heartbeat", flaky_record_heartbeat)
    worker = JobWorker(
        store=store,
        runs_root=tmp_path / "runs",
        worker_id="worker_1",
        train_func=fail_train,
    )

    finished = worker.run_once()

    assert finished is not None
    assert finished.status == "failed"
    assert finished.error is not None
    assert "worker boom" in finished.error
    assert "heartbeat down" not in finished.error
    attempt = store.list_attempts(job.job_id)[0]
    assert attempt["status"] == "failed"
    assert "worker boom" in attempt["error"]


def test_job_worker_uses_safe_runs_root(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe_runs"
    requested_root = tmp_path / "requested_runs"
    store = _sqlite_store(tmp_path)
    config = tiny_config(requested_root, name="worker_safe")
    job = store.create_job("train", config.experiment.name, config.model_dump(mode="json"))
    worker = JobWorker(
        store=store,
        runs_root=safe_root,
        worker_id="worker_1",
        train_func=_write_train_result,
    )

    finished = worker.run_once()

    assert finished is not None
    assert finished.result_path is not None
    assert Path(finished.result_path).is_relative_to(safe_root)
    assert not (requested_root / "worker_safe" / "latest" / "results.json").exists()
    assert store.get_job(job.job_id).status == "succeeded"


def test_job_worker_reads_request_config_snapshot(tmp_path: Path) -> None:
    seen_name: str | None = None

    def capture_train(config: PlatformConfig, runs_root: Path) -> dict[str, Any]:
        nonlocal seen_name
        seen_name = config.experiment.name
        return _write_train_result(config, runs_root)

    store = _sqlite_store(tmp_path)
    config = tiny_config(tmp_path / "requested", name="worker_snapshot")
    store.create_job("train", config.experiment.name, config.model_dump(mode="json"))
    worker = JobWorker(
        store=store,
        runs_root=tmp_path / "runs",
        worker_id="worker_1",
        train_func=capture_train,
    )

    finished = worker.run_once()

    assert finished is not None
    assert seen_name == "worker_snapshot"


def test_job_runner_accepts_sqlite_store(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    runner = JobRunner(store=store, runs_root=tmp_path / "runs")
    try:
        assert runner.store is store
    finally:
        runner.shutdown()


def test_job_runner_train_succeeds_with_sqlite_store(tmp_path: Path) -> None:
    runner = JobRunner(store=_sqlite_store(tmp_path), runs_root=tmp_path / "runs")
    try:
        job = runner.submit_train(tiny_config(tmp_path / "requested", name="sqlite_runner_train"))

        finished = runner.wait(job.job_id, timeout=30)

        assert finished.status == "succeeded"
        assert finished.run_id
        assert finished.result_path is not None
        assert Path(finished.result_path).is_file()
        assert [event["event_type"] for event in runner.store.list_events(job.job_id)] == [
            "job_created",
            "job_running",
            "job_succeeded",
        ]
    finally:
        runner.shutdown()


def test_job_runner_compare_succeeds_with_sqlite_store(tmp_path: Path) -> None:
    runner = JobRunner(store=_sqlite_store(tmp_path), runs_root=tmp_path / "runs")
    try:
        job = runner.submit_compare(_compare_config(tmp_path / "requested", name="sqlite_compare"))

        finished = runner.wait(job.job_id, timeout=30)

        assert finished.status == "succeeded"
        assert finished.compare_run_id
        assert finished.leaderboard_json_path is not None
        assert Path(finished.leaderboard_json_path).is_file()
    finally:
        runner.shutdown()


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
