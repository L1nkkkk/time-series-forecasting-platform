from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import Any

from fastapi.testclient import TestClient

from tests.helpers import tiny_config
from ts_platform.api.app import create_app
from ts_platform.api.jobs.runner import JobRunner
from ts_platform.api.jobs.sqlite_store import SQLiteJobStore
from ts_platform.api.routes import jobs
from ts_platform.api.settings import APISettings
from ts_platform.cli.main import main as cli_main
from ts_platform.config.compare_schema import CompareConfig, CompareModelConfig
from ts_platform.config.schema import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    PlatformConfig,
    ScalerConfig,
    TrainingConfig,
)


def _compare_config(tmp_path: Path, *, name: str = "api_job_compare") -> CompareConfig:
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


def _install_runner(
    monkeypatch: Any,
    tmp_path: Path,
    *,
    request: Any | None = None,
    train_func: Any = None,
    compare_func: Any = None,
    max_workers: int = 1,
) -> JobRunner:
    runner = JobRunner(
        jobs_root=tmp_path / "jobs",
        runs_root=tmp_path / "runs",
        max_workers=max_workers,
        train_func=train_func,
        compare_func=compare_func,
    )
    monkeypatch.setattr(jobs, "_JOB_RUNNER", runner)
    if request is not None:
        request.addfinalizer(lambda: runner.shutdown(wait=False))
    return runner


def _configure_external_worker_mode(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(
        jobs,
        "API_SETTINGS",
        APISettings(
            job_backend="sqlite",
            job_execution_mode="external_worker",
            runs_root=tmp_path / "runs",
            jobs_root=tmp_path / "jobs",
            sqlite_jobs_db_path=tmp_path / "jobs.sqlite3",
        ),
    )
    monkeypatch.setattr(jobs, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(jobs, "JOBS_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(jobs, "SQLITE_JOBS_DB_PATH", tmp_path / "jobs.sqlite3")
    monkeypatch.setattr(jobs, "_JOB_RUNNER", None)


def _write_train_result(config: PlatformConfig, runs_root: Path) -> dict[str, Any]:
    run_dir = runs_root / config.experiment.name / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_type": "train",
        "experiment_name": config.experiment.name,
        "run_id": "fake_train_run",
        "run_dir": str(run_dir),
        "test_metrics": {"original": {"mae": 1.0}},
    }
    (run_dir / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    (run_dir / "artifacts.json").write_text("{}", encoding="utf-8")
    (run_dir / "train.log").write_text("training complete", encoding="utf-8")
    return payload


def _write_compare_result(config: CompareConfig, runs_root: Path) -> dict[str, Any]:
    run_dir = runs_root / config.experiment.name / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = run_dir / "leaderboard.json"
    leaderboard_path.write_text("[]", encoding="utf-8")
    payload = {
        "run_type": "compare",
        "experiment_name": config.experiment.name,
        "compare_run_id": "fake_compare_run",
        "compare_run_dir": str(run_dir),
        "leaderboard_json_path": str(leaderboard_path),
        "success_count": 2,
        "failed_count": 0,
    }
    (run_dir / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    (run_dir / "artifacts.json").write_text("{}", encoding="utf-8")
    return payload


def test_shutdown_job_runner_resets_singleton(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(jobs, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(jobs, "JOBS_ROOT", tmp_path / "runs" / "jobs")
    monkeypatch.setattr(jobs, "_JOB_RUNNER", None)

    first = jobs.get_job_runner()
    jobs.shutdown_job_runner()

    assert jobs._JOB_RUNNER is None
    second = jobs.get_job_runner()
    assert second is not first
    jobs.shutdown_job_runner()


def test_app_shutdown_closes_job_runner(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(jobs, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(jobs, "JOBS_ROOT", tmp_path / "runs" / "jobs")
    monkeypatch.setattr(jobs, "_JOB_RUNNER", None)

    with TestClient(create_app()) as client:
        response = client.get("/jobs")
        runner = jobs._JOB_RUNNER

        assert response.status_code == 200
        assert runner is not None

    assert jobs._JOB_RUNNER is None
    assert jobs.get_job_runner() is not runner
    jobs.shutdown_job_runner()


def test_api_submit_train_job_returns_queued_or_running(
    tmp_path: Path,
    monkeypatch: Any,
    request: Any,
) -> None:
    runner = _install_runner(
        monkeypatch,
        tmp_path,
        request=request,
        train_func=_write_train_result,
    )
    client = TestClient(create_app())
    config = tiny_config(tmp_path / "requested", name="api_job_train").model_dump(mode="json")

    response = client.post("/jobs/train", json=config)
    payload = response.json()
    runner.wait(payload["job_id"], timeout=5)

    assert response.status_code == 200
    assert payload["job_type"] == "train"
    assert payload["status"] in {"queued", "running"}


def test_api_submit_compare_job_returns_queued_or_running(
    tmp_path: Path,
    monkeypatch: Any,
    request: Any,
) -> None:
    runner = _install_runner(
        monkeypatch,
        tmp_path,
        request=request,
        compare_func=_write_compare_result,
    )
    client = TestClient(create_app())
    config = _compare_config(tmp_path / "requested").model_dump(mode="json")

    response = client.post("/jobs/compare", json=config)
    payload = response.json()
    runner.wait(payload["job_id"], timeout=5)

    assert response.status_code == 200
    assert payload["job_type"] == "compare"
    assert payload["status"] in {"queued", "running"}


def test_api_get_job(tmp_path: Path, monkeypatch: Any, request: Any) -> None:
    runner = _install_runner(
        monkeypatch,
        tmp_path,
        request=request,
        train_func=_write_train_result,
    )
    client = TestClient(create_app())
    config = tiny_config(tmp_path / "requested", name="api_job_get").model_dump(mode="json")
    submitted = client.post("/jobs/train", json=config).json()
    runner.wait(submitted["job_id"], timeout=5)

    response = client.get(f"/jobs/{submitted['job_id']}")

    assert response.status_code == 200
    assert response.json()["job_id"] == submitted["job_id"]


def test_api_list_jobs(tmp_path: Path, monkeypatch: Any, request: Any) -> None:
    runner = _install_runner(
        monkeypatch,
        tmp_path,
        request=request,
        train_func=_write_train_result,
    )
    client = TestClient(create_app())
    config = tiny_config(tmp_path / "requested", name="api_job_list").model_dump(mode="json")
    submitted = client.post("/jobs/train", json=config).json()
    runner.wait(submitted["job_id"], timeout=5)

    response = client.get("/jobs")

    assert response.status_code == 200
    assert [job["job_id"] for job in response.json()["jobs"]] == [submitted["job_id"]]


def test_api_jobs_can_use_sqlite_backend(tmp_path: Path, monkeypatch: Any, request: Any) -> None:
    monkeypatch.setattr(
        jobs,
        "API_SETTINGS",
        APISettings(
            job_backend="sqlite",
            runs_root=tmp_path / "runs",
            jobs_root=tmp_path / "jobs",
            sqlite_jobs_db_path=tmp_path / "jobs.sqlite3",
        ),
    )
    monkeypatch.setattr(jobs, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(jobs, "JOBS_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(jobs, "SQLITE_JOBS_DB_PATH", tmp_path / "jobs.sqlite3")
    monkeypatch.setattr(jobs, "_JOB_RUNNER", None)
    request.addfinalizer(jobs.shutdown_job_runner)
    client = TestClient(create_app())
    config = tiny_config(tmp_path / "requested", name="api_sqlite_job").model_dump(mode="json")

    response = client.post("/jobs/train", json=config)
    payload = response.json()
    runner = jobs._JOB_RUNNER

    assert response.status_code == 200
    assert runner is not None
    assert isinstance(runner.store, SQLiteJobStore)
    assert runner.wait(payload["job_id"], timeout=30).status == "succeeded"
    assert (tmp_path / "jobs.sqlite3").is_file()


def test_api_submit_job_external_worker_mode_returns_queued(
    tmp_path: Path,
    monkeypatch: Any,
    request: Any,
) -> None:
    _configure_external_worker_mode(monkeypatch, tmp_path)
    request.addfinalizer(jobs.shutdown_job_runner)
    client = TestClient(create_app())
    config = tiny_config(tmp_path / "requested", name="api_external_queued").model_dump(mode="json")

    response = client.post("/jobs/train", json=config)

    assert response.status_code == 200
    assert response.json()["status"] == "queued"


def test_api_submit_job_external_worker_does_not_execute_immediately(
    tmp_path: Path,
    monkeypatch: Any,
    request: Any,
) -> None:
    _configure_external_worker_mode(monkeypatch, tmp_path)
    request.addfinalizer(jobs.shutdown_job_runner)
    client = TestClient(create_app())
    config = tiny_config(tmp_path / "requested", name="api_external_not_run").model_dump(
        mode="json"
    )

    submitted = client.post("/jobs/train", json=config).json()
    response = client.get(f"/jobs/{submitted['job_id']}")

    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    assert response.json()["result_path"] is None
    assert not (tmp_path / "runs" / "api_external_not_run" / "latest" / "results.json").exists()


def test_api_get_job_events_sqlite_backend(
    tmp_path: Path,
    monkeypatch: Any,
    request: Any,
) -> None:
    _configure_external_worker_mode(monkeypatch, tmp_path)
    request.addfinalizer(jobs.shutdown_job_runner)
    client = TestClient(create_app())
    config = tiny_config(tmp_path / "requested", name="api_events").model_dump(mode="json")
    submitted = client.post("/jobs/train", json=config).json()

    response = client.get(f"/jobs/{submitted['job_id']}/events")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert [event["event_type"] for event in payload] == ["job_created"]


def test_api_get_job_attempts_sqlite_backend(
    tmp_path: Path,
    monkeypatch: Any,
    request: Any,
) -> None:
    _configure_external_worker_mode(monkeypatch, tmp_path)
    request.addfinalizer(jobs.shutdown_job_runner)
    client = TestClient(create_app())
    config = tiny_config(tmp_path / "requested", name="api_attempts").model_dump(mode="json")
    submitted = client.post("/jobs/train", json=config).json()
    runner = jobs.get_job_runner()
    assert isinstance(runner.store, SQLiteJobStore)
    claimed = runner.store.claim_next_queued_job(worker_id="api_worker")
    assert claimed is not None

    response = client.get(f"/jobs/{submitted['job_id']}/attempts")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert payload[0]["attempt_id"] == claimed.attempt_id
    assert payload[0]["worker_id"] == "api_worker"


def test_api_get_job_events_requires_sqlite_backend(
    tmp_path: Path,
    monkeypatch: Any,
    request: Any,
) -> None:
    runner = _install_runner(monkeypatch, tmp_path, request=request)
    job = runner.store.create_job("train", "api_events_json", {"config": True})
    client = TestClient(create_app())

    response = client.get(f"/jobs/{job.job_id}/events")

    assert response.status_code == 400
    assert response.json()["detail"] == "job events require sqlite backend"


def test_api_get_job_attempts_requires_sqlite_backend(
    tmp_path: Path,
    monkeypatch: Any,
    request: Any,
) -> None:
    runner = _install_runner(monkeypatch, tmp_path, request=request)
    job = runner.store.create_job("train", "api_attempts_json", {"config": True})
    client = TestClient(create_app())

    response = client.get(f"/jobs/{job.job_id}/attempts")

    assert response.status_code == 400
    assert response.json()["detail"] == "job attempts require sqlite backend"


def test_api_get_job_events_rejects_unsafe_job_id(
    tmp_path: Path,
    monkeypatch: Any,
    request: Any,
) -> None:
    _configure_external_worker_mode(monkeypatch, tmp_path)
    request.addfinalizer(jobs.shutdown_job_runner)
    client = TestClient(create_app())

    response = client.get("/jobs/bad%20id/events")

    assert response.status_code == 400


def test_api_get_stale_jobs_sqlite_backend(
    tmp_path: Path,
    monkeypatch: Any,
    request: Any,
) -> None:
    _configure_external_worker_mode(monkeypatch, tmp_path)
    request.addfinalizer(jobs.shutdown_job_runner)
    client = TestClient(create_app())
    config = tiny_config(tmp_path / "requested", name="api_stale").model_dump(mode="json")
    submitted = client.post("/jobs/train", json=config).json()
    runner = jobs.get_job_runner()
    assert isinstance(runner.store, SQLiteJobStore)
    old = "2000-01-01T00:00:00+00:00"
    stale = replace(
        runner.store.get_job(submitted["job_id"]),
        status="running",
        started_at=old,
        updated_at=old,
    )
    runner.store.update_job(stale, touch=False)

    response = client.get("/jobs/stale?older_than_seconds=60")

    assert response.status_code == 200
    payload = response.json()
    assert [job["job_id"] for job in payload] == [submitted["job_id"]]
    assert runner.store.get_job(submitted["job_id"]).status == "running"


def test_api_get_stale_jobs_requires_sqlite_backend(
    tmp_path: Path,
    monkeypatch: Any,
    request: Any,
) -> None:
    _install_runner(monkeypatch, tmp_path, request=request)
    client = TestClient(create_app())

    response = client.get("/jobs/stale")

    assert response.status_code == 400
    assert response.json()["detail"] == "stale jobs require sqlite backend"


def test_api_get_stale_jobs_rejects_invalid_threshold(
    tmp_path: Path,
    monkeypatch: Any,
    request: Any,
) -> None:
    _configure_external_worker_mode(monkeypatch, tmp_path)
    request.addfinalizer(jobs.shutdown_job_runner)
    client = TestClient(create_app())

    response = client.get("/jobs/stale?older_than_seconds=0")

    assert response.status_code == 400
    assert response.json()["detail"] == "older_than_seconds must be > 0"


def test_api_timeout_job_sqlite_backend(
    tmp_path: Path,
    monkeypatch: Any,
    request: Any,
) -> None:
    _configure_external_worker_mode(monkeypatch, tmp_path)
    request.addfinalizer(jobs.shutdown_job_runner)
    client = TestClient(create_app())
    config = tiny_config(tmp_path / "requested", name="api_timeout").model_dump(mode="json")
    submitted = client.post("/jobs/train", json=config).json()
    runner = jobs.get_job_runner()
    assert isinstance(runner.store, SQLiteJobStore)
    claimed = runner.store.claim_next_queued_job(worker_id="api_worker")
    assert claimed is not None

    response = client.post(f"/jobs/{submitted['job_id']}/timeout?reason=api-timeout")

    assert response.status_code == 200
    assert response.json()["status"] == "timed_out"
    assert response.json()["error"] == "api-timeout"
    assert runner.store.list_attempts(submitted["job_id"])[0]["status"] == "failed"


def test_api_timeout_job_requires_sqlite_backend(
    tmp_path: Path,
    monkeypatch: Any,
    request: Any,
) -> None:
    runner = _install_runner(monkeypatch, tmp_path, request=request)
    job = runner.store.create_job("train", "api_timeout_json", {"config": True})
    client = TestClient(create_app())

    response = client.post(f"/jobs/{job.job_id}/timeout")

    assert response.status_code == 400
    assert response.json()["detail"] == "job timeout requires sqlite backend"


def test_api_retry_job_sqlite_backend(
    tmp_path: Path,
    monkeypatch: Any,
    request: Any,
) -> None:
    _configure_external_worker_mode(monkeypatch, tmp_path)
    request.addfinalizer(jobs.shutdown_job_runner)
    client = TestClient(create_app())
    config = tiny_config(tmp_path / "requested", name="api_retry").model_dump(mode="json")
    submitted = client.post("/jobs/train", json=config).json()
    runner = jobs.get_job_runner()
    assert isinstance(runner.store, SQLiteJobStore)
    runner.store.mark_failed(submitted["job_id"], "boom")

    response = client.post(f"/jobs/{submitted['job_id']}/retry")

    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    assert response.json()["error"] is None


def test_api_retry_job_respects_max_attempts(
    tmp_path: Path,
    monkeypatch: Any,
    request: Any,
) -> None:
    _configure_external_worker_mode(monkeypatch, tmp_path)
    request.addfinalizer(jobs.shutdown_job_runner)
    client = TestClient(create_app())
    config = tiny_config(tmp_path / "requested", name="api_retry_max").model_dump(mode="json")
    submitted = client.post("/jobs/train", json=config).json()
    runner = jobs.get_job_runner()
    assert isinstance(runner.store, SQLiteJobStore)
    claimed = runner.store.claim_next_queued_job(worker_id="api_worker")
    assert claimed is not None
    runner.store.mark_failed(submitted["job_id"], "boom")
    runner.store.mark_attempt_failed(claimed.attempt_id, "boom")

    response = client.post(f"/jobs/{submitted['job_id']}/retry?max_attempts=1")

    assert response.status_code == 409
    assert "max attempts" in response.json()["detail"]


def test_api_retry_job_rejects_running_job(
    tmp_path: Path,
    monkeypatch: Any,
    request: Any,
) -> None:
    _configure_external_worker_mode(monkeypatch, tmp_path)
    request.addfinalizer(jobs.shutdown_job_runner)
    client = TestClient(create_app())
    config = tiny_config(tmp_path / "requested", name="api_retry_running").model_dump(mode="json")
    submitted = client.post("/jobs/train", json=config).json()
    runner = jobs.get_job_runner()
    assert isinstance(runner.store, SQLiteJobStore)
    runner.store.mark_running(submitted["job_id"])

    response = client.post(f"/jobs/{submitted['job_id']}/retry")

    assert response.status_code == 409
    assert "cannot retry job with status" in response.json()["detail"]


def test_api_retry_job_requires_sqlite_backend(
    tmp_path: Path,
    monkeypatch: Any,
    request: Any,
) -> None:
    runner = _install_runner(monkeypatch, tmp_path, request=request)
    job = runner.store.create_job("train", "api_retry_json", {"config": True})
    client = TestClient(create_app())

    response = client.post(f"/jobs/{job.job_id}/retry")

    assert response.status_code == 400
    assert response.json()["detail"] == "job retry requires sqlite backend"


def test_api_external_worker_mode_requires_sqlite_backend(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        jobs,
        "API_SETTINGS",
        APISettings(
            job_backend="json",
            job_execution_mode="external_worker",
            runs_root=tmp_path / "runs",
            jobs_root=tmp_path / "jobs",
        ),
    )
    monkeypatch.setattr(jobs, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(jobs, "JOBS_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(jobs, "_JOB_RUNNER", None)
    client = TestClient(create_app(), raise_server_exceptions=False)
    config = tiny_config(tmp_path / "requested", name="api_external_bad_backend").model_dump(
        mode="json"
    )

    response = client.post("/jobs/train", json=config)

    assert response.status_code == 500
    assert "requires sqlite job backend" in response.json()["detail"]


def test_worker_once_processes_api_submitted_queued_job(
    tmp_path: Path,
    monkeypatch: Any,
    request: Any,
    capsys: Any,
) -> None:
    _configure_external_worker_mode(monkeypatch, tmp_path)
    request.addfinalizer(jobs.shutdown_job_runner)
    client = TestClient(create_app())
    config = tiny_config(tmp_path / "requested", name="api_external_worker_once").model_dump(
        mode="json"
    )
    submitted = client.post("/jobs/train", json=config).json()

    exit_code = cli_main(
        [
            "worker-once",
            "--jobs-root",
            str(tmp_path / "jobs"),
            "--sqlite-db",
            str(tmp_path / "jobs.sqlite3"),
            "--runs-root",
            str(tmp_path / "runs"),
            "--worker-id",
            "api_worker",
        ]
    )
    worker_payload = json.loads(capsys.readouterr().out)
    response = client.get(f"/jobs/{submitted['job_id']}")

    assert exit_code == 0
    assert worker_payload["job_id"] == submitted["job_id"]
    assert response.status_code == 200
    assert response.json()["status"] == "succeeded"
    assert Path(response.json()["result_path"]).is_file()


def test_api_list_jobs_skips_corrupt_metadata(
    tmp_path: Path,
    monkeypatch: Any,
    request: Any,
) -> None:
    runner = _install_runner(
        monkeypatch,
        tmp_path,
        request=request,
        train_func=_write_train_result,
    )
    good = runner.store.create_job("train", "api_good", {"config": 1})
    corrupt = runner.store.create_job("compare", "api_corrupt", {"config": 2})
    (tmp_path / "jobs" / corrupt.job_id / "job.json").write_text("{", encoding="utf-8")
    client = TestClient(create_app())

    response = client.get("/jobs")

    assert response.status_code == 200
    assert [job["job_id"] for job in response.json()["jobs"]] == [good.job_id]


def test_api_get_corrupt_job_returns_500(
    tmp_path: Path,
    monkeypatch: Any,
    request: Any,
) -> None:
    runner = _install_runner(
        monkeypatch,
        tmp_path,
        request=request,
        train_func=_write_train_result,
    )
    corrupt = runner.store.create_job("train", "api_corrupt_get", {"config": True})
    (tmp_path / "jobs" / corrupt.job_id / "job.json").write_text("{", encoding="utf-8")
    client = TestClient(create_app())

    response = client.get(f"/jobs/{corrupt.job_id}")

    assert response.status_code == 500


def test_api_job_result_after_success(tmp_path: Path, monkeypatch: Any, request: Any) -> None:
    runner = _install_runner(
        monkeypatch,
        tmp_path,
        request=request,
        train_func=_write_train_result,
    )
    client = TestClient(create_app())
    config = tiny_config(tmp_path / "requested", name="api_job_result").model_dump(mode="json")
    submitted = client.post("/jobs/train", json=config).json()
    runner.wait(submitted["job_id"], timeout=5)

    response = client.get(f"/jobs/{submitted['job_id']}/result")
    logs_response = client.get(f"/jobs/{submitted['job_id']}/logs")

    assert response.status_code == 200
    assert response.json()["experiment_name"] == "api_job_result"
    assert logs_response.status_code == 200
    assert logs_response.json()["log"] == "training complete"


def test_api_job_result_returns_conflict_when_not_ready(
    tmp_path: Path,
    monkeypatch: Any,
    request: Any,
) -> None:
    started = Event()
    release = Event()

    def slow_train(config: PlatformConfig, runs_root: Path) -> dict[str, Any]:
        started.set()
        release.wait(timeout=5)
        return _write_train_result(config, runs_root)

    runner = _install_runner(monkeypatch, tmp_path, request=request, train_func=slow_train)
    client = TestClient(create_app())
    config = tiny_config(tmp_path / "requested", name="api_job_not_ready").model_dump(mode="json")
    submitted = client.post("/jobs/train", json=config).json()
    assert started.wait(timeout=5)

    response = client.get(f"/jobs/{submitted['job_id']}/result")

    release.set()
    runner.wait(submitted["job_id"], timeout=5)
    assert response.status_code == 409
    assert response.json()["detail"]["status"] in {"queued", "running"}


def test_api_job_records_failed_status(tmp_path: Path, monkeypatch: Any, request: Any) -> None:
    def fail_train(config: PlatformConfig, runs_root: Path) -> dict[str, Any]:
        raise RuntimeError("api boom")

    runner = _install_runner(monkeypatch, tmp_path, request=request, train_func=fail_train)
    client = TestClient(create_app())
    config = tiny_config(tmp_path / "requested", name="api_job_failed").model_dump(mode="json")
    submitted = client.post("/jobs/train", json=config).json()
    runner.wait(submitted["job_id"], timeout=5)

    response = client.get(f"/jobs/{submitted['job_id']}")
    logs_response = client.get(f"/jobs/{submitted['job_id']}/logs")

    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert "api boom" in response.json()["error"]
    assert logs_response.status_code == 200
    assert "api boom" in logs_response.json()["error"]


def test_api_cancel_queued_job(tmp_path: Path, monkeypatch: Any, request: Any) -> None:
    started = Event()
    release = Event()

    def slow_train(config: PlatformConfig, runs_root: Path) -> dict[str, Any]:
        started.set()
        release.wait(timeout=5)
        return _write_train_result(config, runs_root)

    runner = _install_runner(monkeypatch, tmp_path, request=request, train_func=slow_train)
    client = TestClient(create_app())
    first_config = tiny_config(tmp_path / "requested", name="api_job_first").model_dump(mode="json")
    second_config = tiny_config(tmp_path / "requested", name="api_job_second").model_dump(
        mode="json"
    )
    first = client.post("/jobs/train", json=first_config).json()
    assert started.wait(timeout=5)
    second = client.post("/jobs/train", json=second_config).json()

    response = client.post(f"/jobs/{second['job_id']}/cancel")

    release.set()
    runner.wait(first["job_id"], timeout=5)
    runner.wait(second["job_id"], timeout=5)
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"


def test_api_cancel_running_job_marks_cancel_requested(
    tmp_path: Path,
    monkeypatch: Any,
    request: Any,
) -> None:
    started = Event()
    release = Event()

    def slow_train(config: PlatformConfig, runs_root: Path) -> dict[str, Any]:
        started.set()
        release.wait(timeout=5)
        return _write_train_result(config, runs_root)

    runner = _install_runner(monkeypatch, tmp_path, request=request, train_func=slow_train)
    client = TestClient(create_app())
    config = tiny_config(tmp_path / "requested", name="api_job_running_cancel").model_dump(
        mode="json"
    )
    submitted = client.post("/jobs/train", json=config).json()
    assert started.wait(timeout=5)

    response = client.post(f"/jobs/{submitted['job_id']}/cancel")

    release.set()
    runner.wait(submitted["job_id"], timeout=5)
    assert response.status_code == 200
    assert response.json()["status"] == "cancel_requested"


def test_api_job_rejects_unsafe_job_id(tmp_path: Path, monkeypatch: Any, request: Any) -> None:
    _install_runner(monkeypatch, tmp_path, request=request, train_func=_write_train_result)
    client = TestClient(create_app())

    response = client.get("/jobs/bad%20id")

    assert response.status_code == 400
