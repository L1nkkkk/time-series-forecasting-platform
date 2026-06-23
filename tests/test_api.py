from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from tests.helpers import tiny_config
from ts_platform.api.app import create_app
from ts_platform.api.jobs.runner import JobRunner
from ts_platform.api.jobs.store import JsonJobStore
from ts_platform.api.routes import datasets, demo, experiments, jobs, predict, tools
from ts_platform.api.settings import APISettings
from ts_platform.config.compare_schema import CompareConfig, CompareModelConfig
from ts_platform.config.schema import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ScalerConfig,
    TrainingConfig,
)
from ts_platform.data import DATASET_CATALOG, DatasetMetadata
from ts_platform.runner.comparer import CompareRunner
from ts_platform.runner.trainer import Trainer


def _compare_config(tmp_path, *, name: str = "api_compare") -> CompareConfig:
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


def _write_download_run(
    runs_root: Path,
    *,
    experiment_name: str,
    artifacts: list[dict[str, str]],
) -> Path:
    run_dir = runs_root / experiment_name / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts.json").write_text(
        json.dumps(
            {
                "run_type": "train",
                "experiment_name": experiment_name,
                "run_id": "latest",
                "run_dir": str(run_dir),
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_api_health_datasets_and_models() -> None:
    client = TestClient(create_app())

    health = client.get("/health")
    datasets = client.get("/datasets")
    models = client.get("/models")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert isinstance(health.json()["cuda_available"], bool)
    assert isinstance(health.json()["cuda_device_count"], int)
    assert datasets.status_code == 200
    assert "synthetic" in datasets.json()["names"]
    assert "csv" in datasets.json()["names"]
    assert any(item["name"] == "tiny_csv" for item in datasets.json()["datasets"])
    assert models.status_code == 200
    assert {"naive", "linear", "mlp", "transformer", "nbeats"}.issubset(
        set(models.json()["models"])
    )


def test_api_key_auth_blocks_non_exempt_routes() -> None:
    client = TestClient(create_app(APISettings(api_key="secret")))

    assert client.get("/health").status_code == 200
    assert client.get("/models").status_code == 401
    assert client.get("/models", headers={"x-api-key": "secret"}).status_code == 200
    assert client.get("/models", headers={"authorization": "Bearer secret"}).status_code == 200


def test_api_rejects_oversized_request_body() -> None:
    client = TestClient(create_app(APISettings(max_request_body_bytes=10)))

    response = client.post(
        "/datasets/profile-csv",
        content='{"too_large":"xxxxxxxxxxxxxxxxxxxxxxxx"}',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413


def test_api_rate_limit_returns_429() -> None:
    client = TestClient(create_app(APISettings(rate_limit_requests_per_minute=1)))

    assert client.get("/health").status_code == 200
    assert client.get("/health").status_code == 429


def test_api_audit_log_writes_jsonl(tmp_path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    client = TestClient(create_app(APISettings(audit_log_path=audit_path)))

    response = client.get("/health")

    assert response.status_code == 200
    event = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])
    assert event["method"] == "GET"
    assert event["path"] == "/health"
    assert event["status_code"] == 200


def test_ui_index_served() -> None:
    client = TestClient(create_app())

    response = client.get("/ui")

    assert response.status_code == 200
    assert "时间序列平台实验中心" in response.text
    assert "数据集库" in response.text
    assert "实验结果管理" in response.text
    assert "自定义实验" in response.text
    assert "English" in response.text
    assert "/ui/static/app.js" in response.text
    assert "/ui/static/styles.css" in response.text
    assert "dataset-search" in response.text
    assert "dataset-domain-filter" in response.text
    assert "job-demo-kind" in response.text
    assert "submit-demo-job" in response.text
    assert "20260624-cli-parity" in response.text
    assert "data-page-nav" in response.text
    assert "data-page-section" in response.text
    assert "export-report-run" in response.text
    assert "run-lookup-experiment" in response.text
    assert "run-lookup-runs-root" in response.text
    assert "lookup-results" in response.text
    assert "download-lookup-artifact" in response.text
    assert "run-config-file" in response.text
    assert "prediction-values-file" in response.text
    assert "predict-selected-run" in response.text
    assert "list-registered-datasets-tool" in response.text
    assert "list-models-tool" in response.text
    assert "list-catalog-tool" in response.text
    assert "profile-catalog-tool" in response.text
    assert "worker-once" in response.text
    assert "job-backend" in response.text
    assert "jobs-root" in response.text
    assert "jobs-sqlite-db" in response.text
    assert "jobs-runs-root" in response.text
    assert "dataset-tool-output-path" in response.text
    assert "dataset-tool-profile-name" in response.text
    assert "retry-max-attempts" in response.text
    assert "timeout-reason" in response.text
    assert "worker-id" in response.text
    assert "worker-max-jobs" in response.text
    assert "worker-idle-cycles" in response.text
    assert "worker-sleep-seconds" in response.text
    assert "load-job-progress" in response.text
    assert "auto-job-progress" in response.text
    assert "start-half-hour-demo" in response.text


def test_ui_static_assets_served() -> None:
    client = TestClient(create_app())

    script = client.get("/ui/static/app.js")
    styles = client.get("/ui/static/styles.css")

    assert script.status_code == 200
    assert "loadDashboard" in script.text
    assert "renderLeaderboardChart" in script.text
    assert "buildCustomConfig" in script.text
    assert "saveUserDataset" in script.text
    assert "deleteUserDataset" in script.text
    assert "filterDatasetRows" in script.text
    assert "countMergedDatasets" in script.text
    assert "profileDataset" in script.text
    assert "renderDatasetProfile" in script.text
    assert "datasetDomainFilter" in script.text
    assert "artifactDownload" in script.text
    assert "download>" in script.text
    assert "uploadCsvFile" in script.text
    assert "/datasets/user" in script.text
    assert "renderForecastChart" in script.text
    assert "renderTrainingMonitor" in script.text
    assert "monitor-smoothing" in script.text
    assert "collectTrainingSeries" in script.text
    assert "buildExperimentReport" in script.text
    assert "loadRunLookupPayload" in script.text
    assert "downloadRunLookupArtifact" in script.text
    assert "handlePredictionValuesFile" in script.text
    assert "listRegisteredDatasetsTool" in script.text
    assert "listModelsTool" in script.text
    assert "downloadTextFile" in script.text
    assert "downloadBlob" in script.text
    assert "readJobCliSettings" in script.text
    assert "jobQuery" in script.text
    assert "loadJobProgress" in script.text
    assert "startAutoJobProgress" in script.text
    assert "toggleAutoJobProgress" in script.text
    assert "renderJobProgress" in script.text
    assert "startHalfHourDemo" in script.text
    assert "appliances_energy_half_hour_demo" in script.text
    assert "setDashboardPage" in script.text
    assert "pageFromHash" in script.text
    assert "dataset-catalog-select" in script.text
    assert "submitSelectedDemoJob" in script.text
    assert "exportReport" in script.text
    assert "eyebrow" in script.text
    assert "Number.isInteger" in script.text
    assert "translations" in script.text
    assert "实验结果管理" in script.text
    assert styles.status_code == 200
    assert ".workbench-grid" in styles.text
    assert ".dataset-catalog-grid" in styles.text
    assert ".custom-grid" in styles.text
    assert ".danger-action" in styles.text
    assert ".dataset-filters" in styles.text
    assert ".dataset-detail-card" in styles.text
    assert ".job-launch-grid" in styles.text
    assert ".page-nav" in styles.text
    assert ".page-section" in styles.text
    assert ".monitor-panels" in styles.text
    assert ".monitor-chart" in styles.text
    assert ".profile-panel" in styles.text
    assert ".run-lookup-panel" in styles.text
    assert ".table-action-link" in styles.text


def test_demo_configs_lists_whitelist() -> None:
    client = TestClient(create_app())

    response = client.get("/demo/configs")
    payload = response.json()

    assert response.status_code == 200
    assert payload["configs"] == [
        "simple_forecast",
        "csv_forecast",
        "csv_feature_forecast",
        "appliances_energy_half_hour_demo",
        "compare_forecast",
        "compare_model_zoo",
        "compare_feature_forecast",
    ]
    assert payload["train"] == [
        "simple_forecast",
        "csv_forecast",
        "csv_feature_forecast",
        "appliances_energy_half_hour_demo",
    ]
    assert payload["compare"] == [
        "compare_forecast",
        "compare_model_zoo",
        "compare_feature_forecast",
    ]


def test_demo_train_rejects_unknown_demo_name() -> None:
    client = TestClient(create_app())

    response = client.post("/demo/train/not_allowed")

    assert response.status_code == 404


def test_demo_compare_rejects_unknown_demo_name() -> None:
    client = TestClient(create_app())

    response = client.post("/demo/compare/not_allowed")

    assert response.status_code == 404


def test_demo_train_job_submits_whitelisted_config(tmp_path, monkeypatch) -> None:
    safe_root = tmp_path / "safe_runs"
    runner = JobRunner(
        runs_root=safe_root,
        jobs_root=tmp_path / "jobs",
        train_func=lambda config, runs_root: {
            "experiment_name": config.experiment.name,
            "run_id": "fake_train_run",
            "run_dir": str(runs_root / config.experiment.name / "fake_train_run"),
        },
    )
    monkeypatch.setattr(jobs, "_JOB_RUNNER", runner)
    monkeypatch.setattr(jobs, "RUNS_ROOT", safe_root)
    monkeypatch.setattr(jobs, "JOBS_ROOT", tmp_path / "jobs")
    client = TestClient(create_app())

    try:
        response = client.post("/demo/jobs/train/simple_forecast")
        payload = response.json()

        assert response.status_code == 200
        assert payload["job_type"] == "train"
        assert payload["experiment_name"] == "simple_forecast"
        assert payload["job_id"]
        assert runner.wait(payload["job_id"], timeout=5).status == "succeeded"
    finally:
        jobs.shutdown_job_runner()


def test_demo_half_hour_job_submits_whitelisted_config(tmp_path, monkeypatch) -> None:
    safe_root = tmp_path / "safe_runs"
    runner = JobRunner(
        runs_root=safe_root,
        jobs_root=tmp_path / "jobs",
        train_func=lambda config, runs_root: {
            "experiment_name": config.experiment.name,
            "run_id": "fake_half_hour_run",
            "run_dir": str(runs_root / config.experiment.name / "fake_half_hour_run"),
        },
    )
    monkeypatch.setattr(jobs, "_JOB_RUNNER", runner)
    monkeypatch.setattr(jobs, "RUNS_ROOT", safe_root)
    monkeypatch.setattr(jobs, "JOBS_ROOT", tmp_path / "jobs")
    client = TestClient(create_app())

    try:
        response = client.post("/demo/jobs/train/appliances_energy_half_hour_demo")
        payload = response.json()

        assert response.status_code == 200
        assert payload["job_type"] == "train"
        assert payload["experiment_name"] == "appliances_energy_half_hour_demo"
        assert payload["job_id"]
        assert runner.wait(payload["job_id"], timeout=5).status == "succeeded"
    finally:
        jobs.shutdown_job_runner()


def test_demo_compare_job_submits_whitelisted_config(tmp_path, monkeypatch) -> None:
    safe_root = tmp_path / "safe_runs"
    runner = JobRunner(
        runs_root=safe_root,
        jobs_root=tmp_path / "jobs",
        compare_func=lambda config, runs_root: {
            "experiment_name": config.experiment.name,
            "compare_run_id": "fake_compare_run",
            "compare_run_dir": str(runs_root / config.experiment.name / "fake_compare_run"),
        },
    )
    monkeypatch.setattr(jobs, "_JOB_RUNNER", runner)
    monkeypatch.setattr(jobs, "RUNS_ROOT", safe_root)
    monkeypatch.setattr(jobs, "JOBS_ROOT", tmp_path / "jobs")
    client = TestClient(create_app())

    try:
        response = client.post("/demo/jobs/compare/compare_forecast")
        payload = response.json()

        assert response.status_code == 200
        assert payload["job_type"] == "compare"
        assert payload["experiment_name"] == "compare_forecast"
        assert payload["job_id"]
        assert runner.wait(payload["job_id"], timeout=5).status == "succeeded"
    finally:
        jobs.shutdown_job_runner()


def test_demo_train_simple_forecast_runs(tmp_path, monkeypatch) -> None:
    safe_root = tmp_path / "safe_runs"
    monkeypatch.setattr(demo, "RUNS_ROOT", safe_root)
    client = TestClient(create_app())

    response = client.post("/demo/train/simple_forecast")
    payload = response.json()

    assert response.status_code == 200
    assert payload["experiment_name"] == "simple_forecast"
    assert payload["run_id"]
    assert payload["test_metrics"]["original"]
    assert payload["data_metadata"]["feature_aware"] is False
    assert Path(payload["checkpoint_path"]).is_relative_to(safe_root)
    assert (safe_root / "simple_forecast" / "latest" / "results.json").exists()


def test_demo_compare_feature_forecast_runs(tmp_path, monkeypatch) -> None:
    safe_root = tmp_path / "safe_runs"
    monkeypatch.setattr(demo, "RUNS_ROOT", safe_root)
    client = TestClient(create_app())

    response = client.post("/demo/compare/compare_feature_forecast")
    payload = response.json()

    assert response.status_code == 200
    assert payload["experiment_name"] == "compare_feature_forecast"
    assert payload["run_type"] == "compare"
    assert payload["success_count"] >= 1
    assert payload["failed_count"] == 0
    assert payload["primary_metric"] == "mae"
    assert any(row["feature_aware"] is True for row in payload["rows"])
    assert any(row["input_dim"] == 3 for row in payload["rows"])
    assert (safe_root / "compare_feature_forecast" / "latest" / "leaderboard.json").exists()


def test_api_train_endpoint(tmp_path, monkeypatch) -> None:
    safe_root = tmp_path / "safe_runs"
    monkeypatch.setattr(experiments, "RUNS_ROOT", safe_root)
    client = TestClient(create_app())
    config = tiny_config(tmp_path / "requested", name="api").model_dump(mode="json")

    response = client.post("/experiments/train", json=config)
    payload = response.json()

    assert response.status_code == 200
    assert payload["run_dir"]
    assert payload["checkpoint_path"]
    assert payload["test_metrics"]["original"]
    assert Path(payload["checkpoint_path"]).exists()
    assert Path(payload["run_dir"]).is_relative_to(safe_root)
    assert (safe_root / "api" / "latest" / "results.json").exists()


def test_api_train_cuda_unavailable_returns_400(tmp_path, monkeypatch) -> None:
    safe_root = tmp_path / "safe_runs"
    monkeypatch.setattr(experiments, "RUNS_ROOT", safe_root)
    monkeypatch.setattr("ts_platform.runner.devices.torch.cuda.is_available", lambda: False)
    client = TestClient(create_app())
    config = tiny_config(tmp_path / "requested", name="api_cuda")
    config = config.model_copy(
        update={"training": TrainingConfig(epochs=1, learning_rate=0.01, device="cuda")}
    )

    response = client.post("/experiments/train", json=config.model_dump(mode="json"))
    payload = response.json()

    assert response.status_code == 400
    assert "CUDA was requested" in payload["detail"]
    assert not (safe_root / "api_cuda" / "latest").exists()


def test_api_train_runtime_failure_returns_clear_500(tmp_path, monkeypatch) -> None:
    def fail_train(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("training exploded")

    safe_root = tmp_path / "safe_runs"
    monkeypatch.setattr(experiments, "RUNS_ROOT", safe_root)
    monkeypatch.setattr(experiments, "train_with_safe_output_dir", fail_train)
    client = TestClient(create_app())
    config = tiny_config(tmp_path / "requested", name="api_failure").model_dump(mode="json")

    response = client.post("/experiments/train", json=config)
    payload = response.json()

    assert response.status_code == 500
    assert payload["detail"] == "training failed: training exploded"


def test_api_train_rejects_or_overrides_unsafe_output_dir(tmp_path, monkeypatch) -> None:
    safe_root = tmp_path / "safe_runs"
    unsafe_root = tmp_path / "unsafe"
    monkeypatch.setattr(experiments, "RUNS_ROOT", safe_root)
    client = TestClient(create_app())
    config = tiny_config(unsafe_root, name="api_unsafe").model_dump(mode="json")

    response = client.post("/experiments/train", json=config)
    payload = response.json()

    assert response.status_code == 200
    assert Path(payload["run_dir"]).is_relative_to(safe_root)
    assert (safe_root / "api_unsafe" / "latest" / "results.json").exists()
    assert not (unsafe_root / "api_unsafe" / "latest" / "results.json").exists()


def test_api_train_uses_safe_runs_root(tmp_path, monkeypatch) -> None:
    safe_root = tmp_path / "safe_runs"
    monkeypatch.setattr(experiments, "RUNS_ROOT", safe_root)
    client = TestClient(create_app())
    config = tiny_config(tmp_path / "requested", name="api_safe").model_dump(mode="json")

    response = client.post("/experiments/train", json=config)
    payload = response.json()

    assert response.status_code == 200
    assert Path(payload["checkpoint_path"]).is_relative_to(safe_root)
    assert (safe_root / "api_safe" / "latest" / "checkpoint.pt").exists()


def test_api_train_rejects_unsafe_experiment_name(tmp_path, monkeypatch) -> None:
    safe_root = tmp_path / "safe_runs"
    monkeypatch.setattr(experiments, "RUNS_ROOT", safe_root)
    client = TestClient(create_app())
    config = tiny_config(tmp_path / "requested", name="api_safe").model_dump(mode="json")
    config["experiment"]["name"] = "../escape"

    response = client.post("/experiments/train", json=config)

    assert response.status_code == 422
    assert "experiment.name must be a safe path component" in response.text


def test_api_experiments_still_ignores_root_query(tmp_path, monkeypatch) -> None:
    safe_root = tmp_path / "safe_runs"
    external_root = tmp_path / "external"
    external_run = external_root / "secret" / "latest"
    external_run.mkdir(parents=True)
    (external_run / "results.json").write_text(
        '{"experiment_name": "secret", "run_id": "external"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(experiments, "RUNS_ROOT", safe_root)
    client = TestClient(create_app())

    response = client.get(f"/experiments?root={external_root}")

    assert response.status_code == 200
    assert response.json()["experiments"] == []


def test_api_datasets_loads_local_catalog() -> None:
    client = TestClient(create_app())

    response = client.get("/datasets")
    datasets = response.json()["datasets"]

    assert response.status_code == 200
    assert any(item["name"] == "tiny_csv" for item in datasets)
    assert any(
        item["name"] == "etth1" and item["source"].startswith("https://") for item in datasets
    )


def test_api_upload_csv_dataset_saves_managed_copy(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(datasets, "UPLOADS_ROOT", tmp_path / "uploads")
    client = TestClient(create_app())

    response = client.post(
        "/datasets/upload-csv",
        json={"filename": "My Data.csv", "content": "timestamp,value,temp\n2024-01-01,1,20\n"},
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["filename"].startswith("My_Data_")
    assert payload["path"].startswith((tmp_path / "uploads").as_posix())
    assert payload["columns"] == ["timestamp", "value", "temp"]
    assert (
        Path(payload["path"]).read_text(encoding="utf-8")
        == "timestamp,value,temp\n2024-01-01,1,20\n"
    )


def test_api_upload_csv_dataset_rejects_non_csv(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(datasets, "UPLOADS_ROOT", tmp_path / "uploads")
    client = TestClient(create_app())

    response = client.post(
        "/datasets/upload-csv",
        json={"filename": "data.txt", "content": "timestamp,value\n2024-01-01,1\n"},
    )

    assert response.status_code == 422
    assert "must be a .csv" in response.text


def test_api_save_user_dataset_rejects_remote_csv_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(datasets, "USER_DATASETS_PATH", tmp_path / "user_datasets.json")
    client = TestClient(create_app())

    response = client.post(
        "/datasets/user",
        json={
            "name": "remote_csv",
            "dataset_type": "csv",
            "domain": "custom",
            "description": "Remote path should not be stored as a local CSV.",
            "path": "https://example.com/data.csv",
            "target_cols": ["value"],
        },
    )

    assert response.status_code == 422
    assert "must be a local path" in response.text


def test_api_save_user_dataset_persists_and_lists(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(datasets, "USER_DATASETS_PATH", tmp_path / "user_datasets.json")
    client = TestClient(create_app())

    response = client.post(
        "/datasets/user",
        json={
            "name": "my_csv",
            "dataset_type": "csv",
            "domain": "industrial",
            "description": "User motor sensor series.",
            "source": "internal lab",
            "path": "data/uploads/motor.csv",
            "timestamp_col": "timestamp",
            "target_cols": ["vibration"],
            "feature_cols": ["temperature"],
            "frequency": "1min",
        },
    )
    payload = response.json()
    listed = client.get("/datasets").json()["datasets"]
    detail = client.get("/datasets/my_csv").json()

    assert response.status_code == 200
    assert payload["name"] == "my_csv"
    assert payload["user_defined"] is True
    assert any(item["name"] == "my_csv" and item["user_defined"] is True for item in listed)
    assert detail["domain"] == "industrial"
    assert detail["target_cols"] == ["vibration"]
    assert (tmp_path / "user_datasets.json").exists()


def test_api_clear_user_datasets(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(datasets, "USER_DATASETS_PATH", tmp_path / "user_datasets.json")
    client = TestClient(create_app())
    client.post(
        "/datasets/user",
        json={
            "name": "clear_me",
            "dataset_type": "csv",
            "domain": "custom",
            "description": "Temporary user series.",
            "path": "data/uploads/clear.csv",
            "target_cols": ["value"],
        },
    )

    response = client.delete("/datasets/user")
    listed = client.get("/datasets").json()["datasets"]

    assert response.status_code == 200
    assert not any(item["name"] == "clear_me" for item in listed)


def test_api_delete_one_user_dataset(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(datasets, "USER_DATASETS_PATH", tmp_path / "user_datasets.json")
    client = TestClient(create_app())
    for name in ["keep_me", "delete_me"]:
        client.post(
            "/datasets/user",
            json={
                "name": name,
                "dataset_type": "csv",
                "domain": "custom",
                "description": f"{name} dataset.",
                "path": f"data/uploads/{name}.csv",
                "target_cols": ["value"],
            },
        )

    response = client.delete("/datasets/user/delete_me")
    listed = client.get("/datasets").json()["datasets"]

    assert response.status_code == 200
    assert any(item["name"] == "keep_me" for item in listed)
    assert not any(item["name"] == "delete_me" for item in listed)


def test_api_delete_user_dataset_rejects_unsafe_name(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(datasets, "USER_DATASETS_PATH", tmp_path / "user_datasets.json")
    client = TestClient(create_app())

    response = client.delete("/datasets/user/bad%20name")

    assert response.status_code == 400
    assert "dataset.name must be a safe path component" in response.text


def test_api_profile_user_dataset(tmp_path, monkeypatch) -> None:
    csv_path = tmp_path / "custom.csv"
    csv_path.write_text(
        "timestamp,value,temp\n"
        "2024-01-01,1,20\n"
        "2024-01-02,2,21\n"
        "2024-01-03,3,22\n"
        "2024-01-04,4,23\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(datasets, "USER_DATASETS_PATH", tmp_path / "user_datasets.json")
    client = TestClient(create_app())
    client.post(
        "/datasets/user",
        json={
            "name": "profile_me",
            "dataset_type": "csv",
            "domain": "custom",
            "description": "Profile user dataset.",
            "path": str(csv_path),
            "timestamp_col": "timestamp",
            "target_cols": ["value"],
            "feature_cols": ["temp"],
        },
    )

    response = client.get("/datasets/profile_me/profile", params={"input_len": 2, "output_len": 1})
    payload = response.json()

    assert response.status_code == 200
    assert payload["name"] == "profile_me"
    assert payload["exists"] is True
    assert payload["row_count"] == 4
    assert payload["columns"] == ["timestamp", "value", "temp"]
    assert payload["can_build_windows"] is True


def test_api_get_dataset_detail() -> None:
    client = TestClient(create_app())

    response = client.get("/datasets/tiny_csv")
    payload = response.json()

    assert response.status_code == 200
    assert payload["name"] == "tiny_csv"
    assert payload["dataset_type"] == "csv"
    assert payload["path"] == "tests/fixtures/tiny_series.csv"


def test_api_get_dataset_detail_missing_returns_404() -> None:
    client = TestClient(create_app())

    response = client.get("/datasets/missing_dataset")

    assert response.status_code == 404


def test_api_profile_dataset_from_catalog() -> None:
    client = TestClient(create_app())

    response = client.get("/datasets/tiny_csv/profile", params={"input_len": 8, "output_len": 2})
    payload = response.json()

    assert response.status_code == 200
    assert payload["name"] == "tiny_csv"
    assert payload["exists"] is True
    assert payload["row_count"] == 90
    assert payload["can_build_windows"] is True


def test_api_profile_dataset_does_not_accept_arbitrary_path(tmp_path) -> None:
    outside = tmp_path / "outside.csv"
    outside.write_text("timestamp,value\n2024-01-01,999\n", encoding="utf-8")
    client = TestClient(create_app())

    response = client.get("/datasets/tiny_csv/profile", params={"path": str(outside)})

    assert response.status_code == 422
    assert "path" in response.text


def test_api_profile_dataset_missing_file_returns_exists_false(tmp_path) -> None:
    DATASET_CATALOG.register(
        DatasetMetadata(
            name="missing_profile_csv",
            dataset_type="csv",
            domain="demo",
            description="Missing profile CSV",
            source=str(tmp_path / "missing.csv"),
            path=str(tmp_path / "missing.csv"),
            target_cols=["value"],
            timestamp_col="timestamp",
        )
    )
    client = TestClient(create_app())

    response = client.get("/datasets/missing_profile_csv/profile")
    payload = response.json()

    assert response.status_code == 200
    assert payload["exists"] is False
    assert payload["can_build_windows"] is False
    assert "missing file" in payload["warnings"]


def test_api_profile_dataset_rejects_non_csv_dataset() -> None:
    client = TestClient(create_app())

    response = client.get("/datasets/synthetic/profile")

    assert response.status_code == 400
    assert "only supports csv" in response.text


def test_api_experiments_lists_run_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    Trainer(tiny_config(tmp_path, name="api_list")).run()
    client = TestClient(create_app())

    response = client.get("/experiments")
    payload = response.json()

    assert response.status_code == 200
    assert payload["experiments"][0]["experiment_name"] == "api_list"
    assert payload["experiments"][0]["run_id"]
    assert payload["experiments"][0]["created_at"]
    assert payload["experiments"][0]["model_export_path"]


def test_api_experiments_lists_train_and_compare_runs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    Trainer(tiny_config(tmp_path, name="api_train_list")).run()
    CompareRunner(_compare_config(tmp_path, name="api_compare_list")).run()
    client = TestClient(create_app())

    response = client.get("/experiments")
    runs = response.json()["experiments"]

    assert response.status_code == 200
    assert {run["run_type"] for run in runs} == {"train", "compare"}
    assert any(run["experiment_name"] == "api_train_list" for run in runs)
    compare = next(run for run in runs if run["experiment_name"] == "api_compare_list")
    assert compare["status"] == "complete"
    assert compare["success_count"] == 2
    assert compare["failed_count"] == 0


def test_api_experiments_marks_incomplete_runs(tmp_path, monkeypatch) -> None:
    incomplete = tmp_path / "unfinished" / "latest"
    incomplete.mkdir(parents=True)
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    client = TestClient(create_app())

    response = client.get("/experiments")
    payload = response.json()["experiments"]

    assert response.status_code == 200
    assert payload == [
        {
            "status": "incomplete",
            "run_type": "unknown",
            "experiment_name": "unfinished",
            "run_dir": str(incomplete),
        }
    ]


def test_api_experiments_uses_experiment_store(tmp_path, monkeypatch) -> None:
    calls: list[Path] = []

    class FakeExperimentStore:
        def __init__(self, runs_root: Path) -> None:
            calls.append(runs_root)

        def list_experiments(self) -> list[dict[str, str]]:
            return [{"status": "complete", "run_type": "fake"}]

    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    monkeypatch.setattr(experiments, "ExperimentStore", FakeExperimentStore)
    client = TestClient(create_app())

    response = client.get("/experiments")

    assert response.status_code == 200
    assert calls == [tmp_path]
    assert response.json() == {"experiments": [{"status": "complete", "run_type": "fake"}]}


def test_api_get_train_results(tmp_path, monkeypatch) -> None:
    result = Trainer(tiny_config(tmp_path, name="api_train_results")).run()
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    client = TestClient(create_app())

    response = client.get("/experiments/api_train_results/latest/results")
    payload = response.json()

    assert response.status_code == 200
    assert payload["run_id"] == result.run_id
    assert payload["test_metrics"]["original"]


def test_api_get_compare_results(tmp_path, monkeypatch) -> None:
    result = CompareRunner(_compare_config(tmp_path, name="api_compare_results")).run()
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    client = TestClient(create_app())

    response = client.get("/experiments/api_compare_results/latest/results")
    payload = response.json()

    assert response.status_code == 200
    assert payload["run_type"] == "compare"
    assert payload["compare_run_id"] == result.compare_run_id


def test_api_get_train_artifacts(tmp_path, monkeypatch) -> None:
    result = Trainer(tiny_config(tmp_path, name="api_train_artifacts")).run()
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    client = TestClient(create_app())

    response = client.get("/experiments/api_train_artifacts/latest/artifacts")
    payload = response.json()

    assert response.status_code == 200
    assert payload["run_type"] == "train"
    assert payload["run_id"] == result.run_id
    assert any(artifact["name"] == "checkpoint" for artifact in payload["artifacts"])
    assert any(artifact["name"] == "model_export" for artifact in payload["artifacts"])
    assert any(artifact["name"] == "model_export_metadata" for artifact in payload["artifacts"])


def test_api_get_compare_artifacts(tmp_path, monkeypatch) -> None:
    result = CompareRunner(_compare_config(tmp_path, name="api_compare_artifacts")).run()
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    client = TestClient(create_app())

    response = client.get("/experiments/api_compare_artifacts/latest/artifacts")
    payload = response.json()

    assert response.status_code == 200
    assert payload["run_type"] == "compare"
    assert payload["compare_run_id"] == result.compare_run_id
    assert any(artifact["name"] == "leaderboard_json" for artifact in payload["artifacts"])


def test_api_get_artifacts_supports_latest(tmp_path, monkeypatch) -> None:
    Trainer(tiny_config(tmp_path, name="api_latest_artifacts")).run()
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    client = TestClient(create_app())

    response = client.get("/experiments/api_latest_artifacts/latest/artifacts")

    assert response.status_code == 200
    assert response.json()["experiment_name"] == "api_latest_artifacts"


def test_api_get_artifacts_returns_404_for_missing_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    client = TestClient(create_app())

    response = client.get("/experiments/missing/latest/artifacts")

    assert response.status_code == 404


def test_api_get_artifacts_rejects_unsafe_path_component(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    client = TestClient(create_app())

    response = client.get("/experiments/bad%20name/latest/artifacts")

    assert response.status_code == 400


def test_api_download_json_artifact(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "api_download_json" / "latest"
    run_dir.mkdir(parents=True)
    artifact_path = run_dir / "payload.json"
    artifact_path.write_text('{"ok": true}', encoding="utf-8")
    _write_download_run(
        tmp_path,
        experiment_name="api_download_json",
        artifacts=[
            {
                "name": "payload",
                "kind": "json",
                "path": str(artifact_path),
                "description": "JSON payload",
            }
        ],
    )
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    client = TestClient(create_app())

    response = client.get("/experiments/api_download_json/latest/artifacts/payload")

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert response.headers["content-type"].startswith("application/json")
    assert "payload.json" in response.headers["content-disposition"]


def test_api_download_csv_artifact(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "api_download_csv" / "latest"
    run_dir.mkdir(parents=True)
    artifact_path = run_dir / "leaderboard.csv"
    artifact_path.write_bytes(b"rank,model\n1,naive\n")
    _write_download_run(
        tmp_path,
        experiment_name="api_download_csv",
        artifacts=[
            {
                "name": "leaderboard_csv",
                "kind": "csv",
                "path": str(artifact_path),
                "description": "CSV payload",
            }
        ],
    )
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    client = TestClient(create_app())

    response = client.get("/experiments/api_download_csv/latest/artifacts/leaderboard_csv")

    assert response.status_code == 200
    assert response.text == "rank,model\n1,naive\n"
    assert response.headers["content-type"].startswith("text/csv")


def test_api_download_log_artifact(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "api_download_log" / "latest"
    run_dir.mkdir(parents=True)
    artifact_path = run_dir / "train.log"
    artifact_path.write_bytes(b"training complete\n")
    _write_download_run(
        tmp_path,
        experiment_name="api_download_log",
        artifacts=[
            {
                "name": "train_log",
                "kind": "log",
                "path": str(artifact_path),
                "description": "Log payload",
            }
        ],
    )
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    client = TestClient(create_app())

    response = client.get("/experiments/api_download_log/latest/artifacts/train_log")

    assert response.status_code == 200
    assert response.text == "training complete\n"
    assert response.headers["content-type"].startswith("text/plain")


def test_api_download_model_export_artifact(tmp_path, monkeypatch) -> None:
    result = Trainer(tiny_config(tmp_path, name="api_download_model_export")).run()
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    client = TestClient(create_app())

    response = client.get("/experiments/api_download_model_export/latest/artifacts/model_export")

    assert response.status_code == 200
    assert response.content == result.model_export_path.read_bytes()
    assert response.headers["content-type"].startswith("application/octet-stream")
    assert "model_export.pt" in response.headers["content-disposition"]


def test_api_predict_from_model_export_path(tmp_path) -> None:
    result = Trainer(tiny_config(tmp_path, name="api_predict_path")).run()
    client = TestClient(create_app())

    response = client.post(
        "/predict/model-export",
        json={
            "model_export_path": str(result.model_export_path),
            "values": [[[0.0] for _ in range(6)]],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["format"] == "ts_platform_prediction"
    assert len(payload["prediction"][0]) == 2


def test_api_predict_from_run_model_export(tmp_path, monkeypatch) -> None:
    Trainer(tiny_config(tmp_path, name="api_predict_run")).run()
    monkeypatch.setattr(predict, "RUNS_ROOT", tmp_path)
    client = TestClient(create_app())

    response = client.post(
        "/experiments/api_predict_run/latest/predict",
        json={"values": [[[0.0] for _ in range(6)]]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["format"] == "ts_platform_prediction"
    assert payload["model"]["output_len"] == 2


def test_api_tools_lookup_run_with_custom_runs_root(tmp_path) -> None:
    result = Trainer(tiny_config(tmp_path, name="api_tools_lookup")).run()
    client = TestClient(create_app())
    request = {
        "runs_root": str(tmp_path),
        "experiment": "api_tools_lookup",
        "run": "latest",
    }

    results_response = client.post("/tools/experiments/results", json=request)
    artifacts_response = client.post("/tools/experiments/artifacts", json=request)
    artifact_response = client.post(
        "/tools/experiments/artifact",
        json={**request, "artifact": "results"},
    )

    assert results_response.status_code == 200
    assert results_response.json()["run_id"] == result.run_id
    assert artifacts_response.status_code == 200
    assert any(item["name"] == "results" for item in artifacts_response.json()["artifacts"])
    assert artifact_response.status_code == 200
    assert artifact_response.json()["run_id"] == result.run_id


def test_api_jobs_list_with_custom_jobs_root(tmp_path) -> None:
    store = JsonJobStore(tmp_path / "custom_jobs")
    job = store.create_job("train", "custom_job", {"experiment": {"name": "custom_job"}})
    client = TestClient(create_app())

    list_response = client.get(
        "/jobs",
        params={
            "job_backend": "json",
            "jobs_root": str(store.jobs_root),
        },
    )
    get_response = client.get(
        f"/jobs/{job.job_id}",
        params={
            "job_backend": "json",
            "jobs_root": str(store.jobs_root),
        },
    )

    assert list_response.status_code == 200
    assert list_response.json()["jobs"][0]["job_id"] == job.job_id
    assert get_response.status_code == 200
    assert get_response.json()["experiment_name"] == "custom_job"


def test_api_profile_csv_path(tmp_path) -> None:
    csv_path = tmp_path / "series.csv"
    csv_path.write_text("timestamp,value\n2024-01-01,1\n2024-01-02,2\n", encoding="utf-8")
    client = TestClient(create_app())

    response = client.post(
        "/datasets/profile-csv",
        json={
            "path": str(csv_path),
            "target_cols": ["value"],
            "timestamp_col": "timestamp",
            "input_len": 1,
            "output_len": 1,
        },
    )

    assert response.status_code == 200
    assert response.json()["row_count"] == 2
    assert response.json()["can_build_windows"] is True


def test_api_profile_catalog_and_generate_config(tmp_path) -> None:
    csv_path = tmp_path / "series.csv"
    catalog_path = tmp_path / "catalog.yaml"
    csv_path.write_text("timestamp,value\n2024-01-01,1\n2024-01-02,2\n", encoding="utf-8")
    catalog_path.write_text(
        "\n".join(
            [
                "datasets:",
                "  - name: local_series",
                "    dataset_type: csv",
                "    domain: test",
                "    description: Local test data",
                "    source: local",
                f"    path: {csv_path.as_posix()}",
                "    timestamp_col: timestamp",
                "    target_cols: [value]",
            ]
        ),
        encoding="utf-8",
    )
    client = TestClient(create_app())

    profile_response = client.post(
        "/datasets/catalog/profile",
        json={"catalog_path": str(catalog_path), "input_len": 1, "output_len": 1},
    )
    list_response = client.post(
        "/datasets/catalog/list",
        json={"catalog_path": str(catalog_path)},
    )
    config_response = client.post(
        "/datasets/catalog/config",
        json={
            "catalog_path": str(catalog_path),
            "dataset": "local_series",
            "input_len": 1,
            "output_len": 1,
            "model": "linear",
            "epochs": 1,
            "output_path": str(tmp_path / "generated.yaml"),
        },
    )

    assert profile_response.status_code == 200
    assert profile_response.json()["profiles"][0]["name"] == "local_series"
    assert list_response.status_code == 200
    assert list_response.json()["datasets"][0]["name"] == "local_series"
    assert config_response.status_code == 200
    assert config_response.json()["config"]["experiment"]["name"] == "train_local_series_linear"
    assert Path(config_response.json()["output"]).exists()


def test_api_run_train_config_path(tmp_path, monkeypatch) -> None:
    safe_root = tmp_path / "safe_runs"
    config = tiny_config(tmp_path / "requested", name="api_train_config_path")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        json.dumps(config.model_dump(mode="json")),
        encoding="utf-8",
    )
    monkeypatch.setattr(tools, "API_SETTINGS", APISettings(runs_root=safe_root))
    client = TestClient(create_app())

    response = client.post("/configs/train/run", json={"config_path": str(config_path)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["experiment_name"] == "api_train_config_path"
    assert Path(payload["run_dir"]).is_relative_to(safe_root)


def test_api_download_unknown_artifact_returns_404(tmp_path, monkeypatch) -> None:
    _write_download_run(tmp_path, experiment_name="api_download_missing", artifacts=[])
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    client = TestClient(create_app())

    response = client.get("/experiments/api_download_missing/latest/artifacts/missing")

    assert response.status_code == 404


def test_api_download_rejects_unsafe_artifact_name(tmp_path, monkeypatch) -> None:
    _write_download_run(tmp_path, experiment_name="api_download_unsafe", artifacts=[])
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    client = TestClient(create_app())

    response = client.get("/experiments/api_download_unsafe/latest/artifacts/bad%20name")

    assert response.status_code == 400


def test_api_download_rejects_checkpoint_by_default(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "api_download_checkpoint" / "latest"
    run_dir.mkdir(parents=True)
    checkpoint_path = run_dir / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    _write_download_run(
        tmp_path,
        experiment_name="api_download_checkpoint",
        artifacts=[
            {
                "name": "checkpoint",
                "kind": "checkpoint",
                "path": str(checkpoint_path),
                "description": "Checkpoint",
            }
        ],
    )
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    client = TestClient(create_app())

    response = client.get("/experiments/api_download_checkpoint/latest/artifacts/checkpoint")

    assert response.status_code == 403


def test_api_download_rejects_large_artifact(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "api_download_large" / "latest"
    run_dir.mkdir(parents=True)
    artifact_path = run_dir / "large.csv"
    artifact_path.write_bytes(b"x" * (5 * 1024 * 1024 + 1))
    _write_download_run(
        tmp_path,
        experiment_name="api_download_large",
        artifacts=[
            {
                "name": "large_csv",
                "kind": "csv",
                "path": str(artifact_path),
                "description": "Large CSV",
            }
        ],
    )
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    client = TestClient(create_app())

    response = client.get("/experiments/api_download_large/latest/artifacts/large_csv")

    assert response.status_code == 413


def test_api_download_rejects_cross_run_artifact_path(tmp_path, monkeypatch) -> None:
    other_run_dir = tmp_path / "api_download_other" / "latest"
    other_run_dir.mkdir(parents=True)
    other_artifact_path = other_run_dir / "secret.json"
    other_artifact_path.write_text('{"secret": true}', encoding="utf-8")
    _write_download_run(
        tmp_path,
        experiment_name="api_download_cross_run",
        artifacts=[
            {
                "name": "secret",
                "kind": "json",
                "path": str(other_artifact_path),
                "description": "Cross-run artifact",
            }
        ],
    )
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    client = TestClient(create_app())

    response = client.get("/experiments/api_download_cross_run/latest/artifacts/secret")

    assert response.status_code == 400
    assert "escapes run directory" in response.text


def test_api_download_rejects_tampered_manifest_run_dir(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "api_download_tampered" / "latest"
    other_run_dir = tmp_path / "api_download_tampered_other" / "latest"
    run_dir.mkdir(parents=True)
    other_run_dir.mkdir(parents=True)
    other_artifact_path = other_run_dir / "secret.json"
    other_artifact_path.write_text('{"secret": true}', encoding="utf-8")
    (run_dir / "artifacts.json").write_text(
        json.dumps(
            {
                "run_type": "train",
                "experiment_name": "api_download_tampered",
                "run_id": "latest",
                "run_dir": str(other_run_dir),
                "artifacts": [
                    {
                        "name": "secret",
                        "kind": "json",
                        "path": str(other_artifact_path),
                        "description": "Tampered run_dir artifact",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    client = TestClient(create_app())

    response = client.get("/experiments/api_download_tampered/latest/artifacts/secret")

    assert response.status_code == 400
    assert "escapes run directory" in response.text


def test_api_download_uses_settings_max_bytes(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "api_download_settings_size" / "latest"
    run_dir.mkdir(parents=True)
    artifact_path = run_dir / "small.csv"
    artifact_path.write_text("abcd", encoding="utf-8")
    _write_download_run(
        tmp_path,
        experiment_name="api_download_settings_size",
        artifacts=[
            {
                "name": "small_csv",
                "kind": "csv",
                "path": str(artifact_path),
                "description": "Small CSV",
            }
        ],
    )
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    monkeypatch.setattr(experiments, "API_SETTINGS", APISettings(artifact_max_bytes=3))
    client = TestClient(create_app())

    response = client.get("/experiments/api_download_settings_size/latest/artifacts/small_csv")

    assert response.status_code == 413


def test_api_download_checkpoint_allowed_when_settings_enable_it(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "api_download_settings_checkpoint" / "latest"
    run_dir.mkdir(parents=True)
    checkpoint_path = run_dir / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    _write_download_run(
        tmp_path,
        experiment_name="api_download_settings_checkpoint",
        artifacts=[
            {
                "name": "checkpoint",
                "kind": "checkpoint",
                "path": str(checkpoint_path),
                "description": "Checkpoint",
            }
        ],
    )
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    monkeypatch.setattr(
        experiments,
        "API_SETTINGS",
        APISettings(allow_checkpoint_download=True),
    )
    client = TestClient(create_app())

    response = client.get(
        "/experiments/api_download_settings_checkpoint/latest/artifacts/checkpoint"
    )

    assert response.status_code == 200
    assert response.content == b"checkpoint"
    assert response.headers["content-type"].startswith("application/octet-stream")


def test_api_download_rejects_kind_not_in_settings_allowed_kinds(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "api_download_settings_kind" / "latest"
    run_dir.mkdir(parents=True)
    artifact_path = run_dir / "leaderboard.csv"
    artifact_path.write_text("rank,model\n1,naive\n", encoding="utf-8")
    _write_download_run(
        tmp_path,
        experiment_name="api_download_settings_kind",
        artifacts=[
            {
                "name": "leaderboard_csv",
                "kind": "csv",
                "path": str(artifact_path),
                "description": "CSV payload",
            }
        ],
    )
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    monkeypatch.setattr(
        experiments,
        "API_SETTINGS",
        APISettings(artifact_allowed_kinds=("json",)),
    )
    client = TestClient(create_app())

    response = client.get(
        "/experiments/api_download_settings_kind/latest/artifacts/leaderboard_csv"
    )

    assert response.status_code == 403


def test_api_download_corrupt_manifest_returns_500(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "api_download_corrupt" / "latest"
    run_dir.mkdir(parents=True)
    (run_dir / "artifacts.json").write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    client = TestClient(create_app())

    response = client.get("/experiments/api_download_corrupt/latest/artifacts/results")

    assert response.status_code == 500


def test_api_download_does_not_allow_path_query(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "api_download_query" / "latest"
    run_dir.mkdir(parents=True)
    artifact_path = run_dir / "allowed.json"
    artifact_path.write_text('{"allowed": true}', encoding="utf-8")
    outside_path = tmp_path.parent / "secret.json"
    outside_path.write_text('{"secret": true}', encoding="utf-8")
    _write_download_run(
        tmp_path,
        experiment_name="api_download_query",
        artifacts=[
            {
                "name": "allowed",
                "kind": "json",
                "path": str(artifact_path),
                "description": "Allowed JSON",
            }
        ],
    )
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    client = TestClient(create_app())

    response = client.get(
        "/experiments/api_download_query/latest/artifacts/allowed",
        params={"path": str(outside_path)},
    )

    assert response.status_code == 200
    assert response.json() == {"allowed": True}
    assert "secret" not in response.text


def test_api_get_results_supports_latest(tmp_path, monkeypatch) -> None:
    Trainer(tiny_config(tmp_path, name="api_latest")).run()
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    client = TestClient(create_app())

    response = client.get("/experiments/api_latest/latest/results")

    assert response.status_code == 200
    assert response.json()["experiment_name"] == "api_latest"


def test_api_get_results_rejects_unsafe_path_component(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    client = TestClient(create_app())

    response = client.get("/experiments/bad%20name/latest/results")

    assert response.status_code == 400


def test_api_get_results_returns_404_for_missing_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    client = TestClient(create_app())

    response = client.get("/experiments/missing/latest/results")

    assert response.status_code == 404


def test_api_get_compare_leaderboard(tmp_path, monkeypatch) -> None:
    CompareRunner(_compare_config(tmp_path, name="api_compare_board")).run()
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    client = TestClient(create_app())

    response = client.get("/experiments/api_compare_board/latest/leaderboard")
    rows = response.json()

    assert response.status_code == 200
    assert len(rows) == 2
    assert all(isinstance(row["model_params"], dict) for row in rows)


def test_api_get_leaderboard_supports_latest(tmp_path, monkeypatch) -> None:
    CompareRunner(_compare_config(tmp_path, name="api_latest_board")).run()
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    client = TestClient(create_app())

    response = client.get("/experiments/api_latest_board/latest/leaderboard")

    assert response.status_code == 200
    assert response.json()


def test_api_get_leaderboard_returns_404_for_train_run(tmp_path, monkeypatch) -> None:
    Trainer(tiny_config(tmp_path, name="api_train_no_board")).run()
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    client = TestClient(create_app())

    response = client.get("/experiments/api_train_no_board/latest/leaderboard")

    assert response.status_code == 404


def test_api_get_leaderboard_rejects_unsafe_path_component(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(experiments, "RUNS_ROOT", tmp_path)
    client = TestClient(create_app())

    response = client.get("/experiments/safe_experiment/bad%20run/leaderboard")

    assert response.status_code == 400


def test_api_compare_endpoint_runs(tmp_path, monkeypatch) -> None:
    safe_root = tmp_path / "safe_runs"
    monkeypatch.setattr(experiments, "RUNS_ROOT", safe_root)
    client = TestClient(create_app())
    config = _compare_config(tmp_path / "requested", name="api_compare_run").model_dump(mode="json")

    response = client.post("/experiments/compare", json=config)
    payload = response.json()

    assert response.status_code == 200
    assert payload["run_type"] == "compare"
    assert len(payload["rows"]) == 2
    assert Path(payload["compare_run_dir"]).is_relative_to(safe_root)
    assert (safe_root / "api_compare_run" / "latest" / "results.json").exists()


def test_api_compare_overrides_unsafe_output_dir(tmp_path, monkeypatch) -> None:
    safe_root = tmp_path / "safe_runs"
    unsafe_root = tmp_path / "unsafe_runs"
    monkeypatch.setattr(experiments, "RUNS_ROOT", safe_root)
    client = TestClient(create_app())
    config = _compare_config(unsafe_root, name="api_compare_safe").model_dump(mode="json")

    response = client.post("/experiments/compare", json=config)
    payload = response.json()

    assert response.status_code == 200
    assert Path(payload["compare_run_dir"]).is_relative_to(safe_root)
    assert (safe_root / "api_compare_safe" / "latest" / "results.json").exists()
    assert not (unsafe_root / "api_compare_safe" / "latest" / "results.json").exists()


def test_api_compare_rejects_unsafe_experiment_name(tmp_path, monkeypatch) -> None:
    safe_root = tmp_path / "safe_runs"
    monkeypatch.setattr(experiments, "RUNS_ROOT", safe_root)
    client = TestClient(create_app())
    config = _compare_config(tmp_path / "requested", name="api_compare_safe").model_dump(
        mode="json"
    )
    config["experiment"]["name"] = "../escape"

    response = client.post("/experiments/compare", json=config)

    assert response.status_code == 422
    assert "experiment.name must be a safe path component" in response.text


def test_api_compare_returns_leaderboard_paths(tmp_path, monkeypatch) -> None:
    safe_root = tmp_path / "safe_runs"
    monkeypatch.setattr(experiments, "RUNS_ROOT", safe_root)
    client = TestClient(create_app())
    config = _compare_config(tmp_path / "requested", name="api_compare_paths").model_dump(
        mode="json"
    )

    response = client.post("/experiments/compare", json=config)
    payload = response.json()

    assert response.status_code == 200
    assert Path(payload["leaderboard_json_path"]).is_file()
    assert Path(payload["leaderboard_csv_path"]).is_file()


def test_api_compare_results_can_be_fetched_after_run(tmp_path, monkeypatch) -> None:
    safe_root = tmp_path / "safe_runs"
    monkeypatch.setattr(experiments, "RUNS_ROOT", safe_root)
    client = TestClient(create_app())
    config = _compare_config(tmp_path / "requested", name="api_compare_fetch").model_dump(
        mode="json"
    )

    compare_response = client.post("/experiments/compare", json=config)
    results_response = client.get("/experiments/api_compare_fetch/latest/results")

    assert compare_response.status_code == 200
    assert results_response.status_code == 200
    assert results_response.json()["compare_run_id"] == compare_response.json()["compare_run_id"]
