from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from tests.helpers import tiny_config
from ts_platform.api.app import create_app
from ts_platform.api.routes import datasets, demo, experiments
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
    assert datasets.status_code == 200
    assert "synthetic" in datasets.json()["names"]
    assert "csv" in datasets.json()["names"]
    assert any(item["name"] == "tiny_csv" for item in datasets.json()["datasets"])
    assert models.status_code == 200
    assert {"naive", "linear", "mlp", "transformer", "nbeats"}.issubset(
        set(models.json()["models"])
    )


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
    assert "20260622-report-export" in response.text
    assert "export-report-run" in response.text


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
    assert "downloadTextFile" in script.text
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
    assert ".monitor-panels" in styles.text
    assert ".monitor-chart" in styles.text
    assert ".profile-panel" in styles.text
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
        "compare_forecast",
        "compare_model_zoo",
        "compare_feature_forecast",
    ]
    assert payload["train"] == [
        "simple_forecast",
        "csv_forecast",
        "csv_feature_forecast",
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
