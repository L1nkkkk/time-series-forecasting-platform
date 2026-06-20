from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from tests.helpers import tiny_config
from ts_platform.api.app import create_app
from ts_platform.api.routes import experiments
from ts_platform.config.compare_schema import CompareConfig, CompareModelConfig
from ts_platform.config.schema import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ScalerConfig,
    TrainingConfig,
)
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
    assert {"naive", "linear", "mlp"}.issubset(set(models.json()["models"]))


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
