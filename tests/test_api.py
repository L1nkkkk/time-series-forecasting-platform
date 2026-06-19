from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from tests.helpers import tiny_config
from ts_platform.api.app import create_app
from ts_platform.api.routes import experiments
from ts_platform.runner.trainer import Trainer


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
