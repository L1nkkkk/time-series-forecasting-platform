from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from tests.helpers import tiny_config
from ts_platform.api.app import create_app


def test_api_health_datasets_and_models() -> None:
    client = TestClient(create_app())

    health = client.get("/health")
    datasets = client.get("/datasets")
    models = client.get("/models")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert datasets.status_code == 200
    assert "synthetic" in datasets.json()["names"]
    assert models.status_code == 200
    assert {"naive", "linear", "mlp"}.issubset(set(models.json()["models"]))


def test_api_train_endpoint(tmp_path) -> None:
    client = TestClient(create_app())
    config = tiny_config(tmp_path, name="api").model_dump(mode="json")

    response = client.post("/experiments/train", json=config)
    payload = response.json()

    assert response.status_code == 200
    assert payload["run_dir"]
    assert payload["checkpoint_path"]
    assert payload["test_metrics"]["original"]
    assert Path(payload["checkpoint_path"]).exists()
    assert (tmp_path / "api" / "latest" / "results.json").exists()
