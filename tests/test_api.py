from __future__ import annotations

from fastapi.testclient import TestClient

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
