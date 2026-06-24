from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from ts_platform.api.app import create_app
from ts_platform.api.routes import datasets
from ts_platform.data.assets import (
    clear_dataset_asset,
    dataset_asset_status,
    list_dataset_cache,
    prepare_dataset_asset,
)
from ts_platform.data.catalog import DATASET_CATALOG, DatasetMetadata


def test_prepare_dataset_asset_writes_manifest_catalog_and_config(tmp_path) -> None:
    metadata = DatasetMetadata(
        name="remote_tiny",
        dataset_type="external_csv",
        domain="demo",
        description="Remote tiny CSV.",
        source="https://example.test/remote_tiny.csv",
        download_url="https://example.test/remote_tiny.csv",
        archive_format="raw_csv",
        local_path="remote_tiny.csv",
        version="v1",
        timestamp_col="timestamp",
        target_cols=["value"],
    )
    download_calls: list[str] = []

    def downloader(url: str, destination: Path) -> None:
        download_calls.append(url)
        destination.write_text(_tiny_csv_text(), encoding="utf-8")

    record = prepare_dataset_asset(
        metadata,
        cache_root=tmp_path / "cache",
        external_root=tmp_path / "external",
        downloader=downloader,
    )

    assert record["prepared"] is True
    assert record["row_count"] == 60
    assert Path(record["path"]).exists()
    assert Path(record["catalog_path"]).exists()
    assert Path(record["config_path"]).exists()
    assert download_calls == ["https://example.test/remote_tiny.csv"]

    second = prepare_dataset_asset(
        metadata,
        cache_root=tmp_path / "cache",
        external_root=tmp_path / "external",
        downloader=lambda _url, _destination: (_ for _ in ()).throw(AssertionError("redownloaded")),
    )

    assert second["prepared"] is True
    assert dataset_asset_status(metadata, cache_root=tmp_path / "cache")["prepared"] is True
    assert list_dataset_cache(cache_root=tmp_path / "cache")["assets"][0]["name"] == "remote_tiny"


def test_prepare_raw_matrix_dataset_converts_to_csv(tmp_path) -> None:
    metadata = DatasetMetadata(
        name="remote_matrix",
        dataset_type="external_csv",
        domain="demo",
        description="Remote matrix text.",
        source="https://example.test/matrix.txt",
        download_url="https://example.test/matrix.txt",
        archive_format="raw_matrix",
        local_path="remote_matrix.csv",
        version="v1",
        target_cols=["value_0"],
    )

    def downloader(_url: str, destination: Path) -> None:
        rows = [f"{index},{index + 1}" for index in range(60)]
        destination.write_text("\n".join(rows), encoding="utf-8")

    record = prepare_dataset_asset(
        metadata,
        cache_root=tmp_path / "cache",
        external_root=tmp_path / "external",
        downloader=downloader,
    )

    assert record["columns"] == ["value_0", "value_1"]
    assert Path(record["path"]).read_text(encoding="utf-8").startswith("value_0,value_1")


def test_clear_dataset_asset_removes_manifest_entry_and_external_dir(tmp_path) -> None:
    metadata = DatasetMetadata(
        name="remote_clear",
        dataset_type="external_csv",
        domain="demo",
        description="Remote clear CSV.",
        source="https://example.test/remote_clear.csv",
        download_url="https://example.test/remote_clear.csv",
        archive_format="raw_csv",
        local_path="remote_clear.csv",
        target_cols=["value"],
    )

    prepare_dataset_asset(
        metadata,
        cache_root=tmp_path / "cache",
        external_root=tmp_path / "external",
        downloader=lambda _url, destination: destination.write_text(
            _tiny_csv_text(),
            encoding="utf-8",
        ),
    )

    payload = clear_dataset_asset(
        "remote_clear",
        cache_root=tmp_path / "cache",
        external_root=tmp_path / "external",
    )

    manifest = json.loads((tmp_path / "cache" / "manifest.json").read_text(encoding="utf-8"))
    assert payload == {"removed": True, "dataset": "remote_clear"}
    assert "remote_clear" not in manifest["assets"]
    assert not (tmp_path / "external" / "remote_clear").exists()


def test_api_dataset_asset_routes(monkeypatch) -> None:
    DATASET_CATALOG.register(
        DatasetMetadata(
            name="api_asset",
            dataset_type="external_csv",
            domain="demo",
            description="API asset.",
            source="https://example.test/api_asset.csv",
            download_url="https://example.test/api_asset.csv",
            archive_format="raw_csv",
            local_path="api_asset.csv",
            target_cols=["value"],
        )
    )
    status = {
        "name": "api_asset",
        "prepared": True,
        "path": "data/external/api_asset/v1/api_asset.csv",
        "target_cols": ["value"],
        "feature_cols": [],
        "timestamp_col": None,
        "dataset_type": "external_csv",
        "domain": "demo",
        "description": "API asset.",
        "source": "https://example.test/api_asset.csv",
        "download_url": "https://example.test/api_asset.csv",
        "archive_format": "raw_csv",
        "frequency": None,
        "license": None,
        "citation": None,
        "checksum": None,
    }
    monkeypatch.setattr(datasets, "dataset_asset_status", lambda _metadata: status)
    monkeypatch.setattr(datasets, "prepare_dataset_asset", lambda _metadata: status)
    monkeypatch.setattr(datasets, "list_dataset_cache", lambda: {"assets": [status]})

    client = TestClient(create_app())

    assert client.get("/datasets/cache").json()["assets"][0]["name"] == "api_asset"
    assert client.get("/datasets/api_asset/asset").json()["prepared"] is True
    assert client.post("/datasets/api_asset/prepare").json()["path"].endswith("api_asset.csv")
    detail = client.get("/datasets/api_asset").json()
    assert detail["prepared"] is True
    assert detail["dataset_type"] == "csv"


def _tiny_csv_text() -> str:
    rows = ["timestamp,value"]
    rows.extend(f"2024-01-{(index % 28) + 1:02d},{index}" for index in range(60))
    return "\n".join(rows)
