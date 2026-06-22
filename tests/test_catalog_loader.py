from __future__ import annotations

import pytest

from ts_platform.data.catalog import DatasetCatalog, DatasetMetadata
from ts_platform.data.catalog_loader import load_dataset_catalog, register_dataset_catalog


def test_dataset_catalog_loader() -> None:
    metadata = load_dataset_catalog("configs/datasets/local_csv.yaml")

    assert len(metadata) == 1
    assert metadata[0].name == "tiny_csv"
    assert metadata[0].dataset_type == "csv"
    assert metadata[0].target_cols == ["value"]

    registered = register_dataset_catalog("configs/datasets/local_csv.yaml")
    assert registered[0].path == "tests/fixtures/tiny_series.csv"


def test_public_time_series_catalog_has_source_links() -> None:
    metadata = load_dataset_catalog("configs/datasets/public_time_series.yaml")

    names = {item.name for item in metadata}
    assert len(metadata) >= 20
    assert {
        "etth1",
        "electricity_load_diagrams_2011_2014",
        "monash_time_series_forecasting_repository",
    }.issubset(names)
    assert all(item.source.startswith(("http://", "https://")) for item in metadata)


def test_catalog_duplicate_name_overwrites_documented() -> None:
    catalog = DatasetCatalog()
    catalog.register(
        DatasetMetadata(
            name="demo",
            domain="old",
            description="old description",
            source="old",
        )
    )
    catalog.register(
        DatasetMetadata(
            name="Demo",
            domain="new",
            description="new description",
            source="new",
        )
    )

    assert catalog.get("demo").domain == "new"
    assert catalog.get("demo").description == "new description"


def test_catalog_loader_rejects_invalid_schema(tmp_path) -> None:
    catalog_path = tmp_path / "invalid.yaml"
    catalog_path.write_text("datasets:\n  - name: bad\n", encoding="utf-8")

    with pytest.raises(ValueError, match="field 'dataset_type'"):
        load_dataset_catalog(catalog_path)


def test_catalog_loader_rejects_missing_name(tmp_path) -> None:
    catalog_path = tmp_path / "missing_name.yaml"
    catalog_path.write_text(
        "datasets:\n"
        "  - dataset_type: csv\n"
        "    domain: demo\n"
        "    description: Missing name\n"
        "    path: tests/fixtures/tiny_series.csv\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="field 'name'"):
        load_dataset_catalog(catalog_path)


def test_catalog_loader_rejects_csv_without_path(tmp_path) -> None:
    catalog_path = tmp_path / "csv_without_path.yaml"
    catalog_path.write_text(
        "datasets:\n"
        "  - name: no_path\n"
        "    dataset_type: csv\n"
        "    domain: demo\n"
        "    description: Missing path\n"
        "    target_cols: [value]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="path.*required for csv"):
        load_dataset_catalog(catalog_path)


def test_catalog_loader_rejects_target_cols_string(tmp_path) -> None:
    catalog_path = tmp_path / "target_cols_string.yaml"
    catalog_path.write_text(
        "datasets:\n"
        "  - name: bad_targets\n"
        "    dataset_type: csv\n"
        "    domain: demo\n"
        "    description: Bad target cols\n"
        "    path: tests/fixtures/tiny_series.csv\n"
        "    target_cols: value\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target_cols"):
        load_dataset_catalog(catalog_path)


def test_catalog_loader_accepts_valid_csv_metadata(tmp_path) -> None:
    catalog_path = tmp_path / "valid_csv.yaml"
    catalog_path.write_text(
        "datasets:\n"
        "  - name: valid_csv\n"
        "    dataset_type: csv\n"
        "    domain: demo\n"
        "    description: Valid CSV\n"
        "    path: tests/fixtures/tiny_series.csv\n"
        "    timestamp_col: timestamp\n"
        "    target_cols: [value]\n"
        "    feature_cols: [temperature]\n"
        "    frequency: D\n"
        "    license: test-fixture\n",
        encoding="utf-8",
    )

    metadata = load_dataset_catalog(catalog_path)

    assert metadata[0].name == "valid_csv"
    assert metadata[0].path == "tests/fixtures/tiny_series.csv"
    assert metadata[0].timestamp_col == "timestamp"
    assert metadata[0].target_cols == ["value"]
    assert metadata[0].feature_cols == ["temperature"]
