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


def test_catalog_duplicate_name_overwrites_or_rejects_documented_behavior() -> None:
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
