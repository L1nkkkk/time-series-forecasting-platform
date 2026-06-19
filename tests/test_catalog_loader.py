from __future__ import annotations

from ts_platform.data.catalog_loader import load_dataset_catalog, register_dataset_catalog


def test_dataset_catalog_loader() -> None:
    metadata = load_dataset_catalog("configs/datasets/local_csv.yaml")

    assert len(metadata) == 1
    assert metadata[0].name == "tiny_csv"
    assert metadata[0].dataset_type == "csv"
    assert metadata[0].target_cols == ["value"]

    registered = register_dataset_catalog("configs/datasets/local_csv.yaml")
    assert registered[0].path == "tests/fixtures/tiny_series.csv"
