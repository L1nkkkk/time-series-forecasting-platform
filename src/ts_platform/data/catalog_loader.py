"""Load local dataset catalog metadata files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

from ts_platform.data.catalog import DATASET_CATALOG, DatasetMetadata


def load_dataset_catalog(path: str | Path) -> list[DatasetMetadata]:
    """Load and validate a local dataset catalog YAML file."""

    catalog_path = Path(path)
    if not catalog_path.exists():
        msg = f"Dataset catalog file does not exist: {catalog_path}"
        raise FileNotFoundError(msg)
    raw = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = "Dataset catalog root must be a mapping"
        raise ValueError(msg)
    entries = raw.get("datasets")
    if not isinstance(entries, list):
        msg = "Dataset catalog must contain a 'datasets' list"
        raise ValueError(msg)
    return [
        _metadata_from_entry(cast(dict[str, Any], entry), index)
        for index, entry in enumerate(entries)
    ]


def register_dataset_catalog(path: str | Path) -> list[DatasetMetadata]:
    """Load catalog metadata and register it in the global catalog."""

    metadata = load_dataset_catalog(path)
    for item in metadata:
        DATASET_CATALOG.register(item)
    return metadata


def _metadata_from_entry(entry: dict[str, Any], index: int) -> DatasetMetadata:
    if not isinstance(entry, dict):
        msg = f"Dataset catalog entry {index} must be a mapping"
        raise ValueError(msg)
    required = ["name", "dataset_type", "domain", "description"]
    for field in required:
        if not isinstance(entry.get(field), str) or not entry[field]:
            msg = f"Dataset catalog entry {index} field {field!r} must be a non-empty string"
            raise ValueError(msg)
    target_cols = entry.get("target_cols")
    if target_cols is not None and (
        not isinstance(target_cols, list) or not all(isinstance(item, str) for item in target_cols)
    ):
        msg = f"Dataset catalog entry {index} field 'target_cols' must be a list of strings"
        raise ValueError(msg)
    feature_cols = entry.get("feature_cols")
    if feature_cols is not None and (
        not isinstance(feature_cols, list)
        or not all(isinstance(item, str) for item in feature_cols)
    ):
        msg = f"Dataset catalog entry {index} field 'feature_cols' must be a list of strings"
        raise ValueError(msg)
    timestamp_col = entry.get("timestamp_col")
    if timestamp_col is not None and not isinstance(timestamp_col, str):
        msg = f"Dataset catalog entry {index} field 'timestamp_col' must be a string"
        raise ValueError(msg)
    path = entry.get("path")
    if path is not None and not isinstance(path, str):
        msg = f"Dataset catalog entry {index} field 'path' must be a string"
        raise ValueError(msg)
    if entry["dataset_type"] == "csv" and not path:
        msg = f"Dataset catalog entry {index} field 'path' is required for csv datasets"
        raise ValueError(msg)
    license_name = entry.get("license")
    if license_name is not None and not isinstance(license_name, str):
        msg = f"Dataset catalog entry {index} field 'license' must be a string"
        raise ValueError(msg)
    citation = entry.get("citation")
    if citation is not None and not isinstance(citation, str):
        msg = f"Dataset catalog entry {index} field 'citation' must be a string"
        raise ValueError(msg)
    frequency = entry.get("frequency")
    if frequency is not None and not isinstance(frequency, str):
        msg = f"Dataset catalog entry {index} field 'frequency' must be a string"
        raise ValueError(msg)
    source = entry.get("source")
    if source is not None and not isinstance(source, str):
        msg = f"Dataset catalog entry {index} field 'source' must be a string"
        raise ValueError(msg)

    return DatasetMetadata(
        name=entry["name"],
        dataset_type=entry["dataset_type"],
        domain=entry["domain"],
        description=entry["description"],
        source=source or path or "local",
        frequency=frequency,
        path=path,
        license=license_name,
        citation=citation,
        target_cols=target_cols,
        feature_cols=feature_cols,
        timestamp_col=timestamp_col,
    )
