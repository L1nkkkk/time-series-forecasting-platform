"""Dataset catalog metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DatasetMetadata:
    """Human-readable dataset metadata."""

    name: str
    domain: str
    description: str
    source: str
    dataset_type: str = "unknown"
    frequency: str | None = None
    path: str | None = None
    license: str | None = None
    citation: str | None = None
    target_cols: list[str] | None = None
    feature_cols: list[str] | None = None
    timestamp_col: str | None = None
    download_url: str | None = None
    archive_format: str | None = None
    local_path: str | None = None
    version: str | None = None
    checksum: str | None = None
    prepared: bool | None = None


class DatasetCatalog:
    """In-memory metadata catalog for registered datasets."""

    def __init__(self) -> None:
        self._items: dict[str, DatasetMetadata] = {}

    def register(self, metadata: DatasetMetadata) -> None:
        """Register metadata for a dataset, overwriting existing metadata by name."""

        normalized = metadata.name.strip().lower()
        if not normalized:
            msg = "dataset metadata name must not be empty"
            raise ValueError(msg)
        self._items[normalized] = metadata

    def get(self, name: str) -> DatasetMetadata:
        """Return metadata for a dataset."""

        normalized = name.strip().lower()
        try:
            return self._items[normalized]
        except KeyError as exc:
            msg = f"unknown dataset metadata: {normalized}"
            raise KeyError(msg) from exc

    def names(self) -> list[str]:
        """Return registered metadata names."""

        return sorted(self._items)

    def list(self) -> list[dict[str, Any]]:
        """Return catalog metadata as serializable dictionaries."""

        return [asdict(item) for item in self._items.values()]


DATASET_CATALOG = DatasetCatalog()
