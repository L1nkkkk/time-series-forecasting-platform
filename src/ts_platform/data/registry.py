"""Dataset registry."""

from __future__ import annotations

from typing import Generic, TypeVar

from ts_platform.data.base import ForecastingDataset

T = TypeVar("T")


class Registry(Generic[T]):
    """Name-to-object registry with clear duplicate errors."""

    def __init__(self) -> None:
        self._items: dict[str, T] = {}

    def register(self, name: str, item: T) -> None:
        """Register an item by name."""

        normalized = name.strip().lower()
        if not normalized:
            msg = "registry name must not be empty"
            raise ValueError(msg)
        if normalized in self._items:
            msg = f"registry item already exists: {normalized}"
            raise KeyError(msg)
        self._items[normalized] = item

    def get(self, name: str) -> T:
        """Return a registered item by name."""

        normalized = name.strip().lower()
        try:
            return self._items[normalized]
        except KeyError as exc:
            available = ", ".join(self.names()) or "<none>"
            msg = f"unknown registry item {normalized!r}; available: {available}"
            raise KeyError(msg) from exc

    def names(self) -> list[str]:
        """Return sorted registered names."""

        return sorted(self._items)

    def items(self) -> dict[str, T]:
        """Return a shallow copy of registry items."""

        return dict(self._items)


DATASET_REGISTRY: Registry[type[ForecastingDataset]] = Registry()
