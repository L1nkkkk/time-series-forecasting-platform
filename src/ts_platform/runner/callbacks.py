"""Callback extension points for future runner customization."""

from __future__ import annotations

from typing import Protocol


class Callback(Protocol):
    """Minimal callback protocol for future extension."""

    def on_epoch_start(self, epoch: int) -> None:
        """Run at the start of an epoch."""

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Run at the end of an epoch."""
