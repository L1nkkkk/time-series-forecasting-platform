"""Retry and timeout policy primitives for SQLite job prototypes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    """Small local retry policy used by explicit retry operations."""

    max_attempts: int = 3
    stale_after_seconds: int = 3600

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            msg = "max_attempts must be >= 1"
            raise ValueError(msg)
        if self.stale_after_seconds <= 0:
            msg = "stale_after_seconds must be > 0"
            raise ValueError(msg)
