"""Shared HTTP error translation helpers for API routes."""

from __future__ import annotations

from typing import NoReturn

from fastapi import HTTPException

from ts_platform.runner.devices import DeviceUnavailableError


def raise_execution_http_error(action: str, exc: Exception) -> NoReturn:
    """Translate expected execution failures into concise HTTP responses."""

    if isinstance(exc, DeviceUnavailableError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    detail = f"{action} failed: {exc}"
    raise HTTPException(status_code=500, detail=detail) from exc
