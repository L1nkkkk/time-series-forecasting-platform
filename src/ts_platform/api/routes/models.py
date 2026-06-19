"""Model API routes."""

from __future__ import annotations

from fastapi import APIRouter

from ts_platform.models.registry import registered_model_names

router = APIRouter()


@router.get("/models")
def list_models() -> dict[str, list[str]]:
    """Return registered model names."""

    return {"models": registered_model_names()}
