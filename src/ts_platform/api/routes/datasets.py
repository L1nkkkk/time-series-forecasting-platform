"""Dataset API routes."""

from __future__ import annotations

from fastapi import APIRouter

from ts_platform.data import DATASET_CATALOG, DATASET_REGISTRY

router = APIRouter()


@router.get("/datasets")
def list_datasets() -> dict[str, object]:
    """Return registered datasets."""

    return {"datasets": DATASET_CATALOG.list(), "names": DATASET_REGISTRY.names()}
