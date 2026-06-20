"""Dataset API routes."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ts_platform.data import DATASET_CATALOG, DATASET_REGISTRY
from ts_platform.data.profile import profile_csv_dataset

router = APIRouter()


@router.get("/datasets")
def list_datasets() -> dict[str, object]:
    """Return registered datasets."""

    return {"datasets": DATASET_CATALOG.list(), "names": DATASET_REGISTRY.names()}


@router.get("/datasets/{dataset_name}")
def get_dataset_detail(dataset_name: str) -> dict[str, Any]:
    """Return catalog metadata for one dataset."""

    try:
        return asdict(DATASET_CATALOG.get(dataset_name))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/datasets/{dataset_name}/profile")
def get_dataset_profile(
    dataset_name: str,
    request: Request,
    input_len: int | None = None,
    output_len: int | None = None,
) -> dict[str, Any]:
    """Profile one local CSV dataset from catalog metadata."""

    unsupported_queries = set(request.query_params) - {"input_len", "output_len"}
    if unsupported_queries:
        detail = f"unsupported query parameters: {sorted(unsupported_queries)}"
        raise HTTPException(status_code=422, detail=detail)
    try:
        metadata = DATASET_CATALOG.get(dataset_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if metadata.dataset_type != "csv":
        raise HTTPException(status_code=400, detail="dataset profile only supports csv datasets")
    if metadata.path is None or metadata.target_cols is None:
        detail = "csv dataset metadata must include path and target_cols"
        raise HTTPException(status_code=422, detail=detail)
    try:
        profile = profile_csv_dataset(
            path=metadata.path,
            target_cols=metadata.target_cols,
            timestamp_col=metadata.timestamp_col,
            input_len=input_len,
            output_len=output_len,
            name=metadata.name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return profile.to_dict()
