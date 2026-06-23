"""Dataset API routes."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from ts_platform.config.schema import validate_safe_path_component
from ts_platform.data import DATASET_CATALOG, DATASET_REGISTRY
from ts_platform.data.catalog import DatasetMetadata
from ts_platform.data.profile import profile_csv_dataset

router = APIRouter()
UPLOADS_ROOT = Path("data/uploads")
USER_DATASETS_PATH = Path("data/user_datasets.json")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_METADATA_TEXT_LENGTH = 4096
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class CSVUploadRequest(BaseModel):
    """CSV upload payload read from a browser file picker."""

    filename: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1)

    @field_validator("filename")
    @classmethod
    def validate_csv_filename(cls, value: str) -> str:
        if not value.lower().endswith(".csv"):
            msg = "uploaded dataset file must be a .csv"
            raise ValueError(msg)
        return value


class UserDatasetRequest(BaseModel):
    """User-maintained CSV dataset metadata."""

    name: str = Field(min_length=1, max_length=80)
    dataset_type: str = "csv"
    domain: str = Field(default="custom", min_length=1, max_length=80)
    description: str = Field(
        default="User supplied local CSV dataset.",
        min_length=1,
        max_length=MAX_METADATA_TEXT_LENGTH,
    )
    source: str | None = Field(default=None, max_length=MAX_METADATA_TEXT_LENGTH)
    path: str = Field(min_length=1, max_length=MAX_METADATA_TEXT_LENGTH)
    timestamp_col: str | None = Field(default=None, max_length=255)
    target_cols: list[str] = Field(min_length=1)
    feature_cols: list[str] = Field(default_factory=list)
    frequency: str | None = Field(default=None, max_length=255)
    license: str | None = Field(default="user-supplied", max_length=255)
    citation: str | None = Field(default=None, max_length=MAX_METADATA_TEXT_LENGTH)

    @field_validator("name")
    @classmethod
    def validate_dataset_name(cls, value: str) -> str:
        return validate_safe_path_component(value, field_name="dataset.name")

    @field_validator("dataset_type")
    @classmethod
    def validate_dataset_type(cls, value: str) -> str:
        if value != "csv":
            msg = "user dataset metadata currently supports only csv datasets"
            raise ValueError(msg)
        return value

    @field_validator("path")
    @classmethod
    def validate_local_csv_path(cls, value: str) -> str:
        if value.lower().startswith(("http://", "https://")):
            msg = "user CSV dataset path must be a local path"
            raise ValueError(msg)
        return value

    @field_validator("target_cols")
    @classmethod
    def validate_target_column_names(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if not cleaned:
            msg = "column list must contain at least one non-empty string"
            raise ValueError(msg)
        return cleaned

    @field_validator("feature_cols")
    @classmethod
    def validate_feature_column_names(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]


@router.get("/datasets")
def list_datasets() -> dict[str, object]:
    """Return registered datasets."""

    rows = _merged_dataset_rows()
    return {"datasets": rows, "names": DATASET_REGISTRY.names()}


@router.post("/datasets/upload-csv")
def upload_csv_dataset(payload: CSVUploadRequest) -> dict[str, object]:
    """Save a browser-selected CSV file into a local managed upload directory."""

    content_bytes = payload.content.encode("utf-8")
    if len(content_bytes) > MAX_UPLOAD_BYTES:
        detail = f"CSV upload exceeds {MAX_UPLOAD_BYTES} bytes"
        raise HTTPException(status_code=413, detail=detail)

    columns = _read_csv_columns(payload.content)
    if not columns:
        raise HTTPException(status_code=422, detail="CSV header row is required")

    filename = _safe_upload_filename(payload.filename)
    destination = UPLOADS_ROOT / filename
    try:
        UPLOADS_ROOT.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload.content, encoding="utf-8", newline="")
    except OSError as exc:
        msg = f"CSV upload cannot be written: {destination}"
        raise HTTPException(status_code=500, detail=msg) from exc
    return {
        "filename": filename,
        "path": destination.as_posix(),
        "columns": columns,
        "size_bytes": len(content_bytes),
    }


@router.post("/datasets/user")
def save_user_dataset(payload: UserDatasetRequest) -> dict[str, Any]:
    """Persist one user-supplied local CSV dataset metadata entry."""

    metadata = _metadata_from_user_payload(payload)
    rows = [
        item
        for item in _load_user_dataset_metadata()
        if item.name.strip().lower() != metadata.name.strip().lower()
    ]
    rows.append(metadata)
    _save_user_dataset_metadata(rows)
    return _user_metadata_to_dict(metadata)


@router.delete("/datasets/user")
def clear_user_datasets() -> dict[str, object]:
    """Remove all persisted user dataset metadata entries."""

    _save_user_dataset_metadata([])
    return {"datasets": []}


@router.delete("/datasets/user/{dataset_name}")
def delete_user_dataset(dataset_name: str) -> dict[str, object]:
    """Remove one persisted user dataset metadata entry by name."""

    try:
        safe_name = validate_safe_path_component(dataset_name, field_name="dataset.name")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    rows = [
        item
        for item in _load_user_dataset_metadata()
        if item.name.strip().lower() != safe_name.strip().lower()
    ]
    _save_user_dataset_metadata(rows)
    return {"datasets": [_user_metadata_to_dict(item) for item in rows]}


@router.get("/datasets/{dataset_name}")
def get_dataset_detail(dataset_name: str) -> dict[str, Any]:
    """Return catalog metadata for one dataset."""

    try:
        return _metadata_detail(dataset_name)
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
        metadata = _get_dataset_metadata(dataset_name)
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


def _safe_upload_filename(filename: str) -> str:
    raw_name = Path(filename).name
    safe_name = SAFE_FILENAME_RE.sub("_", raw_name).strip("._")
    if not safe_name:
        safe_name = "dataset.csv"
    stem = Path(safe_name).stem[:80] or "dataset"
    suffix = Path(safe_name).suffix.lower() or ".csv"
    return f"{stem}_{uuid4().hex[:10]}{suffix}"


def _read_csv_columns(content: str) -> list[str]:
    try:
        reader = csv.reader(StringIO(content))
        row = next(reader, [])
    except csv.Error as exc:
        raise HTTPException(status_code=422, detail=f"invalid CSV header: {exc}") from exc
    return [column.strip() for column in row if column.strip()]


def _merged_dataset_rows() -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for catalog_item in DATASET_CATALOG.list():
        name = catalog_item.get("name")
        if isinstance(name, str) and name.strip():
            rows[name.strip().lower()] = catalog_item
    for user_item in _load_user_dataset_metadata():
        rows[user_item.name.strip().lower()] = _user_metadata_to_dict(user_item)
    return list(rows.values())


def _metadata_detail(dataset_name: str) -> dict[str, Any]:
    metadata = _get_dataset_metadata(dataset_name)
    payload = asdict(metadata)
    normalized = dataset_name.strip().lower()
    if any(item.name.strip().lower() == normalized for item in _load_user_dataset_metadata()):
        payload["user_defined"] = True
    return payload


def _get_dataset_metadata(dataset_name: str) -> DatasetMetadata:
    normalized = dataset_name.strip().lower()
    for item in _load_user_dataset_metadata():
        if item.name.strip().lower() == normalized:
            return item
    return DATASET_CATALOG.get(dataset_name)


def _metadata_from_user_payload(payload: UserDatasetRequest) -> DatasetMetadata:
    return DatasetMetadata(
        name=payload.name,
        dataset_type=payload.dataset_type,
        domain=payload.domain,
        description=payload.description,
        source=payload.source or payload.path,
        frequency=payload.frequency,
        path=payload.path,
        license=payload.license,
        citation=payload.citation,
        target_cols=payload.target_cols,
        feature_cols=payload.feature_cols,
        timestamp_col=payload.timestamp_col,
    )


def _load_user_dataset_metadata() -> list[DatasetMetadata]:
    if not USER_DATASETS_PATH.exists():
        return []
    try:
        raw = json.loads(USER_DATASETS_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            msg = "user dataset catalog must be a JSON list"
            raise ValueError(msg)
        return [
            _metadata_from_user_payload(UserDatasetRequest.model_validate(item))
            for item in raw
            if isinstance(item, dict)
        ]
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        msg = f"user dataset catalog cannot be read: {USER_DATASETS_PATH}"
        raise HTTPException(status_code=500, detail=msg) from exc


def _save_user_dataset_metadata(rows: list[DatasetMetadata]) -> None:
    payload = [asdict(item) for item in rows]
    try:
        USER_DATASETS_PATH.parent.mkdir(parents=True, exist_ok=True)
        USER_DATASETS_PATH.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError as exc:
        msg = f"user dataset catalog cannot be written: {USER_DATASETS_PATH}"
        raise HTTPException(status_code=500, detail=msg) from exc


def _user_metadata_to_dict(metadata: DatasetMetadata) -> dict[str, Any]:
    payload = asdict(metadata)
    payload["user_defined"] = True
    return payload
