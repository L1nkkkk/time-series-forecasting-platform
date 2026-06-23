"""Local tooling routes matching CLI operations."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, NoReturn

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ts_platform.api.routes.errors import raise_execution_http_error
from ts_platform.api.services.artifact_service import (
    ArtifactAccessForbiddenError,
    ArtifactAccessPolicy,
    ArtifactService,
    ArtifactTooLargeError,
)
from ts_platform.api.services.compare_service import compare_with_safe_output_dir
from ts_platform.api.services.experiment_store import (
    CorruptExperimentArtifactError,
    ExperimentArtifactNotFoundError,
    ExperimentStore,
    UnsafePathComponentError,
)
from ts_platform.api.services.training_service import train_with_safe_output_dir
from ts_platform.api.settings import APISettings
from ts_platform.config.compare_loader import load_compare_config
from ts_platform.config.loader import load_config
from ts_platform.data.catalog_loader import load_dataset_catalog
from ts_platform.data.catalog_tools import (
    config_from_catalog,
    profile_catalog,
    save_generated_config,
)
from ts_platform.data.profile import profile_csv_dataset

router = APIRouter()
API_SETTINGS = APISettings()


class ConfigPathRequest(BaseModel):
    """Local config path request."""

    config_path: str = Field(min_length=1)


class ProfileCsvRequest(BaseModel):
    """Profile one local CSV dataset."""

    path: str = Field(min_length=1)
    target_cols: list[str] = Field(min_length=1)
    timestamp_col: str | None = None
    input_len: int | None = Field(default=None, gt=0)
    output_len: int | None = Field(default=None, gt=0)
    name: str | None = None


class CatalogProfileRequest(BaseModel):
    """Profile CSV entries in a local catalog."""

    catalog_path: str = Field(min_length=1)
    input_len: int | None = Field(default=None, gt=0)
    output_len: int | None = Field(default=None, gt=0)


class CatalogConfigRequest(BaseModel):
    """Generate a training config from a local catalog entry."""

    catalog_path: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    input_len: int = Field(gt=0)
    output_len: int = Field(gt=0)
    model: str = Field(min_length=1)
    epochs: int = Field(gt=0)
    batch_size: int = Field(default=8, gt=0)
    output_path: str | None = None


class RunLookupRequest(BaseModel):
    """Read one experiment artifact from a trusted local runs root."""

    runs_root: str = Field(default="runs", min_length=1)
    experiment: str = Field(min_length=1)
    run: str = Field(default="latest", min_length=1)


class ArtifactLookupRequest(RunLookupRequest):
    """Read one named artifact from a trusted local runs root."""

    artifact: str = Field(min_length=1)


@router.post("/configs/train/run")
def run_train_config(payload: ConfigPathRequest) -> dict[str, Any]:
    """Run a training config from a local YAML or JSON path."""

    try:
        config = load_config(payload.config_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        return train_with_safe_output_dir(config, runs_root=API_SETTINGS.runs_root)
    except Exception as exc:
        raise_execution_http_error("training", exc)


@router.post("/configs/compare/run")
def run_compare_config(payload: ConfigPathRequest) -> dict[str, Any]:
    """Run a compare config from a local YAML or JSON path."""

    try:
        config = load_compare_config(payload.config_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        return compare_with_safe_output_dir(config, runs_root=API_SETTINGS.runs_root)
    except Exception as exc:
        raise_execution_http_error("compare", exc)


@router.post("/datasets/profile-csv")
def profile_csv(payload: ProfileCsvRequest) -> dict[str, Any]:
    """Profile one local CSV path."""

    try:
        return profile_csv_dataset(
            path=payload.path,
            target_cols=payload.target_cols,
            timestamp_col=payload.timestamp_col,
            input_len=payload.input_len,
            output_len=payload.output_len,
            name=payload.name,
        ).to_dict()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/datasets/catalog/profile")
def profile_dataset_catalog(payload: CatalogProfileRequest) -> dict[str, Any]:
    """Profile CSV entries from a local dataset catalog."""

    try:
        return {
            "profiles": profile_catalog(
                payload.catalog_path,
                input_len=payload.input_len,
                output_len=payload.output_len,
            )
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/datasets/catalog/list")
def list_dataset_catalog(payload: CatalogProfileRequest) -> dict[str, Any]:
    """List metadata entries from a local dataset catalog."""

    try:
        return {"datasets": [asdict(item) for item in load_dataset_catalog(payload.catalog_path)]}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/datasets/catalog/config")
def generate_config_from_catalog(payload: CatalogConfigRequest) -> dict[str, Any]:
    """Generate a training config from one local catalog entry."""

    try:
        config = config_from_catalog(
            payload.catalog_path,
            dataset_name=payload.dataset,
            input_len=payload.input_len,
            output_len=payload.output_len,
            model_name=payload.model,
            epochs=payload.epochs,
            batch_size=payload.batch_size,
        )
        response: dict[str, Any] = {"config": config.model_dump(mode="json")}
        if payload.output_path:
            response["output"] = str(save_generated_config(config, Path(payload.output_path)))
        return response
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/tools/experiments/results")
def lookup_run_results(payload: RunLookupRequest) -> dict[str, Any]:
    """Read results JSON from a trusted local runs root."""

    try:
        return ExperimentStore(Path(payload.runs_root)).read_results(
            payload.experiment, payload.run
        )
    except (
        UnsafePathComponentError,
        ExperimentArtifactNotFoundError,
        CorruptExperimentArtifactError,
    ) as exc:
        _raise_lookup_http_error(exc)


@router.post("/tools/experiments/leaderboard")
def lookup_run_leaderboard(payload: RunLookupRequest) -> list[dict[str, Any]]:
    """Read leaderboard JSON from a trusted local runs root."""

    try:
        return ExperimentStore(Path(payload.runs_root)).read_leaderboard(
            payload.experiment, payload.run
        )
    except (
        UnsafePathComponentError,
        ExperimentArtifactNotFoundError,
        CorruptExperimentArtifactError,
    ) as exc:
        _raise_lookup_http_error(exc)


@router.post("/tools/experiments/artifacts")
def lookup_run_artifacts(payload: RunLookupRequest) -> dict[str, Any]:
    """Read an artifact manifest from a trusted local runs root."""

    try:
        return ExperimentStore(Path(payload.runs_root)).read_artifacts(
            payload.experiment, payload.run
        )
    except (
        UnsafePathComponentError,
        ExperimentArtifactNotFoundError,
        CorruptExperimentArtifactError,
    ) as exc:
        _raise_lookup_http_error(exc)


@router.post("/tools/experiments/artifact")
def download_lookup_artifact(payload: ArtifactLookupRequest) -> FileResponse:
    """Download one manifest-backed artifact from a trusted local runs root."""

    try:
        artifact = ArtifactService(
            Path(payload.runs_root),
            policy=_artifact_access_policy(),
        ).resolve_artifact(payload.experiment, payload.run, payload.artifact)
    except (
        UnsafePathComponentError,
        ExperimentArtifactNotFoundError,
        ArtifactAccessForbiddenError,
        ArtifactTooLargeError,
        CorruptExperimentArtifactError,
    ) as exc:
        _raise_lookup_http_error(exc)
    return FileResponse(
        artifact.path,
        media_type=artifact.media_type,
        filename=artifact.path.name,
    )


def _artifact_access_policy() -> ArtifactAccessPolicy:
    return ArtifactAccessPolicy(
        max_bytes=API_SETTINGS.artifact_max_bytes,
        allow_checkpoint_download=API_SETTINGS.allow_checkpoint_download,
        allowed_kinds=frozenset(API_SETTINGS.artifact_allowed_kinds),
    )


def _raise_lookup_http_error(
    exc: UnsafePathComponentError
    | ExperimentArtifactNotFoundError
    | ArtifactAccessForbiddenError
    | ArtifactTooLargeError
    | CorruptExperimentArtifactError,
) -> NoReturn:
    if isinstance(exc, UnsafePathComponentError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, ExperimentArtifactNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, ArtifactAccessForbiddenError):
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if isinstance(exc, ArtifactTooLargeError):
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail=str(exc)) from exc
