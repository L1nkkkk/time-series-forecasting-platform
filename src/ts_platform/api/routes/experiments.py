"""Experiment API routes."""

from __future__ import annotations

from typing import Any, NoReturn

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

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
from ts_platform.config.compare_schema import CompareConfig
from ts_platform.config.schema import PlatformConfig

router = APIRouter()
API_SETTINGS = APISettings()
RUNS_ROOT = API_SETTINGS.runs_root


@router.get("/experiments")
def list_experiments() -> dict[str, list[dict[str, Any]]]:
    """List local experiment metadata from the fixed runs root."""

    return {"experiments": ExperimentStore(RUNS_ROOT).list_experiments()}


@router.post("/experiments/train")
def train_experiment(config: PlatformConfig) -> dict[str, Any]:
    """Run a synchronous training demo from a config payload."""

    try:
        return train_with_safe_output_dir(config, runs_root=RUNS_ROOT)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise_execution_http_error("training", exc)


@router.post("/experiments/compare")
def compare_experiments(config: CompareConfig) -> dict[str, Any]:
    """Run a synchronous compare demo from a compare config payload."""

    try:
        return compare_with_safe_output_dir(config, runs_root=RUNS_ROOT)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise_execution_http_error("compare", exc)


@router.get("/experiments/{experiment_name}/{run_id}/results")
def get_experiment_results(experiment_name: str, run_id: str) -> dict[str, Any]:
    """Read a train or compare run results payload."""

    try:
        return ExperimentStore(RUNS_ROOT).read_results(experiment_name, run_id)
    except (
        UnsafePathComponentError,
        ExperimentArtifactNotFoundError,
        CorruptExperimentArtifactError,
    ) as exc:
        _raise_http_error(exc)


@router.get("/experiments/{experiment_name}/{run_id}/artifacts")
def get_experiment_artifacts(experiment_name: str, run_id: str) -> dict[str, Any]:
    """Read a train or compare run artifact manifest."""

    try:
        return ExperimentStore(RUNS_ROOT).read_artifacts(experiment_name, run_id)
    except (
        UnsafePathComponentError,
        ExperimentArtifactNotFoundError,
        CorruptExperimentArtifactError,
    ) as exc:
        _raise_http_error(exc)


@router.get("/experiments/{experiment_name}/{run_id}/artifacts/{artifact_name}")
def download_experiment_artifact(
    experiment_name: str,
    run_id: str,
    artifact_name: str,
) -> FileResponse:
    """Download one safe artifact registered in a run manifest."""

    try:
        artifact = ArtifactService(RUNS_ROOT, policy=_artifact_access_policy()).resolve_artifact(
            experiment_name,
            run_id,
            artifact_name,
        )
    except (
        UnsafePathComponentError,
        ExperimentArtifactNotFoundError,
        ArtifactAccessForbiddenError,
        ArtifactTooLargeError,
        CorruptExperimentArtifactError,
    ) as exc:
        _raise_http_error(exc)
    return FileResponse(
        artifact.path,
        media_type=artifact.media_type,
        filename=artifact.path.name,
    )


@router.get("/experiments/{experiment_name}/{run_id}/leaderboard")
def get_experiment_leaderboard(experiment_name: str, run_id: str) -> list[dict[str, Any]]:
    """Read a compare run leaderboard payload."""

    try:
        return ExperimentStore(RUNS_ROOT).read_leaderboard(experiment_name, run_id)
    except (
        UnsafePathComponentError,
        ExperimentArtifactNotFoundError,
        CorruptExperimentArtifactError,
    ) as exc:
        _raise_http_error(exc)


def _raise_http_error(
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


def _artifact_access_policy() -> ArtifactAccessPolicy:
    return ArtifactAccessPolicy(
        max_bytes=API_SETTINGS.artifact_max_bytes,
        allow_checkpoint_download=API_SETTINGS.allow_checkpoint_download,
        allowed_kinds=frozenset(API_SETTINGS.artifact_allowed_kinds),
    )
