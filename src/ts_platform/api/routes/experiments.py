"""Experiment API routes."""

from __future__ import annotations

from typing import Any, NoReturn

from fastapi import APIRouter, HTTPException

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
RUNS_ROOT = APISettings().runs_root


@router.get("/experiments")
def list_experiments() -> dict[str, list[dict[str, Any]]]:
    """List local experiment metadata from the fixed runs root."""

    return {"experiments": ExperimentStore(RUNS_ROOT).list_experiments()}


@router.post("/experiments/train")
def train_experiment(config: PlatformConfig) -> dict[str, Any]:
    """Run a synchronous training demo from a config payload."""

    return train_with_safe_output_dir(config, runs_root=RUNS_ROOT)


@router.post("/experiments/compare")
def compare_experiments(config: CompareConfig) -> dict[str, Any]:
    """Run a synchronous compare demo from a compare config payload."""

    return compare_with_safe_output_dir(config, runs_root=RUNS_ROOT)


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
    | CorruptExperimentArtifactError,
) -> NoReturn:
    if isinstance(exc, UnsafePathComponentError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, ExperimentArtifactNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail=str(exc)) from exc
