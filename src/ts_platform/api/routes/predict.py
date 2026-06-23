"""Prediction routes for model export artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ts_platform.api.services.artifact_service import ArtifactService
from ts_platform.api.settings import APISettings
from ts_platform.runner.predictor import predict_from_model_export

router = APIRouter()
API_SETTINGS = APISettings()
RUNS_ROOT = API_SETTINGS.runs_root


class PredictionRequest(BaseModel):
    """Input windows for inference."""

    values: list[Any] = Field(min_length=1)


class ModelExportPredictionRequest(PredictionRequest):
    """Trusted local model export prediction request."""

    model_export_path: str = Field(min_length=1)


@router.post("/predict/model-export")
def predict_model_export(payload: ModelExportPredictionRequest) -> dict[str, Any]:
    """Run prediction from a trusted local ``model_export.pt`` path."""

    try:
        return predict_from_model_export(
            Path(payload.model_export_path),
            values=payload.values,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=f"prediction failed: {exc}") from exc


@router.post("/experiments/{experiment_name}/{run_id}/predict")
def predict_experiment_run(
    experiment_name: str,
    run_id: str,
    payload: PredictionRequest,
) -> dict[str, Any]:
    """Run prediction from the ``model_export`` artifact of a completed run."""

    try:
        artifact = ArtifactService(RUNS_ROOT).resolve_artifact(
            experiment_name,
            run_id,
            "model_export",
        )
        if artifact.kind != "model":
            raise ValueError("model_export artifact must have kind 'model'")
        return predict_from_model_export(artifact.path, values=payload.values)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=f"prediction failed: {exc}") from exc
