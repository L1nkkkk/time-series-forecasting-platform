"""Whitelisted local demo routes for the dashboard UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from fastapi import APIRouter, HTTPException

from ts_platform.api.routes import jobs as job_routes
from ts_platform.api.routes.errors import raise_execution_http_error
from ts_platform.api.services.compare_service import compare_with_safe_output_dir
from ts_platform.api.services.training_service import train_with_safe_output_dir
from ts_platform.api.settings import APISettings
from ts_platform.config.compare_loader import load_compare_config
from ts_platform.config.loader import load_config
from ts_platform.data.assets import prepare_dataset_asset
from ts_platform.data.catalog import DATASET_CATALOG
from ts_platform.data.catalog_loader import load_dataset_catalog
from ts_platform.data.catalog_tools import find_catalog_metadata

router = APIRouter(prefix="/demo")
API_SETTINGS = APISettings()
RUNS_ROOT = API_SETTINGS.runs_root

TRAIN_DEMO_NAMES: Final[tuple[str, ...]] = (
    "simple_forecast",
    "csv_forecast",
    "csv_feature_forecast",
    "appliances_energy_half_hour_demo",
    "ideal_training_30min_demo",
)
SYNC_TRAIN_DEMO_NAMES: Final[tuple[str, ...]] = (
    "simple_forecast",
    "csv_forecast",
    "csv_feature_forecast",
    "appliances_energy_half_hour_demo",
)
COMPARE_DEMO_NAMES: Final[tuple[str, ...]] = (
    "compare_forecast",
    "compare_model_zoo",
    "compare_feature_forecast",
    "ideal_target_demo",
)
ALL_DEMO_NAMES: Final[tuple[str, ...]] = (*TRAIN_DEMO_NAMES, *COMPARE_DEMO_NAMES)
IDEAL_TARGET_DEMO_NAME: Final = "ideal_target_demo"
IDEAL_TRAINING_DEMO_NAME: Final = "ideal_training_30min_demo"
IDEAL_TARGET_DATASET_NAME: Final = "etth1"


@router.get("/configs")
def list_demo_configs() -> dict[str, list[str]]:
    """Return the demo config names that may be launched by the dashboard."""

    return {
        "configs": list(ALL_DEMO_NAMES),
        "train": list(TRAIN_DEMO_NAMES),
        "compare": list(COMPARE_DEMO_NAMES),
    }


@router.post("/train/{demo_name}")
def train_demo(demo_name: str) -> dict[str, Any]:
    """Run one whitelisted training demo config."""

    config_path = _demo_config_path(demo_name, allowed=SYNC_TRAIN_DEMO_NAMES)
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    try:
        return train_with_safe_output_dir(config, runs_root=RUNS_ROOT)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise_execution_http_error("training", exc)


@router.post("/compare/{demo_name}")
def compare_demo(demo_name: str) -> dict[str, Any]:
    """Run one whitelisted compare demo config."""

    config_path = _demo_config_path(demo_name, allowed=COMPARE_DEMO_NAMES)
    _prepare_required_demo_assets(demo_name)
    try:
        config = load_compare_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    try:
        return compare_with_safe_output_dir(config, runs_root=RUNS_ROOT)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise_execution_http_error("compare", exc)


@router.post("/jobs/train/{demo_name}")
def submit_train_demo_job(demo_name: str) -> dict[str, Any]:
    """Submit one whitelisted training demo config as a local job."""

    config_path = _demo_config_path(demo_name, allowed=TRAIN_DEMO_NAMES)
    _prepare_required_demo_assets(demo_name)
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return job_routes.submit_train_job(config)


@router.post("/jobs/compare/{demo_name}")
def submit_compare_demo_job(demo_name: str) -> dict[str, Any]:
    """Submit one whitelisted compare demo config as a local job."""

    config_path = _demo_config_path(demo_name, allowed=COMPARE_DEMO_NAMES)
    _prepare_required_demo_assets(demo_name)
    try:
        config = load_compare_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return job_routes.submit_compare_job(config)


def _demo_config_path(demo_name: str, *, allowed: tuple[str, ...]) -> Path:
    if demo_name not in allowed:
        raise HTTPException(status_code=404, detail=f"unknown demo config: {demo_name}")
    return EXAMPLE_CONFIG_ROOT / f"{demo_name}.yaml"


def _prepare_required_demo_assets(demo_name: str) -> None:
    if demo_name not in {IDEAL_TARGET_DEMO_NAME, IDEAL_TRAINING_DEMO_NAME}:
        return
    try:
        try:
            metadata = DATASET_CATALOG.get(IDEAL_TARGET_DATASET_NAME)
        except KeyError:
            public_catalog_path = (
                EXAMPLE_CONFIG_ROOT.parent / "datasets" / "public_time_series.yaml"
            )
            metadata = find_catalog_metadata(
                load_dataset_catalog(public_catalog_path),
                IDEAL_TARGET_DATASET_NAME,
            )
        prepare_dataset_asset(metadata)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _resolve_example_config_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "configs" / "examples"
        if candidate.is_dir():
            return candidate
    return Path("configs/examples")


EXAMPLE_CONFIG_ROOT: Final = _resolve_example_config_root()
