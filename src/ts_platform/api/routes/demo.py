"""Whitelisted local demo routes for the dashboard UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from fastapi import APIRouter, HTTPException

from ts_platform.api.services.compare_service import compare_with_safe_output_dir
from ts_platform.api.services.training_service import train_with_safe_output_dir
from ts_platform.api.settings import APISettings
from ts_platform.config.compare_loader import load_compare_config
from ts_platform.config.loader import load_config

router = APIRouter(prefix="/demo")
API_SETTINGS = APISettings()
RUNS_ROOT = API_SETTINGS.runs_root

TRAIN_DEMO_NAMES: Final[tuple[str, ...]] = (
    "simple_forecast",
    "csv_forecast",
    "csv_feature_forecast",
)
COMPARE_DEMO_NAMES: Final[tuple[str, ...]] = (
    "compare_forecast",
    "compare_model_zoo",
    "compare_feature_forecast",
)
ALL_DEMO_NAMES: Final[tuple[str, ...]] = (*TRAIN_DEMO_NAMES, *COMPARE_DEMO_NAMES)


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

    config_path = _demo_config_path(demo_name, allowed=TRAIN_DEMO_NAMES)
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return train_with_safe_output_dir(config, runs_root=RUNS_ROOT)


@router.post("/compare/{demo_name}")
def compare_demo(demo_name: str) -> dict[str, Any]:
    """Run one whitelisted compare demo config."""

    config_path = _demo_config_path(demo_name, allowed=COMPARE_DEMO_NAMES)
    try:
        config = load_compare_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return compare_with_safe_output_dir(config, runs_root=RUNS_ROOT)


def _demo_config_path(demo_name: str, *, allowed: tuple[str, ...]) -> Path:
    if demo_name not in allowed:
        raise HTTPException(status_code=404, detail=f"unknown demo config: {demo_name}")
    return EXAMPLE_CONFIG_ROOT / f"{demo_name}.yaml"


def _resolve_example_config_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "configs" / "examples"
        if candidate.is_dir():
            return candidate
    return Path("configs/examples")


EXAMPLE_CONFIG_ROOT: Final = _resolve_example_config_root()
