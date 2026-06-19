"""Experiment API routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter

from ts_platform.config.schema import PlatformConfig
from ts_platform.runner.trainer import Trainer

router = APIRouter()


@router.get("/experiments")
def list_experiments(root: str = "runs") -> dict[str, list[str]]:
    """List local experiment run directories."""

    root_path = Path(root)
    if not root_path.exists():
        return {"experiments": []}
    experiments = [path.name for path in root_path.iterdir() if path.is_dir()]
    return {"experiments": sorted(experiments)}


@router.post("/experiments/train")
def train_experiment(config: PlatformConfig) -> dict[str, Any]:
    """Run a synchronous training demo from a config payload."""

    result = Trainer(config).run()
    return result.to_dict()
