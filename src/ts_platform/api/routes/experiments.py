"""Experiment API routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from ts_platform.api.services.training_service import train_with_safe_output_dir
from ts_platform.api.settings import APISettings
from ts_platform.config.schema import PlatformConfig

router = APIRouter()
RUNS_ROOT = APISettings().runs_root


@router.get("/experiments")
def list_experiments() -> dict[str, list[dict[str, Any]]]:
    """List local experiment metadata from the fixed runs root."""

    if not RUNS_ROOT.exists():
        return {"experiments": []}
    return {"experiments": _discover_experiments(RUNS_ROOT)}


@router.post("/experiments/train")
def train_experiment(config: PlatformConfig) -> dict[str, Any]:
    """Run a synchronous training demo from a config payload."""

    return train_with_safe_output_dir(config, runs_root=RUNS_ROOT)


def _discover_experiments(root: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in root.glob("*/*") if path.is_dir()):
        results_path = run_dir / "results.json"
        if not results_path.exists():
            summaries.append(
                {
                    "status": "incomplete",
                    "run_dir": str(run_dir),
                    "experiment_name": run_dir.parent.name,
                }
            )
            continue
        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summaries.append(
                {
                    "status": "incomplete",
                    "run_dir": str(run_dir),
                    "experiment_name": run_dir.parent.name,
                }
            )
            continue
        summaries.append(
            {
                "status": "complete",
                "experiment_name": payload.get("experiment_name"),
                "run_id": payload.get("run_id"),
                "created_at": payload.get("created_at"),
                "run_dir": payload.get("run_dir", str(run_dir)),
                "checkpoint_path": payload.get("checkpoint_path"),
                "test_metrics": payload.get("test_metrics"),
            }
        )
    return summaries
