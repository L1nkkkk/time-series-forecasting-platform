"""Experiment artifact recording."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from ts_platform.config.loader import save_config_snapshot
from ts_platform.config.schema import PlatformConfig


class ExperimentRecorder:
    """Create run directories and write reproducibility artifacts."""

    def __init__(self, root_dir: Path, experiment_name: str, overwrite: bool = False) -> None:
        self.root_dir = root_dir
        self.experiment_name = experiment_name
        self.overwrite = overwrite
        self.created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        self.run_id = self._make_run_id()
        self.run_dir = self._resolve_run_dir()

    def prepare(self) -> Path:
        """Create and return the run directory."""

        if self.overwrite and self.run_dir.exists():
            shutil.rmtree(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir

    def save_config(self, config: PlatformConfig) -> Path:
        """Save the validated config snapshot."""

        path = self.run_dir / "config_snapshot.yaml"
        save_config_snapshot(config, path)
        return path

    def save_environment(self, environment: dict[str, Any]) -> Path:
        """Save environment metadata."""

        return self.save_json("environment.json", environment)

    def save_results(self, results: dict[str, Any]) -> Path:
        """Save experiment results."""

        return self.save_json("results.json", results)

    def save_json(self, filename: str, payload: dict[str, Any]) -> Path:
        """Save JSON payload in the run directory."""

        path = self.run_dir / filename
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def metadata(self) -> dict[str, str]:
        """Return run metadata for result payloads."""

        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "run_dir": str(self.run_dir),
            "experiment_name": self.experiment_name,
        }

    def _resolve_run_dir(self) -> Path:
        base = self.root_dir / self.experiment_name
        run_dir = base / "latest" if self.overwrite else base / self.run_id
        root = self.root_dir.resolve()
        resolved_run_dir = run_dir.resolve()
        if not resolved_run_dir.is_relative_to(root):
            msg = f"experiment run_dir escapes root_dir: {run_dir}"
            raise ValueError(msg)
        return run_dir

    def _make_run_id(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{timestamp}_{uuid4().hex[:6]}"
