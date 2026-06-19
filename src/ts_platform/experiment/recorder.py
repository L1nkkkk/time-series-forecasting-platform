"""Experiment artifact recording."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ts_platform.config.loader import save_config_snapshot
from ts_platform.config.schema import PlatformConfig


class ExperimentRecorder:
    """Create run directories and write reproducibility artifacts."""

    def __init__(self, root_dir: Path, experiment_name: str, overwrite: bool = False) -> None:
        self.root_dir = root_dir
        self.experiment_name = experiment_name
        self.overwrite = overwrite
        self.run_dir = self._resolve_run_dir()

    def prepare(self) -> Path:
        """Create and return the run directory."""

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

    def _resolve_run_dir(self) -> Path:
        base = self.root_dir / self.experiment_name
        if self.overwrite or not base.exists():
            return base
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return self.root_dir / f"{self.experiment_name}_{timestamp}"
