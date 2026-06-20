"""Safe experiment artifact lookup for API and CLI callers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ts_platform.config.schema import validate_safe_path_component


class UnsafePathComponentError(ValueError):
    """Raised when a user-controlled path component is unsafe."""


class ExperimentArtifactNotFoundError(FileNotFoundError):
    """Raised when a requested experiment artifact does not exist."""


class CorruptExperimentArtifactError(ValueError):
    """Raised when an experiment artifact cannot be decoded."""


@dataclass(frozen=True)
class ResolvedRun:
    """A safely resolved physical run directory and its standard artifact paths."""

    experiment_name: str
    run_id: str
    run_dir: Path
    results_path: Path
    artifacts_path: Path


class ExperimentStore:
    """Read experiment artifacts from one fixed runs root."""

    def __init__(self, runs_root: Path) -> None:
        self.runs_root = Path(runs_root)
        self._resolved_root = self.runs_root.resolve()

    def resolve_run(self, experiment_name: str, run_id: str) -> ResolvedRun:
        """Resolve a requested run to a safe physical directory under ``runs_root``."""

        safe_experiment_name = self._validate_component(
            experiment_name,
            field_name="experiment_name",
        )
        safe_run_id = self._validate_component(run_id, field_name="run_id")
        run_dir = self._resolve_run_dir_from_safe_components(
            safe_experiment_name,
            safe_run_id,
        )
        resolved_run_dir = run_dir.resolve()
        results_path = (run_dir / "results.json").resolve()
        artifacts_path = (run_dir / "artifacts.json").resolve()
        self._assert_inside_root(resolved_run_dir)
        self._assert_inside_root(results_path)
        self._assert_inside_root(artifacts_path)
        return ResolvedRun(
            experiment_name=safe_experiment_name,
            run_id=safe_run_id,
            run_dir=resolved_run_dir,
            results_path=results_path,
            artifacts_path=artifacts_path,
        )

    def list_experiments(self) -> list[dict[str, Any]]:
        """List train, compare, and incomplete runs under the fixed runs root."""

        if not self.runs_root.exists():
            return []

        summaries: list[dict[str, Any]] = []
        for run_dir in sorted(path for path in self.runs_root.glob("*/*") if path.is_dir()):
            if run_dir.parent.name == "jobs":
                continue
            self._assert_inside_root(run_dir)
            results_path = run_dir / "results.json"
            try:
                payload = self._read_json_object(results_path)
            except (ExperimentArtifactNotFoundError, CorruptExperimentArtifactError):
                summaries.append(self._incomplete_summary(run_dir))
                continue

            if payload.get("run_type") == "compare":
                summaries.append(self._compare_summary(run_dir, payload))
            else:
                summaries.append(self._train_summary(run_dir, payload))
        return summaries

    def read_results(self, experiment_name: str, run_id: str) -> dict[str, Any]:
        """Read a train or compare ``results.json`` payload."""

        resolved_run = self.resolve_run(experiment_name, run_id)
        return self._read_json_object(resolved_run.results_path)

    def read_artifacts(self, experiment_name: str, run_id: str) -> dict[str, Any]:
        """Read a train or compare ``artifacts.json`` payload."""

        resolved_run = self.resolve_run(experiment_name, run_id)
        return self._read_json_object(resolved_run.artifacts_path)

    def read_leaderboard(self, experiment_name: str, run_id: str) -> list[dict[str, Any]]:
        """Read a compare run ``leaderboard.json`` payload."""

        resolved_run = self.resolve_run(experiment_name, run_id)
        leaderboard_path = resolved_run.run_dir / "leaderboard.json"
        payload = self._read_json(leaderboard_path)
        if not isinstance(payload, list):
            msg = f"leaderboard artifact is not a JSON array: {leaderboard_path}"
            raise CorruptExperimentArtifactError(msg)
        rows: list[dict[str, Any]] = []
        for row in payload:
            if not isinstance(row, dict):
                msg = f"leaderboard row is not a JSON object: {leaderboard_path}"
                raise CorruptExperimentArtifactError(msg)
            rows.append(cast(dict[str, Any], row))
        return rows

    def _resolve_run_dir_from_safe_components(
        self,
        safe_experiment_name: str,
        safe_run_id: str,
    ) -> Path:
        experiment_dir = self.runs_root / safe_experiment_name
        self._assert_inside_root(experiment_dir)

        direct_run_dir = experiment_dir / safe_run_id
        self._assert_inside_root(direct_run_dir)
        if direct_run_dir.exists():
            return direct_run_dir

        matched_run_dir = self._find_run_dir_by_payload_id(experiment_dir, safe_run_id)
        if matched_run_dir is not None:
            return matched_run_dir
        return direct_run_dir

    def _find_run_dir_by_payload_id(self, experiment_dir: Path, run_id: str) -> Path | None:
        if not experiment_dir.exists():
            return None

        for run_dir in sorted(path for path in experiment_dir.iterdir() if path.is_dir()):
            self._assert_inside_root(run_dir)
            results_path = run_dir / "results.json"
            if not results_path.exists():
                continue
            try:
                payload = self._read_json_object(results_path)
            except CorruptExperimentArtifactError:
                continue
            if payload.get("run_id") == run_id or payload.get("compare_run_id") == run_id:
                return run_dir
        return None

    def _read_json_object(self, path: Path) -> dict[str, Any]:
        payload = self._read_json(path)
        if not isinstance(payload, dict):
            msg = f"experiment artifact is not a JSON object: {path}"
            raise CorruptExperimentArtifactError(msg)
        return cast(dict[str, Any], payload)

    def _read_json(self, path: Path) -> Any:
        self._assert_inside_root(path)
        if not path.exists():
            msg = f"experiment artifact does not exist: {path}"
            raise ExperimentArtifactNotFoundError(msg)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            msg = f"experiment artifact is not valid JSON: {path}"
            raise CorruptExperimentArtifactError(msg) from exc
        except OSError as exc:
            msg = f"experiment artifact cannot be read: {path}"
            raise ExperimentArtifactNotFoundError(msg) from exc

    def _assert_inside_root(self, path: Path) -> None:
        resolved_path = path.resolve()
        if not resolved_path.is_relative_to(self._resolved_root):
            msg = f"experiment path escapes runs root: {path}"
            raise UnsafePathComponentError(msg)

    def _validate_component(self, value: str, *, field_name: str) -> str:
        try:
            return validate_safe_path_component(value, field_name=field_name)
        except ValueError as exc:
            raise UnsafePathComponentError(str(exc)) from exc

    def _incomplete_summary(self, run_dir: Path) -> dict[str, Any]:
        return {
            "status": "incomplete",
            "run_type": "unknown",
            "experiment_name": run_dir.parent.name,
            "run_dir": str(run_dir),
        }

    def _train_summary(self, run_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "complete",
            "run_type": "train",
            "experiment_name": payload.get("experiment_name", run_dir.parent.name),
            "run_id": payload.get("run_id", run_dir.name),
            "created_at": payload.get("created_at"),
            "run_dir": payload.get("run_dir", str(run_dir)),
            "checkpoint_path": payload.get("checkpoint_path"),
            "test_metrics": payload.get("test_metrics"),
        }

    def _compare_summary(self, run_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
        compare_run_id = payload.get("compare_run_id", payload.get("run_id", run_dir.name))
        return {
            "status": "complete",
            "run_type": "compare",
            "experiment_name": payload.get("experiment_name", run_dir.parent.name),
            "run_id": compare_run_id,
            "compare_run_id": compare_run_id,
            "created_at": payload.get("created_at"),
            "run_dir": payload.get("compare_run_dir", payload.get("run_dir", str(run_dir))),
            "compare_run_dir": payload.get("compare_run_dir", str(run_dir)),
            "primary_metric": payload.get("primary_metric"),
            "success_count": payload.get("success_count"),
            "failed_count": payload.get("failed_count"),
            "leaderboard_json_path": payload.get("leaderboard_json_path"),
            "leaderboard_csv_path": payload.get("leaderboard_csv_path"),
        }
