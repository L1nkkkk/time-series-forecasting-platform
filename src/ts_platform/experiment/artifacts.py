"""Artifact manifest helpers for train and compare runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ArtifactKind = Literal["json", "yaml", "csv", "checkpoint", "log", "model"]
RunType = Literal["train", "compare"]


@dataclass(frozen=True)
class ArtifactEntry:
    """One file produced by a run."""

    name: str
    kind: ArtifactKind
    path: Path
    description: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable artifact entry."""

        return {
            "name": self.name,
            "kind": self.kind,
            "path": str(self.path),
            "description": self.description,
        }


@dataclass(frozen=True)
class ArtifactManifest:
    """Serializable artifact manifest for a train or compare run."""

    run_type: RunType
    experiment_name: str
    run_id: str
    run_dir: Path
    artifacts: list[ArtifactEntry]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest."""

        payload: dict[str, Any] = {
            "run_type": self.run_type,
            "experiment_name": self.experiment_name,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }
        if self.run_type == "compare":
            payload["compare_run_id"] = self.run_id
            payload["compare_run_dir"] = str(self.run_dir)
        else:
            payload["run_id"] = self.run_id
            payload["run_dir"] = str(self.run_dir)
        return payload


def build_train_artifact_manifest(
    *,
    experiment_name: str,
    run_id: str,
    run_dir: Path,
    checkpoint_path: Path,
    model_export_path: Path,
    model_export_metadata_path: Path,
) -> ArtifactManifest:
    """Build a manifest for a completed train run."""

    entries = [
        ArtifactEntry(
            name="results",
            kind="json",
            path=run_dir / "results.json",
            description="Training result payload",
        ),
        ArtifactEntry(
            name="checkpoint",
            kind="checkpoint",
            path=checkpoint_path,
            description="Final model checkpoint",
        ),
        ArtifactEntry(
            name="model_export",
            kind="model",
            path=model_export_path,
            description="Inference model export without optimizer state",
        ),
        ArtifactEntry(
            name="model_export_metadata",
            kind="json",
            path=model_export_metadata_path,
            description="Inference model export metadata",
        ),
        ArtifactEntry(
            name="config_snapshot",
            kind="yaml",
            path=run_dir / "config_snapshot.yaml",
            description="Validated config snapshot",
        ),
        ArtifactEntry(
            name="environment",
            kind="json",
            path=run_dir / "environment.json",
            description="Runtime environment metadata",
        ),
        ArtifactEntry(
            name="forecast_samples",
            kind="json",
            path=run_dir / "forecast_samples.json",
            description="Original-scale forecast samples for visual inspection",
        ),
        ArtifactEntry(
            name="progress",
            kind="json",
            path=run_dir / "progress.json",
            description="Per-epoch training progress",
        ),
        ArtifactEntry(
            name="train_log",
            kind="log",
            path=run_dir / "train.log",
            description="Training log",
        ),
    ]
    return ArtifactManifest(
        run_type="train",
        experiment_name=experiment_name,
        run_id=run_id,
        run_dir=run_dir,
        artifacts=_existing_entries_inside_run_dir(entries, run_dir),
    )


def build_compare_artifact_manifest(
    *,
    experiment_name: str,
    compare_run_id: str,
    compare_run_dir: Path,
    leaderboard_json_path: Path,
    leaderboard_csv_path: Path,
) -> ArtifactManifest:
    """Build a manifest for a completed compare run."""

    entries = [
        ArtifactEntry(
            name="results",
            kind="json",
            path=compare_run_dir / "results.json",
            description="Compare result payload",
        ),
        ArtifactEntry(
            name="leaderboard_json",
            kind="json",
            path=leaderboard_json_path,
            description="Leaderboard rows as JSON",
        ),
        ArtifactEntry(
            name="leaderboard_csv",
            kind="csv",
            path=leaderboard_csv_path,
            description="Leaderboard rows as CSV",
        ),
        ArtifactEntry(
            name="compare_config_snapshot",
            kind="yaml",
            path=compare_run_dir / "compare_config_snapshot.yaml",
            description="Validated compare config snapshot",
        ),
        ArtifactEntry(
            name="environment",
            kind="json",
            path=compare_run_dir / "environment.json",
            description="Runtime environment metadata",
        ),
    ]
    return ArtifactManifest(
        run_type="compare",
        experiment_name=experiment_name,
        run_id=compare_run_id,
        run_dir=compare_run_dir,
        artifacts=_existing_entries_inside_run_dir(entries, compare_run_dir),
    )


def save_artifact_manifest(manifest: ArtifactManifest, path: Path) -> Path:
    """Write an artifact manifest JSON file."""

    _assert_entries_inside_run_dir(manifest.artifacts, manifest.run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _existing_entries_inside_run_dir(
    entries: list[ArtifactEntry],
    run_dir: Path,
) -> list[ArtifactEntry]:
    _assert_entries_inside_run_dir(entries, run_dir)
    return [entry for entry in entries if entry.path.exists()]


def _assert_entries_inside_run_dir(entries: list[ArtifactEntry], run_dir: Path) -> None:
    resolved_run_dir = run_dir.resolve()
    for entry in entries:
        resolved_path = entry.path.resolve()
        if not resolved_path.is_relative_to(resolved_run_dir):
            msg = f"artifact path escapes run_dir: {entry.path}"
            raise ValueError(msg)
