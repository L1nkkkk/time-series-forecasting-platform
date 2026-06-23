from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.helpers import tiny_config
from ts_platform.experiment.artifacts import (
    ArtifactEntry,
    ArtifactManifest,
    build_train_artifact_manifest,
    save_artifact_manifest,
)
from ts_platform.runner.trainer import Trainer


def test_train_run_writes_artifacts_json(tmp_path) -> None:
    result = Trainer(tiny_config(tmp_path, name="artifact_train")).run()

    assert (result.run_dir / "artifacts.json").exists()


def test_train_artifacts_json_contains_expected_entries(tmp_path) -> None:
    result = Trainer(tiny_config(tmp_path, name="artifact_train_entries")).run()
    payload = json.loads((result.run_dir / "artifacts.json").read_text(encoding="utf-8"))

    artifact_names = {artifact["name"] for artifact in payload["artifacts"]}

    assert payload["run_type"] == "train"
    assert payload["experiment_name"] == "artifact_train_entries"
    assert payload["run_id"] == result.run_id
    assert payload["run_dir"] == str(result.run_dir)
    assert {
        "results",
        "checkpoint",
        "model_export",
        "model_export_metadata",
        "config_snapshot",
        "environment",
        "forecast_samples",
        "progress",
        "train_log",
    }.issubset(artifact_names)


def test_train_run_writes_progress_json(tmp_path) -> None:
    result = Trainer(tiny_config(tmp_path, name="artifact_train_progress")).run()
    payload = json.loads((result.run_dir / "progress.json").read_text(encoding="utf-8"))

    assert payload["status"] == "succeeded"
    assert payload["run_id"] == result.run_id
    assert payload["experiment_name"] == "artifact_train_progress"
    assert payload["completed_epochs"] == result.history[-1]["epoch"]
    assert payload["total_epochs"] == result.history[-1]["epoch"]
    assert payload["progress_percent"] == 100
    assert payload["history"] == result.history
    assert payload["test_metrics"] == result.test_metrics


def test_artifact_manifest_paths_stay_inside_run_dir(tmp_path) -> None:
    run_dir = tmp_path / "run"
    outside = tmp_path / "outside.json"
    run_dir.mkdir()
    (run_dir / "results.json").write_text("{}", encoding="utf-8")
    (run_dir / "checkpoint.pt").write_text("checkpoint", encoding="utf-8")
    (run_dir / "model_export.pt").write_text("model", encoding="utf-8")
    (run_dir / "model_export.json").write_text("{}", encoding="utf-8")
    outside.write_text("{}", encoding="utf-8")

    generated = build_train_artifact_manifest(
        experiment_name="safe",
        run_id="20260619T120000Z_a1b2c3",
        run_dir=run_dir,
        checkpoint_path=run_dir / "checkpoint.pt",
        model_export_path=run_dir / "model_export.pt",
        model_export_metadata_path=run_dir / "model_export.json",
    )

    assert all(
        Path(artifact.path).resolve().is_relative_to(run_dir.resolve())
        for artifact in generated.artifacts
    )

    manifest = ArtifactManifest(
        run_type="train",
        experiment_name="escape",
        run_id="20260619T120000Z_a1b2c3",
        run_dir=run_dir,
        artifacts=[
            ArtifactEntry(
                name="outside",
                kind="json",
                path=outside,
                description="Outside file",
            )
        ],
    )

    with pytest.raises(ValueError, match="artifact path escapes run_dir"):
        save_artifact_manifest(manifest, run_dir / "artifacts.json")

    with pytest.raises(ValueError, match="artifact path escapes run_dir"):
        build_train_artifact_manifest(
            experiment_name="escape",
            run_id="20260619T120000Z_a1b2c3",
            run_dir=run_dir,
            checkpoint_path=outside,
            model_export_path=run_dir / "model_export.pt",
            model_export_metadata_path=run_dir / "model_export.json",
        )
