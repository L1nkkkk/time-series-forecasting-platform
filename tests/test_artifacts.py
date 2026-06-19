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
        "config_snapshot",
        "environment",
        "train_log",
    }.issubset(artifact_names)


def test_artifact_manifest_paths_stay_inside_run_dir(tmp_path) -> None:
    run_dir = tmp_path / "run"
    outside = tmp_path / "outside.json"
    run_dir.mkdir()
    (run_dir / "results.json").write_text("{}", encoding="utf-8")
    (run_dir / "checkpoint.pt").write_text("checkpoint", encoding="utf-8")
    outside.write_text("{}", encoding="utf-8")

    generated = build_train_artifact_manifest(
        experiment_name="safe",
        run_id="20260619T120000Z_a1b2c3",
        run_dir=run_dir,
        checkpoint_path=run_dir / "checkpoint.pt",
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
        )
