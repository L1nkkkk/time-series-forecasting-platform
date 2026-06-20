from __future__ import annotations

import json
from pathlib import Path

import pytest

from ts_platform.api.services.artifact_service import (
    ArtifactAccessForbiddenError,
    ArtifactAccessPolicy,
    ArtifactService,
    ArtifactTooLargeError,
)
from ts_platform.api.services.experiment_store import (
    CorruptExperimentArtifactError,
    ExperimentArtifactNotFoundError,
    UnsafePathComponentError,
)


def _write_manifest(
    runs_root: Path,
    *,
    experiment_name: str = "service_artifacts",
    artifacts: list[dict[str, str]],
) -> Path:
    run_dir = runs_root / experiment_name / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts.json").write_text(
        json.dumps(
            {
                "run_type": "train",
                "experiment_name": experiment_name,
                "run_id": "latest",
                "run_dir": str(run_dir),
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_artifact_service_resolves_manifest_artifact(tmp_path) -> None:
    run_dir = tmp_path / "service_artifacts" / "latest"
    run_dir.mkdir(parents=True)
    artifact_path = run_dir / "results.json"
    artifact_path.write_text('{"ok": true}', encoding="utf-8")
    _write_manifest(
        tmp_path,
        artifacts=[
            {
                "name": "results",
                "kind": "json",
                "path": str(artifact_path),
                "description": "Result payload",
            }
        ],
    )

    artifact = ArtifactService(tmp_path).resolve_artifact(
        "service_artifacts",
        "latest",
        "results",
    )

    assert artifact.name == "results"
    assert artifact.kind == "json"
    assert artifact.path == artifact_path.resolve()
    assert artifact.size_bytes == artifact_path.stat().st_size


def test_artifact_service_rejects_unknown_artifact_name(tmp_path) -> None:
    _write_manifest(tmp_path, artifacts=[])

    with pytest.raises(ExperimentArtifactNotFoundError, match="not registered"):
        ArtifactService(tmp_path).resolve_artifact("service_artifacts", "latest", "missing")


def test_artifact_service_rejects_unsafe_artifact_name(tmp_path) -> None:
    _write_manifest(tmp_path, artifacts=[])

    with pytest.raises(UnsafePathComponentError, match="artifact_name"):
        ArtifactService(tmp_path).resolve_artifact("service_artifacts", "latest", "../secret")


def test_artifact_service_rejects_path_escape(tmp_path) -> None:
    outside_path = tmp_path.parent / "outside_artifact.json"
    outside_path.write_text("{}", encoding="utf-8")
    _write_manifest(
        tmp_path,
        artifacts=[
            {
                "name": "outside",
                "kind": "json",
                "path": str(outside_path),
                "description": "Outside file",
            }
        ],
    )

    with pytest.raises(UnsafePathComponentError, match="escapes runs root"):
        ArtifactService(tmp_path).resolve_artifact("service_artifacts", "latest", "outside")


def test_artifact_service_rejects_cross_run_artifact_path(tmp_path) -> None:
    other_run_dir = tmp_path / "other_artifacts" / "latest"
    other_run_dir.mkdir(parents=True)
    other_artifact_path = other_run_dir / "results.json"
    other_artifact_path.write_text('{"other": true}', encoding="utf-8")
    _write_manifest(
        tmp_path,
        artifacts=[
            {
                "name": "results",
                "kind": "json",
                "path": str(other_artifact_path),
                "description": "Cross-run result payload",
            }
        ],
    )

    with pytest.raises(UnsafePathComponentError, match="escapes run directory"):
        ArtifactService(tmp_path).resolve_artifact("service_artifacts", "latest", "results")


def test_artifact_service_rejects_tampered_manifest_run_dir(tmp_path) -> None:
    run_dir = tmp_path / "service_artifacts" / "latest"
    other_run_dir = tmp_path / "other_artifacts" / "latest"
    run_dir.mkdir(parents=True)
    other_run_dir.mkdir(parents=True)
    other_artifact_path = other_run_dir / "secret.json"
    other_artifact_path.write_text('{"secret": true}', encoding="utf-8")
    (run_dir / "artifacts.json").write_text(
        json.dumps(
            {
                "run_type": "train",
                "experiment_name": "service_artifacts",
                "run_id": "latest",
                "run_dir": str(other_run_dir),
                "artifacts": [
                    {
                        "name": "secret",
                        "kind": "json",
                        "path": str(other_artifact_path),
                        "description": "Tampered run_dir artifact",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(UnsafePathComponentError, match="escapes run directory"):
        ArtifactService(tmp_path).resolve_artifact("service_artifacts", "latest", "secret")


def test_artifact_service_uses_store_resolved_run_dir_for_boundary(tmp_path) -> None:
    run_dir = tmp_path / "service_artifacts" / "latest"
    other_run_dir = tmp_path / "other_artifacts" / "latest"
    run_dir.mkdir(parents=True)
    other_run_dir.mkdir(parents=True)
    artifact_path = run_dir / "results.json"
    artifact_path.write_text('{"ok": true}', encoding="utf-8")
    (run_dir / "artifacts.json").write_text(
        json.dumps(
            {
                "run_type": "train",
                "experiment_name": "service_artifacts",
                "run_id": "latest",
                "run_dir": str(other_run_dir),
                "artifacts": [
                    {
                        "name": "results",
                        "kind": "json",
                        "path": str(artifact_path),
                        "description": "Result payload",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    artifact = ArtifactService(tmp_path).resolve_artifact(
        "service_artifacts",
        "latest",
        "results",
    )

    assert artifact.path == artifact_path.resolve()


def test_artifact_service_allows_artifact_inside_current_run_dir(tmp_path) -> None:
    run_dir = _write_manifest(
        tmp_path,
        artifacts=[
            {
                "name": "results",
                "kind": "json",
                "path": str(tmp_path / "service_artifacts" / "latest" / "results.json"),
                "description": "Result payload",
            }
        ],
    )
    artifact_path = run_dir / "results.json"
    artifact_path.write_text('{"ok": true}', encoding="utf-8")

    artifact = ArtifactService(tmp_path).resolve_artifact(
        "service_artifacts",
        "latest",
        "results",
    )

    assert artifact.path == artifact_path.resolve()


def test_artifact_service_rejects_checkpoint_by_default(tmp_path) -> None:
    run_dir = tmp_path / "service_artifacts" / "latest"
    run_dir.mkdir(parents=True)
    checkpoint_path = run_dir / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    _write_manifest(
        tmp_path,
        artifacts=[
            {
                "name": "checkpoint",
                "kind": "checkpoint",
                "path": str(checkpoint_path),
                "description": "Model checkpoint",
            }
        ],
    )

    with pytest.raises(ArtifactAccessForbiddenError, match="checkpoint"):
        ArtifactService(tmp_path).resolve_artifact("service_artifacts", "latest", "checkpoint")


def test_artifact_service_rejects_large_file(tmp_path) -> None:
    run_dir = tmp_path / "service_artifacts" / "latest"
    run_dir.mkdir(parents=True)
    artifact_path = run_dir / "big.csv"
    artifact_path.write_text("abcd", encoding="utf-8")
    _write_manifest(
        tmp_path,
        artifacts=[
            {
                "name": "big_csv",
                "kind": "csv",
                "path": str(artifact_path),
                "description": "Large CSV",
            }
        ],
    )

    with pytest.raises(ArtifactTooLargeError, match="exceeds maximum size"):
        ArtifactService(
            tmp_path,
            policy=ArtifactAccessPolicy(max_bytes=3),
        ).resolve_artifact("service_artifacts", "latest", "big_csv")


def test_artifact_service_rejects_corrupt_manifest(tmp_path) -> None:
    run_dir = tmp_path / "service_artifacts" / "latest"
    run_dir.mkdir(parents=True)
    (run_dir / "artifacts.json").write_text(
        json.dumps({"artifacts": {"name": "results"}}),
        encoding="utf-8",
    )

    with pytest.raises(CorruptExperimentArtifactError, match="artifacts"):
        ArtifactService(tmp_path).resolve_artifact("service_artifacts", "latest", "results")


@pytest.mark.parametrize(
    ("kind", "expected_media_type"),
    [
        ("json", "application/json"),
        ("yaml", "text/yaml"),
        ("csv", "text/csv"),
        ("log", "text/plain"),
    ],
)
def test_artifact_service_sets_media_type(
    tmp_path,
    kind: str,
    expected_media_type: str,
) -> None:
    run_dir = tmp_path / "service_artifacts" / "latest"
    run_dir.mkdir(parents=True)
    artifact_path = run_dir / f"artifact.{kind}"
    artifact_path.write_text("content", encoding="utf-8")
    _write_manifest(
        tmp_path,
        artifacts=[
            {
                "name": f"{kind}_artifact",
                "kind": kind,
                "path": str(artifact_path),
                "description": "Typed artifact",
            }
        ],
    )

    artifact = ArtifactService(tmp_path).resolve_artifact(
        "service_artifacts",
        "latest",
        f"{kind}_artifact",
    )

    assert artifact.media_type == expected_media_type
