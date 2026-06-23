"""Safe artifact file lookup built on top of experiment manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from ts_platform.api.services.experiment_store import (
    CorruptExperimentArtifactError,
    ExperimentArtifactNotFoundError,
    ExperimentStore,
    UnsafePathComponentError,
)
from ts_platform.config.schema import validate_safe_path_component

DEFAULT_MAX_ARTIFACT_BYTES: Final = 5 * 1024 * 1024
DEFAULT_ALLOWED_ARTIFACT_KINDS: Final = frozenset({"json", "yaml", "csv", "log", "model"})
ARTIFACT_MEDIA_TYPES: Final = {
    "json": "application/json",
    "yaml": "text/yaml",
    "csv": "text/csv",
    "log": "text/plain",
    "model": "application/octet-stream",
    "checkpoint": "application/octet-stream",
}


class ArtifactAccessForbiddenError(PermissionError):
    """Raised when an artifact kind is not downloadable by policy."""


class ArtifactTooLargeError(ValueError):
    """Raised when an artifact exceeds the configured download limit."""


@dataclass(frozen=True)
class ArtifactAccessPolicy:
    """Download policy for files referenced by ``artifacts.json``."""

    max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES
    allow_checkpoint_download: bool = False
    allowed_kinds: frozenset[str] = DEFAULT_ALLOWED_ARTIFACT_KINDS


@dataclass(frozen=True)
class ArtifactFile:
    """A resolved artifact file that passed access checks."""

    name: str
    kind: str
    path: Path
    media_type: str
    size_bytes: int


class ArtifactService:
    """Resolve downloadable artifact files from manifest metadata."""

    def __init__(self, runs_root: Path, policy: ArtifactAccessPolicy | None = None) -> None:
        self.runs_root = Path(runs_root)
        self._resolved_root = self.runs_root.resolve()
        self._store = ExperimentStore(self.runs_root)
        self._policy = policy or ArtifactAccessPolicy()

    def read_artifact_manifest(self, experiment_name: str, run_id: str) -> dict[str, Any]:
        """Read a train or compare artifact manifest through ``ExperimentStore``."""

        return self._store.read_artifacts(experiment_name, run_id)

    def resolve_artifact(
        self,
        experiment_name: str,
        run_id: str,
        artifact_name: str,
    ) -> ArtifactFile:
        """Resolve one named artifact from a manifest and enforce access policy."""

        safe_artifact_name = self._validate_artifact_name(artifact_name)
        resolved_run = self._store.resolve_run(experiment_name, run_id)
        manifest = self.read_artifact_manifest(experiment_name, run_id)
        artifact = self._find_manifest_artifact(manifest, safe_artifact_name)
        kind = self._read_required_str(artifact, "kind")
        path_value = self._read_required_str(artifact, "path")

        self._assert_allowed_kind(kind)
        artifact_path = self._resolve_manifest_path(path_value)
        if not artifact_path.is_relative_to(resolved_run.run_dir):
            msg = f"artifact path escapes run directory: {path_value}"
            raise UnsafePathComponentError(msg)
        if not artifact_path.is_file():
            msg = f"artifact file does not exist: {artifact_path}"
            raise ExperimentArtifactNotFoundError(msg)
        size_bytes = self._artifact_size(artifact_path)
        if size_bytes > self._policy.max_bytes:
            msg = (
                f"artifact exceeds maximum size: {artifact_path} "
                f"({size_bytes} bytes > {self._policy.max_bytes} bytes)"
            )
            raise ArtifactTooLargeError(msg)

        return ArtifactFile(
            name=safe_artifact_name,
            kind=kind,
            path=artifact_path,
            media_type=ARTIFACT_MEDIA_TYPES.get(kind, "application/octet-stream"),
            size_bytes=size_bytes,
        )

    def _validate_artifact_name(self, artifact_name: str) -> str:
        try:
            return validate_safe_path_component(artifact_name, field_name="artifact_name")
        except ValueError as exc:
            raise UnsafePathComponentError(str(exc)) from exc

    def _find_manifest_artifact(
        self,
        manifest: dict[str, Any],
        artifact_name: str,
    ) -> dict[str, Any]:
        raw_artifacts = manifest.get("artifacts")
        if not isinstance(raw_artifacts, list):
            msg = "artifact manifest field is not a list: artifacts"
            raise CorruptExperimentArtifactError(msg)

        for raw_artifact in raw_artifacts:
            if not isinstance(raw_artifact, dict):
                msg = "artifact manifest entry is not a JSON object"
                raise CorruptExperimentArtifactError(msg)
            artifact = cast(dict[str, Any], raw_artifact)
            if artifact.get("name") == artifact_name:
                return artifact

        msg = f"artifact is not registered in manifest: {artifact_name}"
        raise ExperimentArtifactNotFoundError(msg)

    def _read_required_str(self, artifact: dict[str, Any], field_name: str) -> str:
        value = artifact.get(field_name)
        if not isinstance(value, str) or not value:
            msg = f"artifact manifest field is invalid: {field_name}"
            raise CorruptExperimentArtifactError(msg)
        return value

    def _assert_allowed_kind(self, kind: str) -> None:
        if kind == "checkpoint":
            if self._policy.allow_checkpoint_download:
                return
            msg = "checkpoint artifact download is disabled by policy"
            raise ArtifactAccessForbiddenError(msg)
        if kind not in self._policy.allowed_kinds:
            msg = f"artifact kind is not downloadable: {kind}"
            raise ArtifactAccessForbiddenError(msg)

    def _resolve_manifest_path(self, path_value: str) -> Path:
        raw_path = Path(path_value)
        candidates = [raw_path] if raw_path.is_absolute() else [raw_path, self.runs_root / raw_path]
        for candidate in candidates:
            resolved_path = candidate.resolve()
            if resolved_path.is_relative_to(self._resolved_root):
                return resolved_path

        msg = f"artifact path escapes runs root: {path_value}"
        raise UnsafePathComponentError(msg)

    def _artifact_size(self, path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError as exc:
            msg = f"artifact file cannot be read: {path}"
            raise ExperimentArtifactNotFoundError(msg) from exc
