"""Downloadable dataset asset preparation and cache metadata."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ts_platform.config.schema import validate_safe_path_component
from ts_platform.data.catalog import DatasetMetadata
from ts_platform.data.catalog_tools import config_from_catalog_metadata, save_generated_config
from ts_platform.data.profile import profile_csv_dataset

DEFAULT_CACHE_ROOT = Path("data/cache/datasets")
DEFAULT_EXTERNAL_ROOT = Path("data/external")
DEFAULT_INPUT_LEN = 24
DEFAULT_OUTPUT_LEN = 12
DEFAULT_MODEL_NAME = "linear"
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 16
MANIFEST_FILENAME = "manifest.json"
PREPARED_CATALOG_FILENAME = "prepared_catalog.yaml"
SUPPORTED_ARCHIVE_FORMATS = {"raw_csv", "csv", "raw_txt", "raw_matrix", "zip_csv"}

Downloader = Callable[[str, Path], None]


def prepare_dataset_asset(
    metadata: DatasetMetadata,
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    external_root: Path = DEFAULT_EXTERNAL_ROOT,
    force: bool = False,
    downloader: Downloader | None = None,
) -> dict[str, Any]:
    """Prepare one downloadable catalog dataset as a local trainable CSV asset."""

    dataset_name = _safe_dataset_name(metadata.name)
    download_url = _download_url(metadata)
    archive_format = _archive_format(metadata, download_url)
    if archive_format not in SUPPORTED_ARCHIVE_FORMATS:
        msg = f"unsupported archive_format: {archive_format}"
        raise ValueError(msg)

    version = metadata.version or "v1"
    dataset_dir = external_root / dataset_name / version
    _assert_inside_root(dataset_dir, external_root)
    prepared_csv_path = dataset_dir / _prepared_filename(metadata, dataset_name)
    raw_path = _download_destination(dataset_dir, download_url, prepared_csv_path, archive_format)

    manifest = read_dataset_cache_manifest(cache_root)
    cached = manifest["assets"].get(dataset_name)
    if (
        not force
        and isinstance(cached, dict)
        and cached.get("prepared") is True
        and prepared_csv_path.exists()
    ):
        return dataset_asset_status(metadata, cache_root=cache_root)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    if force or not raw_path.exists():
        (downloader or _default_download)(download_url, raw_path)
    _verify_checksum(raw_path, metadata.checksum)
    _materialize_csv(archive_format, raw_path, prepared_csv_path, metadata)

    record = _asset_record(metadata, prepared_csv_path, raw_path, cache_root)
    manifest["assets"][dataset_name] = record
    _write_manifest(manifest, cache_root)
    _write_prepared_catalog(manifest, cache_root)
    _write_default_config(record, cache_root)
    return record


def dataset_asset_status(
    metadata: DatasetMetadata,
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> dict[str, Any]:
    """Return cached preparation status for one catalog dataset."""

    dataset_name = _safe_dataset_name(metadata.name)
    manifest = read_dataset_cache_manifest(cache_root)
    cached = manifest["assets"].get(dataset_name)
    if isinstance(cached, dict):
        return dict(cached)
    return {
        **_metadata_asset_fields(metadata),
        "name": metadata.name,
        "version": metadata.version or "v1",
        "prepared": False,
        "path": None,
        "raw_path": None,
        "catalog_path": None,
        "config_path": None,
        "row_count": None,
        "column_count": None,
        "columns": [],
        "warnings": [],
        "updated_at": None,
    }


def prepared_metadata_from_status(
    metadata: DatasetMetadata,
    status: dict[str, Any],
) -> DatasetMetadata:
    """Return trainable CSV metadata when an external dataset has been prepared."""

    if status.get("prepared") is not True or not status.get("path"):
        return metadata
    return replace(
        metadata,
        dataset_type="csv",
        path=str(status["path"]),
        source=metadata.source,
        target_cols=_string_list(status.get("target_cols")) or metadata.target_cols,
        feature_cols=_string_list(status.get("feature_cols")) or metadata.feature_cols,
        timestamp_col=_optional_string(status.get("timestamp_col")) or metadata.timestamp_col,
        prepared=True,
    )


def list_dataset_cache(
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> dict[str, Any]:
    """Return all cached dataset asset records."""

    manifest = read_dataset_cache_manifest(cache_root)
    return {
        "manifest_path": str(cache_root / MANIFEST_FILENAME),
        "assets": sorted(
            [dict(asset) for asset in manifest["assets"].values() if isinstance(asset, dict)],
            key=lambda item: str(item.get("name", "")),
        ),
    }


def clear_dataset_asset(
    dataset_name: str,
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    external_root: Path = DEFAULT_EXTERNAL_ROOT,
) -> dict[str, Any]:
    """Remove one prepared dataset asset from the manifest and local external cache."""

    safe_name = _safe_dataset_name(dataset_name)
    manifest = read_dataset_cache_manifest(cache_root)
    removed = manifest["assets"].pop(safe_name, None)
    _write_manifest(manifest, cache_root)
    _write_prepared_catalog(manifest, cache_root)

    dataset_dir = external_root / safe_name
    _assert_inside_root(dataset_dir, external_root)
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    return {"removed": removed is not None, "dataset": safe_name}


def read_dataset_cache_manifest(
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> dict[str, Any]:
    """Read the dataset asset cache manifest."""

    manifest_path = cache_root / MANIFEST_FILENAME
    if not manifest_path.exists():
        return {"schema_version": 1, "assets": {}}
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = "dataset cache manifest root must be a mapping"
        raise ValueError(msg)
    assets = raw.get("assets")
    if not isinstance(assets, dict):
        msg = "dataset cache manifest assets must be a mapping"
        raise ValueError(msg)
    return {"schema_version": raw.get("schema_version", 1), "assets": assets}


def _asset_record(
    metadata: DatasetMetadata,
    prepared_csv_path: Path,
    raw_path: Path,
    cache_root: Path,
) -> dict[str, Any]:
    target_cols = metadata.target_cols or _fallback_target_cols(prepared_csv_path)
    timestamp_col = metadata.timestamp_col
    profile = profile_csv_dataset(
        path=prepared_csv_path,
        target_cols=target_cols,
        timestamp_col=timestamp_col,
        input_len=DEFAULT_INPUT_LEN,
        output_len=DEFAULT_OUTPUT_LEN,
        name=metadata.name,
    ).to_dict()
    record = {
        **_metadata_asset_fields(metadata),
        "name": metadata.name,
        "version": metadata.version or "v1",
        "prepared": True,
        "path": str(prepared_csv_path),
        "raw_path": str(raw_path),
        "catalog_path": str(cache_root / PREPARED_CATALOG_FILENAME),
        "config_path": str(_default_config_path(metadata.name, cache_root)),
        "row_count": profile["row_count"],
        "column_count": profile["column_count"],
        "columns": profile["columns"],
        "target_cols": target_cols,
        "feature_cols": metadata.feature_cols or [],
        "timestamp_col": timestamp_col,
        "warnings": profile["warnings"],
        "updated_at": _now_iso(),
    }
    return record


def _metadata_asset_fields(metadata: DatasetMetadata) -> dict[str, Any]:
    return {
        "dataset_type": metadata.dataset_type,
        "domain": metadata.domain,
        "description": metadata.description,
        "source": metadata.source,
        "download_url": metadata.download_url,
        "archive_format": metadata.archive_format,
        "frequency": metadata.frequency,
        "license": metadata.license,
        "citation": metadata.citation,
        "checksum": metadata.checksum,
    }


def _write_manifest(manifest: dict[str, Any], cache_root: Path) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_prepared_catalog(manifest: dict[str, Any], cache_root: Path) -> None:
    rows = []
    for asset in manifest["assets"].values():
        if not isinstance(asset, dict) or asset.get("prepared") is not True:
            continue
        rows.append(
            {
                "name": asset["name"],
                "dataset_type": "csv",
                "domain": asset.get("domain") or "external",
                "description": asset.get("description") or "Prepared external dataset.",
                "source": asset.get("source") or asset.get("download_url") or "external",
                "path": asset["path"],
                "timestamp_col": asset.get("timestamp_col"),
                "target_cols": asset.get("target_cols") or [],
                "feature_cols": asset.get("feature_cols") or [],
                "frequency": asset.get("frequency"),
                "license": asset.get("license"),
                "citation": asset.get("citation"),
            }
        )
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / PREPARED_CATALOG_FILENAME).write_text(
        yaml.safe_dump({"datasets": rows}, sort_keys=False),
        encoding="utf-8",
    )


def _write_default_config(record: dict[str, Any], cache_root: Path) -> None:
    metadata = DatasetMetadata(
        name=str(record["name"]),
        dataset_type="csv",
        domain=str(record.get("domain") or "external"),
        description=str(record.get("description") or "Prepared external dataset."),
        source=str(record.get("source") or record.get("download_url") or "external"),
        path=str(record["path"]),
        frequency=_optional_string(record.get("frequency")),
        license=_optional_string(record.get("license")),
        citation=_optional_string(record.get("citation")),
        target_cols=_string_list(record.get("target_cols")),
        feature_cols=_string_list(record.get("feature_cols")),
        timestamp_col=_optional_string(record.get("timestamp_col")),
    )
    config = config_from_catalog_metadata(
        metadata,
        input_len=DEFAULT_INPUT_LEN,
        output_len=DEFAULT_OUTPUT_LEN,
        model_name=DEFAULT_MODEL_NAME,
        epochs=DEFAULT_EPOCHS,
        batch_size=DEFAULT_BATCH_SIZE,
    )
    save_generated_config(config, _default_config_path(metadata.name, cache_root))


def _default_config_path(dataset_name: str, cache_root: Path) -> Path:
    config_name = f"train_{_safe_dataset_name(dataset_name)}_{DEFAULT_MODEL_NAME}.yaml"
    return cache_root / "configs" / config_name


def _materialize_csv(
    archive_format: str,
    raw_path: Path,
    prepared_csv_path: Path,
    metadata: DatasetMetadata,
) -> None:
    prepared_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_format in {"raw_csv", "csv"}:
        if raw_path != prepared_csv_path:
            shutil.copyfile(raw_path, prepared_csv_path)
        return
    if archive_format == "raw_txt" or archive_format == "raw_matrix":
        _matrix_text_to_csv(raw_path, prepared_csv_path)
        return
    if archive_format == "zip_csv":
        _zip_csv_to_csv(raw_path, prepared_csv_path, metadata.local_path)
        return
    msg = f"unsupported archive_format: {archive_format}"
    raise ValueError(msg)


def _matrix_text_to_csv(raw_path: Path, prepared_csv_path: Path) -> None:
    frame = pd.read_csv(raw_path, header=None, sep=None, engine="python")
    frame.columns = [f"value_{index}" for index in range(len(frame.columns))]
    frame.to_csv(prepared_csv_path, index=False)


def _zip_csv_to_csv(raw_path: Path, prepared_csv_path: Path, member_name: str | None) -> None:
    with zipfile.ZipFile(raw_path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if member_name is not None and member_name in archive.namelist():
            selected = member_name
        elif members:
            selected = members[0]
        else:
            msg = f"zip archive contains no CSV files: {raw_path}"
            raise ValueError(msg)
        with archive.open(selected) as source, prepared_csv_path.open("wb") as destination:
            shutil.copyfileobj(source, destination)


def _download_destination(
    dataset_dir: Path,
    download_url: str,
    prepared_csv_path: Path,
    archive_format: str,
) -> Path:
    if archive_format in {"raw_csv", "csv"}:
        return prepared_csv_path
    filename = Path(urllib.parse.urlparse(download_url).path).name or "downloaded_dataset"
    safe_filename = _safe_filename(filename)
    return dataset_dir / safe_filename


def _default_download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _verify_checksum(path: Path, checksum: str | None) -> None:
    if checksum is None:
        return
    if ":" in checksum:
        algorithm, expected = checksum.split(":", 1)
    else:
        algorithm, expected = "sha256", checksum
    if algorithm.lower() != "sha256":
        msg = f"unsupported checksum algorithm: {algorithm}"
        raise ValueError(msg)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest.lower() != expected.lower():
        msg = f"checksum mismatch for {path}"
        raise ValueError(msg)


def _download_url(metadata: DatasetMetadata) -> str:
    candidate = metadata.download_url or metadata.source
    if not candidate or not candidate.startswith(("http://", "https://")):
        msg = f"dataset {metadata.name!r} does not define a download_url"
        raise ValueError(msg)
    return candidate


def _archive_format(metadata: DatasetMetadata, download_url: str) -> str:
    if metadata.archive_format:
        return metadata.archive_format.lower()
    suffix = Path(urllib.parse.urlparse(download_url).path).suffix.lower()
    if suffix == ".zip":
        return "zip_csv"
    if suffix in {".txt", ".tsv"}:
        return "raw_matrix"
    return "raw_csv"


def _prepared_filename(metadata: DatasetMetadata, dataset_name: str) -> str:
    if metadata.archive_format == "zip_csv" and metadata.local_path:
        return f"{dataset_name}.csv"
    candidate = metadata.local_path or f"{dataset_name}.csv"
    return _safe_filename(candidate)


def _safe_dataset_name(value: str) -> str:
    return validate_safe_path_component(value.strip().lower(), field_name="dataset.name")


def _safe_filename(value: str) -> str:
    name = Path(value).name
    if not name:
        msg = "local_path must include a file name"
        raise ValueError(msg)
    if name in {".", ".."} or ".." in name:
        msg = "local_path must be a safe file name"
        raise ValueError(msg)
    return name


def _fallback_target_cols(prepared_csv_path: Path) -> list[str]:
    columns = list(pd.read_csv(prepared_csv_path, nrows=0).columns)
    if not columns:
        msg = f"prepared CSV has no columns: {prepared_csv_path}"
        raise ValueError(msg)
    lower = {column.lower(): column for column in columns}
    if "value" in lower:
        return [lower["value"]]
    if "ot" in lower:
        return [lower["ot"]]
    return [columns[-1]]


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _assert_inside_root(path: Path, root: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        msg = f"dataset asset path escapes root: {path}"
        raise ValueError(msg)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def metadata_to_prepared_dict(metadata: DatasetMetadata, status: dict[str, Any]) -> dict[str, Any]:
    """Return API-ready metadata merged with one asset status record."""

    prepared_metadata = prepared_metadata_from_status(metadata, status)
    payload = asdict(prepared_metadata)
    payload["asset"] = status
    payload["prepared"] = status.get("prepared") is True
    payload["source_dataset_type"] = metadata.dataset_type
    return payload
