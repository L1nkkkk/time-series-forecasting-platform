"""Dataset profiling helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DatasetProfile:
    """Serializable profile for a local dataset."""

    name: str | None
    dataset_type: str
    path: str
    exists: bool
    row_count: int
    column_count: int
    columns: list[str]
    timestamp_col: str | None
    start_timestamp: str | None
    end_timestamp: str | None
    target_cols: list[str]
    target_missing_counts: dict[str, int]
    target_dtypes: dict[str, str]
    duplicate_timestamp_count: int | None
    inferred_frequency: str | None
    min_required_rows: int | None
    can_build_windows: bool
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable profile payload."""

        return asdict(self)


def profile_csv_dataset(
    *,
    path: str | Path,
    target_cols: list[str],
    timestamp_col: str | None = None,
    input_len: int | None = None,
    output_len: int | None = None,
    name: str | None = None,
) -> DatasetProfile:
    """Profile a local CSV dataset without mutating or cleaning data."""

    if not target_cols:
        msg = "target_cols must be non-empty"
        raise ValueError(msg)
    if not all(isinstance(column, str) and column for column in target_cols):
        msg = "target_cols must contain non-empty strings"
        raise ValueError(msg)
    if input_len is not None and input_len <= 0:
        msg = "input_len must be positive when provided"
        raise ValueError(msg)
    if output_len is not None and output_len <= 0:
        msg = "output_len must be positive when provided"
        raise ValueError(msg)

    path_text = str(path)
    if _is_remote_url(path_text):
        msg = "remote dataset URLs are not supported"
        raise ValueError(msg)

    min_required_rows = (
        input_len + output_len if input_len is not None and output_len is not None else None
    )
    csv_path = Path(path)
    warnings: list[str] = []
    if not csv_path.exists():
        warnings.append("missing file")
        return DatasetProfile(
            name=name,
            dataset_type="csv",
            path=path_text,
            exists=False,
            row_count=0,
            column_count=0,
            columns=[],
            timestamp_col=timestamp_col,
            start_timestamp=None,
            end_timestamp=None,
            target_cols=list(target_cols),
            target_missing_counts={},
            target_dtypes={},
            duplicate_timestamp_count=None,
            inferred_frequency=None,
            min_required_rows=min_required_rows,
            can_build_windows=False,
            warnings=warnings,
        )

    frame = pd.read_csv(csv_path)
    row_count = len(frame)
    columns = [str(column) for column in frame.columns]
    target_missing_counts: dict[str, int] = {}
    target_dtypes: dict[str, str] = {}
    for column in target_cols:
        if column not in frame.columns:
            warnings.append(f"missing target column: {column}")
            continue
        missing_count = int(frame[column].isna().sum())
        target_missing_counts[column] = missing_count
        target_dtypes[column] = str(frame[column].dtype)
        if missing_count > 0:
            warnings.append(f"target has missing values: {column}")

    start_timestamp: str | None = None
    end_timestamp: str | None = None
    duplicate_timestamp_count: int | None = None
    inferred_frequency: str | None = None
    if timestamp_col is not None:
        if timestamp_col not in frame.columns:
            warnings.append(f"missing timestamp column: {timestamp_col}")
        else:
            timestamps = pd.to_datetime(frame[timestamp_col], errors="coerce")
            non_null_timestamps = timestamps.dropna()
            if not non_null_timestamps.empty:
                start_timestamp = non_null_timestamps.min().isoformat()
                end_timestamp = non_null_timestamps.max().isoformat()
            duplicate_timestamp_count = int(timestamps.duplicated().sum())
            if duplicate_timestamp_count > 0:
                warnings.append(f"duplicate timestamps: {duplicate_timestamp_count}")
            inferred_frequency = _infer_frequency(non_null_timestamps)
            if inferred_frequency is None:
                warnings.append("cannot infer frequency")

    can_build_windows = True
    if min_required_rows is not None:
        can_build_windows = row_count >= min_required_rows
        if not can_build_windows:
            warnings.append("insufficient rows for windows")

    return DatasetProfile(
        name=name,
        dataset_type="csv",
        path=path_text,
        exists=True,
        row_count=row_count,
        column_count=len(columns),
        columns=columns,
        timestamp_col=timestamp_col,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        target_cols=list(target_cols),
        target_missing_counts=target_missing_counts,
        target_dtypes=target_dtypes,
        duplicate_timestamp_count=duplicate_timestamp_count,
        inferred_frequency=inferred_frequency,
        min_required_rows=min_required_rows,
        can_build_windows=can_build_windows,
        warnings=warnings,
    )


def _infer_frequency(timestamps: pd.Series) -> str | None:
    if len(timestamps) < 3:
        return None
    try:
        frequency = pd.infer_freq(timestamps.sort_values())
    except ValueError:
        return None
    return str(frequency) if frequency is not None else None


def _is_remote_url(path: str) -> bool:
    lowered = path.lower()
    return lowered.startswith(("http://", "https://"))
