from __future__ import annotations

from pathlib import Path

import pytest

from ts_platform.data.profile import profile_csv_dataset

FIXTURE = Path("tests/fixtures/tiny_series.csv")


def test_profile_csv_dataset_basic() -> None:
    profile = profile_csv_dataset(
        path=FIXTURE,
        target_cols=["value"],
        timestamp_col="timestamp",
        input_len=8,
        output_len=2,
        name="tiny_csv",
    )

    assert profile.name == "tiny_csv"
    assert profile.dataset_type == "csv"
    assert profile.exists is True
    assert profile.row_count == 90
    assert profile.column_count == 2
    assert profile.columns == ["timestamp", "value"]
    assert profile.start_timestamp == "2024-01-01T00:00:00"
    assert profile.end_timestamp == "2024-03-30T00:00:00"
    assert profile.target_missing_counts == {"value": 0}
    assert profile.target_dtypes == {"value": "float64"}
    assert profile.duplicate_timestamp_count == 0
    assert profile.inferred_frequency == "D"
    assert profile.min_required_rows == 10
    assert profile.can_build_windows is True
    assert profile.warnings == []


def test_profile_csv_dataset_missing_file(tmp_path) -> None:
    profile = profile_csv_dataset(
        path=tmp_path / "missing.csv",
        target_cols=["value"],
        timestamp_col="timestamp",
        input_len=8,
        output_len=2,
    )

    assert profile.exists is False
    assert profile.row_count == 0
    assert profile.can_build_windows is False
    assert "missing file" in profile.warnings


def test_profile_csv_dataset_missing_file_without_lengths_cannot_build_windows(tmp_path) -> None:
    profile = profile_csv_dataset(
        path=tmp_path / "missing.csv",
        target_cols=["value"],
        timestamp_col="timestamp",
    )

    assert profile.exists is False
    assert profile.can_build_windows is False
    assert "missing file" in profile.warnings


def test_profile_csv_dataset_missing_target_column() -> None:
    profile = profile_csv_dataset(path=FIXTURE, target_cols=["missing"], timestamp_col="timestamp")

    assert profile.target_missing_counts == {}
    assert profile.target_dtypes == {}
    assert "missing target column: missing" in profile.warnings


def test_profile_csv_dataset_missing_timestamp_column() -> None:
    profile = profile_csv_dataset(path=FIXTURE, target_cols=["value"], timestamp_col="missing_ts")

    assert profile.start_timestamp is None
    assert profile.end_timestamp is None
    assert profile.duplicate_timestamp_count is None
    assert "missing timestamp column: missing_ts" in profile.warnings


def test_profile_csv_dataset_counts_missing_targets(tmp_path) -> None:
    path = tmp_path / "missing_values.csv"
    path.write_text(
        "timestamp,value\n2024-01-01,1\n2024-01-02,\n2024-01-03,3\n",
        encoding="utf-8",
    )

    profile = profile_csv_dataset(path=path, target_cols=["value"], timestamp_col="timestamp")

    assert profile.target_missing_counts == {"value": 1}
    assert "target has missing values: value" in profile.warnings


def test_profile_csv_dataset_detects_duplicate_timestamps(tmp_path) -> None:
    path = tmp_path / "duplicate_ts.csv"
    path.write_text(
        "timestamp,value\n2024-01-01,1\n2024-01-01,2\n2024-01-02,3\n",
        encoding="utf-8",
    )

    profile = profile_csv_dataset(path=path, target_cols=["value"], timestamp_col="timestamp")

    assert profile.duplicate_timestamp_count == 1
    assert "duplicate timestamps: 1" in profile.warnings
    assert "cannot infer frequency" in profile.warnings


def test_profile_csv_dataset_detects_insufficient_rows(tmp_path) -> None:
    path = tmp_path / "short.csv"
    path.write_text("timestamp,value\n2024-01-01,1\n2024-01-02,2\n", encoding="utf-8")

    profile = profile_csv_dataset(
        path=path,
        target_cols=["value"],
        timestamp_col="timestamp",
        input_len=2,
        output_len=2,
    )

    assert profile.min_required_rows == 4
    assert profile.can_build_windows is False
    assert "insufficient rows for windows" in profile.warnings


def test_profile_csv_dataset_to_dict() -> None:
    profile = profile_csv_dataset(path=FIXTURE, target_cols=["value"], timestamp_col="timestamp")

    payload = profile.to_dict()

    assert payload["dataset_type"] == "csv"
    assert payload["target_cols"] == ["value"]
    assert payload["exists"] is True


def test_profile_csv_dataset_rejects_empty_target_cols() -> None:
    with pytest.raises(ValueError, match="target_cols must be non-empty"):
        profile_csv_dataset(path=FIXTURE, target_cols=[])
