from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ts_platform.data.csv_dataset import CSVForecastDataset

FIXTURE = Path("tests/fixtures/tiny_series.csv")


def _dataset(mode: str = "train", **params) -> CSVForecastDataset:
    defaults = {
        "input_len": 8,
        "output_len": 2,
        "mode": mode,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "path": FIXTURE,
        "timestamp_col": "timestamp",
        "target_cols": ["value"],
        "missing_policy": "error",
    }
    defaults.update(params)
    return CSVForecastDataset(**defaults)


def _write_timestamp_csv(path: Path, values: list[float | None]) -> None:
    rows = ["timestamp,value"]
    for index, value in enumerate(values, start=1):
        rendered = "" if value is None else str(value)
        rows.append(f"2024-01-{index:02d},{rendered}")
    path.write_text("\n".join(rows), encoding="utf-8")


def test_csv_dataset_loads_windows() -> None:
    dataset = _dataset()
    sample = dataset[0]

    assert len(dataset) > 0
    assert sample["x"].shape == (8, 1)
    assert sample["y"].shape == (2, 1)
    assert sample["x"].dtype == torch.float32
    assert sample["y"].dtype == torch.float32


def test_csv_dataset_time_split_no_overlap() -> None:
    train = _dataset("train")
    val = _dataset("val")
    test = _dataset("test")

    train_meta = train.split_metadata()
    val_meta = val.split_metadata()
    test_meta = test.split_metadata()

    assert train_meta.split_start == 0
    assert train_meta.split_end <= val_meta.split_start
    assert val_meta.split_end <= test_meta.split_start
    assert test_meta.split_end == 90
    assert train_meta.end_timestamp is not None
    assert val_meta.start_timestamp is not None
    assert val_meta.end_timestamp is not None
    assert test_meta.start_timestamp is not None
    assert train_meta.end_timestamp < val_meta.start_timestamp
    assert val_meta.end_timestamp < test_meta.start_timestamp


def test_csv_params_rejects_target_cols_string() -> None:
    with pytest.raises(ValueError, match="target_cols must be a non-empty list"):
        _dataset(target_cols="value")


def test_csv_params_rejects_empty_target_cols() -> None:
    with pytest.raises(ValueError, match="target_cols must be a non-empty list"):
        _dataset(target_cols=[])


def test_csv_params_rejects_invalid_missing_policy_even_without_missing_values() -> None:
    with pytest.raises(ValueError, match="missing_policy"):
        _dataset(missing_policy="interpolate")


def test_csv_params_rejects_non_bool_sort_by_time() -> None:
    with pytest.raises(ValueError, match="sort_by_time must be a boolean"):
        _dataset(sort_by_time="true")


def test_csv_params_rejects_feature_cols_with_clear_error() -> None:
    with pytest.raises(ValueError, match="exogenous feature_cols are not supported yet"):
        _dataset(feature_cols=["temperature"])


def test_csv_dataset_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _dataset(path=tmp_path / "missing.csv")


def test_csv_dataset_rejects_missing_target_column() -> None:
    with pytest.raises(ValueError, match="target columns are missing"):
        _dataset(target_cols=["missing"])


def test_csv_dataset_missing_policy_error(tmp_path) -> None:
    path = tmp_path / "missing.csv"
    path.write_text(
        "timestamp,value\n"
        "2024-01-01,1\n"
        "2024-01-02,\n"
        "2024-01-03,3\n"
        "2024-01-04,4\n"
        "2024-01-05,5\n"
        "2024-01-06,6\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing values"):
        _dataset(
            path=path,
            input_len=1,
            output_len=1,
            train_ratio=1 / 3,
            val_ratio=1 / 3,
            test_ratio=1 / 3,
        )


def test_csv_dataset_missing_policy_forward_fill(tmp_path) -> None:
    path = tmp_path / "ffill.csv"
    path.write_text(
        "timestamp,value\n"
        "2024-01-01,1\n"
        "2024-01-02,\n"
        "2024-01-03,3\n"
        "2024-01-04,4\n"
        "2024-01-05,5\n"
        "2024-01-06,6\n",
        encoding="utf-8",
    )

    dataset = _dataset(
        path=path,
        input_len=1,
        output_len=1,
        train_ratio=1 / 3,
        val_ratio=1 / 3,
        test_ratio=1 / 3,
        missing_policy="forward_fill",
    )

    assert dataset.scaler_fit_values().tolist() == [[1.0], [1.0]]


def test_csv_forward_fill_does_not_cross_split_boundary(tmp_path) -> None:
    path = tmp_path / "split_ffill.csv"
    _write_timestamp_csv(path, [1, 2, 3, None, 5, 6, 7, 8, 9])

    with pytest.raises(ValueError, match="val split forward_fill.*start of the split"):
        _dataset(
            "val",
            path=path,
            input_len=1,
            output_len=1,
            train_ratio=1 / 3,
            val_ratio=1 / 3,
            test_ratio=1 / 3,
            missing_policy="forward_fill",
        )


def test_csv_drop_missing_is_split_local(tmp_path) -> None:
    path = tmp_path / "split_drop.csv"
    _write_timestamp_csv(path, [1, None, 3, 4, 5, 6, 7, 8, 9])

    train = _dataset(
        "train",
        path=path,
        input_len=1,
        output_len=1,
        train_ratio=1 / 3,
        val_ratio=1 / 3,
        test_ratio=1 / 3,
        missing_policy="drop",
    )
    val = _dataset(
        "val",
        path=path,
        input_len=1,
        output_len=1,
        train_ratio=1 / 3,
        val_ratio=1 / 3,
        test_ratio=1 / 3,
        missing_policy="drop",
    )

    assert train.split_metadata().row_count == 2
    assert val.split_metadata().split_start == 3
    assert val.split_metadata().row_count == 3
    assert val[0]["x"].flatten().tolist() == [4.0]


def test_csv_zero_fill_is_split_local(tmp_path) -> None:
    path = tmp_path / "split_zero_fill.csv"
    _write_timestamp_csv(path, [1, 2, 3, None, 5, 6, 7, 8, 9])

    train = _dataset(
        "train",
        path=path,
        input_len=1,
        output_len=1,
        train_ratio=1 / 3,
        val_ratio=1 / 3,
        test_ratio=1 / 3,
        missing_policy="zero_fill",
    )
    val = _dataset(
        "val",
        path=path,
        input_len=1,
        output_len=1,
        train_ratio=1 / 3,
        val_ratio=1 / 3,
        test_ratio=1 / 3,
        missing_policy="zero_fill",
    )

    assert train.scaler_fit_values().flatten().tolist() == [1.0, 2.0, 3.0]
    assert val[0]["x"].flatten().tolist() == [0.0]


def test_csv_missing_policy_after_drop_checks_min_length(tmp_path) -> None:
    path = tmp_path / "drop_too_short.csv"
    _write_timestamp_csv(path, [1, None, 3, 4, 5, 6, 7, 8, 9])

    with pytest.raises(
        ValueError,
        match="train split has 2 rows after applying missing_policy='drop'; requires at least 3",
    ):
        _dataset(
            "train",
            path=path,
            input_len=2,
            output_len=1,
            train_ratio=1 / 3,
            val_ratio=1 / 3,
            test_ratio=1 / 3,
            missing_policy="drop",
        )


def test_csv_missing_policy_error_reports_split_name(tmp_path) -> None:
    path = tmp_path / "val_missing.csv"
    _write_timestamp_csv(path, [1, 2, 3, 4, None, 6, 7, 8, 9])

    with pytest.raises(ValueError, match="val split target columns contain 1 missing values"):
        _dataset(
            "val",
            path=path,
            input_len=1,
            output_len=1,
            train_ratio=1 / 3,
            val_ratio=1 / 3,
            test_ratio=1 / 3,
            missing_policy="error",
        )


def test_csv_split_metadata_contains_window_count() -> None:
    dataset = _dataset("train")
    metadata = dataset.split_metadata()

    assert metadata.mode == "train"
    assert metadata.row_count == 62
    assert metadata.window_count == len(dataset)
    assert metadata.window_count == 53


def test_csv_split_metadata_without_timestamp(tmp_path) -> None:
    path = tmp_path / "no_timestamp.csv"
    rows = ["value", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    path.write_text("\n".join(rows), encoding="utf-8")

    dataset = _dataset(
        "train",
        path=path,
        timestamp_col=None,
        input_len=1,
        output_len=1,
        train_ratio=1 / 3,
        val_ratio=1 / 3,
        test_ratio=1 / 3,
    )
    metadata = dataset.split_metadata()

    assert metadata.start_timestamp is None
    assert metadata.end_timestamp is None


def test_csv_scaler_fits_train_only(tmp_path) -> None:
    rows = ["timestamp,value"]
    rows.extend(f"2024-01-{day:02d},1.0" for day in range(1, 11))
    rows.extend(f"2024-01-{day:02d},100.0" for day in range(11, 21))
    path = tmp_path / "leakage_check.csv"
    path.write_text("\n".join(rows), encoding="utf-8")

    dataset = _dataset(
        path=path,
        input_len=2,
        output_len=1,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
    )

    values = dataset.scaler_fit_values()
    assert torch.max(values).item() == 1.0
