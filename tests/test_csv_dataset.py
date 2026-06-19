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

    assert train.split_start == 0
    assert train.split_end <= val.split_start
    assert val.split_end <= test.split_start
    assert test.split_end == 90
    assert train.split_timestamps[-1] < val.split_timestamps[0]
    assert val.split_timestamps[-1] < test.split_timestamps[0]


def test_csv_dataset_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _dataset(path=tmp_path / "missing.csv")


def test_csv_dataset_rejects_missing_target_column() -> None:
    with pytest.raises(ValueError, match="target columns are missing"):
        _dataset(target_cols=["missing"])


def test_csv_dataset_missing_policy_error(tmp_path) -> None:
    path = tmp_path / "missing.csv"
    path.write_text(
        "timestamp,value\n2024-01-01,1\n2024-01-02,\n2024-01-03,3\n",
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
