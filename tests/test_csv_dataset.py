from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.helpers import tiny_config
from ts_platform.data.csv_dataset import CSVForecastDataset
from ts_platform.data.transforms import FeatureAwareScalerBundle, ScaledForecastingDataset
from ts_platform.runner.trainer import Trainer
from ts_platform.scaler.base import IdentityScaler
from ts_platform.scaler.standard import StandardScaler

FIXTURE = Path("tests/fixtures/tiny_series.csv")
FEATURE_FIXTURE = Path("tests/fixtures/tiny_series_with_features.csv")


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


def _feature_dataset(mode: str = "train", **params) -> CSVForecastDataset:
    defaults = {
        "input_len": 8,
        "output_len": 2,
        "mode": mode,
        "train_ratio": 0.5,
        "val_ratio": 0.25,
        "test_ratio": 0.25,
        "path": FEATURE_FIXTURE,
        "timestamp_col": "timestamp",
        "target_cols": ["value"],
        "feature_cols": ["temperature", "holiday"],
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


def _write_feature_csv(path: Path, temperatures: list[float | None]) -> None:
    rows = ["timestamp,value,temperature,holiday"]
    for index, temperature in enumerate(temperatures, start=1):
        rendered_temperature = "" if temperature is None else str(temperature)
        holiday = 1 if index % 7 == 0 else 0
        rows.append(f"2024-01-{index:02d},{float(index)},{rendered_temperature},{holiday}")
    path.write_text("\n".join(rows), encoding="utf-8")


def test_csv_dataset_loads_windows() -> None:
    dataset = _dataset()
    sample = dataset[0]

    assert len(dataset) > 0
    assert sample["x"].shape == (8, 1)
    assert sample["y"].shape == (2, 1)
    assert sample["x"].dtype == torch.float32
    assert sample["y"].dtype == torch.float32


def test_csv_dataset_with_feature_cols_shapes() -> None:
    dataset = _feature_dataset()
    sample = dataset[0]

    assert sample["x"].shape == (8, 3)
    assert sample["y"].shape == (2, 1)


def test_csv_dataset_with_feature_cols_x_concatenation_order() -> None:
    dataset = _feature_dataset(input_len=3, output_len=2)
    sample = dataset[0]

    assert sample["x"].tolist() == [
        [1.0, 10.0, 1.0],
        [2.0, 10.5, 0.0],
        [3.0, 11.0, 0.0],
    ]


def test_csv_dataset_y_remains_target_only() -> None:
    dataset = _feature_dataset(input_len=3, output_len=2)
    sample = dataset[0]

    assert sample["y"].shape == (2, 1)
    assert sample["y"].flatten().tolist() == [4.0, 5.0]


def test_csv_dataset_returns_target_x_and_feature_x() -> None:
    dataset = _feature_dataset(input_len=3, output_len=2)
    sample = dataset[0]

    assert sample["target_x"].shape == (3, 1)
    assert sample["feature_x"].shape == (3, 2)
    assert sample["target_x"].flatten().tolist() == [1.0, 2.0, 3.0]
    assert sample["feature_x"].tolist() == [
        [10.0, 1.0],
        [10.5, 0.0],
        [11.0, 0.0],
    ]


def test_csv_dataset_metadata_contains_columns() -> None:
    dataset = _feature_dataset()
    sample = dataset[0]

    assert sample["metadata"] == {
        "target_cols": ["value"],
        "feature_cols": ["temperature", "holiday"],
        "input_dim": 3,
        "target_dim": 1,
        "feature_dim": 2,
    }


def test_csv_dataset_feature_cols_do_not_change_scaler_fit_values() -> None:
    dataset = _feature_dataset()

    values = dataset.scaler_fit_values()
    assert values.shape == (20, 1)
    assert values.flatten().tolist() == [float(value) for value in range(1, 21)]


def test_csv_dataset_feature_scaler_fit_values() -> None:
    dataset = _feature_dataset()

    values = dataset.feature_scaler_fit_values()
    assert values.tolist()[0] == [10.0, 1.0]
    assert values.tolist()[-1] == [19.5, 0.0]


def test_csv_dataset_feature_scaler_fit_values_shape() -> None:
    dataset = _feature_dataset()

    assert dataset.feature_scaler_fit_values().shape == (20, 2)


def test_csv_dataset_rejects_missing_feature_column() -> None:
    with pytest.raises(ValueError, match="CSV feature columns are missing"):
        _feature_dataset(feature_cols=["missing"])


def test_csv_dataset_rejects_non_numeric_feature_column(tmp_path) -> None:
    path = tmp_path / "non_numeric_feature.csv"
    path.write_text(
        "timestamp,value,temperature,holiday\n"
        "2024-01-01,1,warm,0\n"
        "2024-01-02,2,11.0,0\n"
        "2024-01-03,3,12.0,0\n"
        "2024-01-04,4,13.0,0\n"
        "2024-01-05,5,14.0,0\n"
        "2024-01-06,6,15.0,0\n"
        "2024-01-07,7,16.0,1\n"
        "2024-01-08,8,17.0,0\n"
        "2024-01-09,9,18.0,0\n"
        "2024-01-10,10,19.0,0\n"
        "2024-01-11,11,20.0,0\n"
        "2024-01-12,12,21.0,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="CSV feature column 'temperature' must be numeric"):
        _feature_dataset(
            path=path,
            input_len=2,
            output_len=1,
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
        )


def test_csv_dataset_rejects_feature_missing_values(tmp_path) -> None:
    path = tmp_path / "missing_feature.csv"
    _write_feature_csv(
        path, [10.0, None, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0]
    )

    with pytest.raises(ValueError, match="feature columns contain missing values"):
        _feature_dataset(
            path=path,
            input_len=2,
            output_len=1,
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
        )


def test_csv_dataset_feature_missing_is_split_local(tmp_path) -> None:
    path = tmp_path / "split_missing_feature.csv"
    _write_feature_csv(
        path, [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, None, 17.0, 18.0, 19.0, 20.0, 21.0]
    )

    train = _feature_dataset(
        "train",
        path=path,
        input_len=2,
        output_len=1,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
    )
    assert len(train) == 4

    with pytest.raises(ValueError, match="val split feature columns contain missing values"):
        _feature_dataset(
            "val",
            path=path,
            input_len=2,
            output_len=1,
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
        )


def test_csv_dataset_target_only_shape_unchanged() -> None:
    dataset = _dataset()
    sample = dataset[0]

    assert sample["x"].shape == (8, 1)
    assert sample["y"].shape == (2, 1)
    assert set(sample) == {"x", "y"}


def test_csv_dataset_dimensions_with_feature_cols() -> None:
    dataset = _feature_dataset()

    assert dataset.target_dim == 1
    assert dataset.feature_dim == 2
    assert dataset.input_dim == 3
    assert dataset.num_features == 1
    assert dataset.dimensions.input_dim == 3
    assert dataset.dimensions.target_dim == 1


def test_target_only_dataset_feature_scaler_fit_values_fails() -> None:
    dataset = _dataset()

    with pytest.raises(ValueError, match="dataset has no feature values for scaler fitting"):
        dataset.feature_scaler_fit_values()


def test_scaled_dataset_delegates_feature_scaler_fit_values() -> None:
    dataset = _feature_dataset()
    scaled = ScaledForecastingDataset(
        dataset,
        FeatureAwareScalerBundle(target=IdentityScaler(), features=IdentityScaler()),
    )

    assert torch.equal(scaled.feature_scaler_fit_values(), dataset.feature_scaler_fit_values())


def test_feature_aware_scaler_bundle_target_only() -> None:
    bundle = FeatureAwareScalerBundle(target=IdentityScaler())

    assert bundle.target.fitted is True
    assert bundle.features is None
    assert not bundle.has_features()


def test_feature_aware_scaler_bundle_with_features() -> None:
    bundle = FeatureAwareScalerBundle(target=IdentityScaler(), features=IdentityScaler())

    assert bundle.has_features()


def test_scaled_dataset_feature_aware_requires_bundle() -> None:
    dataset = _feature_dataset()

    with pytest.raises(
        ValueError,
        match="feature-aware datasets require FeatureAwareScalerBundle",
    ):
        ScaledForecastingDataset(dataset, IdentityScaler())


def test_scaled_dataset_feature_aware_requires_feature_scaler() -> None:
    dataset = _feature_dataset()

    with pytest.raises(ValueError, match="feature-aware datasets require feature scaler"):
        ScaledForecastingDataset(dataset, FeatureAwareScalerBundle(target=IdentityScaler()))


def test_scaled_dataset_feature_aware_scales_target_and_y_with_target_scaler() -> None:
    dataset = _feature_dataset(input_len=3, output_len=2)
    target_scaler = StandardScaler().fit(dataset.target_scaler_fit_values())
    feature_scaler = StandardScaler().fit(dataset.feature_scaler_fit_values())
    scaled = ScaledForecastingDataset(
        dataset,
        FeatureAwareScalerBundle(target=target_scaler, features=feature_scaler),
    )

    raw_sample = dataset[0]
    sample = scaled[0]

    assert torch.allclose(sample["target_x"], target_scaler.transform(raw_sample["target_x"]))
    assert torch.allclose(sample["y"], target_scaler.transform(raw_sample["y"]))


def test_scaled_dataset_feature_aware_scales_features_with_feature_scaler() -> None:
    dataset = _feature_dataset(input_len=3, output_len=2)
    target_scaler = StandardScaler().fit(dataset.target_scaler_fit_values())
    feature_scaler = StandardScaler().fit(dataset.feature_scaler_fit_values())
    scaled = ScaledForecastingDataset(
        dataset,
        FeatureAwareScalerBundle(target=target_scaler, features=feature_scaler),
    )

    raw_sample = dataset[0]
    sample = scaled[0]

    assert torch.allclose(sample["feature_x"], feature_scaler.transform(raw_sample["feature_x"]))


def test_scaled_dataset_feature_aware_reconstructs_x_concatenation() -> None:
    dataset = _feature_dataset(input_len=3, output_len=2)
    target_scaler = StandardScaler().fit(dataset.target_scaler_fit_values())
    feature_scaler = StandardScaler().fit(dataset.feature_scaler_fit_values())
    scaled = ScaledForecastingDataset(
        dataset,
        FeatureAwareScalerBundle(target=target_scaler, features=feature_scaler),
    )

    sample = scaled[0]

    assert torch.allclose(sample["x"], torch.cat([sample["target_x"], sample["feature_x"]], dim=-1))


def test_scaled_dataset_feature_aware_preserves_metadata() -> None:
    dataset = _feature_dataset()
    scaled = ScaledForecastingDataset(
        dataset,
        FeatureAwareScalerBundle(target=IdentityScaler(), features=IdentityScaler()),
    )

    assert scaled[0]["metadata"] == dataset[0]["metadata"]


def test_trainer_rejects_feature_aware_csv_until_phase12d(tmp_path) -> None:
    config = tiny_config(tmp_path, name="feature_csv_blocked")
    data = config.data.model_copy(
        update={
            "name": "csv",
            "input_len": 2,
            "output_len": 1,
            "batch_size": 4,
            "train_ratio": 0.5,
            "val_ratio": 0.25,
            "test_ratio": 0.25,
            "params": {
                "path": str(FEATURE_FIXTURE),
                "timestamp_col": "timestamp",
                "target_cols": ["value"],
                "feature_cols": ["temperature"],
                "missing_policy": "error",
                "sort_by_time": True,
            },
        }
    )
    config = config.model_copy(update={"data": data})

    with pytest.raises(
        NotImplementedError,
        match="feature-aware training is not implemented until Phase 12D/12E",
    ):
        Trainer(config).run()


def test_scaled_dataset_target_only_still_works() -> None:
    dataset = _dataset()
    scaled = ScaledForecastingDataset(dataset, IdentityScaler())
    sample = scaled[0]

    assert scaled.input_dim == 1
    assert scaled.target_dim == 1
    assert sample["x"].shape == (8, 1)
    assert sample["y"].shape == (2, 1)


def test_trainer_target_only_csv_still_works(tmp_path) -> None:
    config = tiny_config(tmp_path, name="target_only_csv")
    data = config.data.model_copy(
        update={
            "name": "csv",
            "input_len": 8,
            "output_len": 2,
            "batch_size": 4,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "params": {
                "path": str(FIXTURE),
                "timestamp_col": "timestamp",
                "target_cols": ["value"],
                "missing_policy": "error",
                "sort_by_time": True,
            },
        }
    )
    config = config.model_copy(update={"data": data})

    result = Trainer(config).run()

    assert result.checkpoint_path.exists()
    assert "original" in result.test_metrics


def test_trainer_synthetic_still_works(tmp_path) -> None:
    result = Trainer(tiny_config(tmp_path, name="synthetic_still_works")).run()

    assert result.checkpoint_path.exists()
    assert "original" in result.test_metrics


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


def test_csv_params_accepts_feature_cols() -> None:
    dataset = _feature_dataset(feature_cols=["temperature"])

    assert dataset.feature_cols == ["temperature"]


def test_csv_params_accepts_empty_feature_cols() -> None:
    dataset = _dataset(feature_cols=[])
    sample = dataset[0]

    assert dataset.feature_cols == []
    assert sample["x"].shape == (8, 1)
    assert sample["y"].shape == (2, 1)
    assert set(sample) == {"x", "y"}


def test_csv_params_rejects_feature_cols_string() -> None:
    with pytest.raises(ValueError, match="feature_cols must be a list of strings or None"):
        _dataset(feature_cols="temperature")


def test_csv_params_rejects_empty_feature_col_name() -> None:
    with pytest.raises(ValueError, match="feature_cols must contain non-empty strings"):
        _dataset(feature_cols=[""])


def test_csv_params_rejects_feature_target_overlap() -> None:
    with pytest.raises(ValueError, match="feature_cols must not overlap target_cols"):
        _dataset(feature_cols=["value"])


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
