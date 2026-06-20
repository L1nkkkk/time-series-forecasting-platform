"""CSV-backed forecasting dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import torch

from ts_platform.data.base import ForecastBatch, ForecastingDataset
from ts_platform.data.csv_params import CSVDatasetParams
from ts_platform.data.splits import compute_time_split_slices

SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class DatasetSplitMetadata:
    """Structured metadata for the selected CSV split."""

    mode: str
    split_start: int
    split_end: int
    row_count: int
    window_count: int
    start_timestamp: str | None
    end_timestamp: str | None


class CSVForecastDataset(ForecastingDataset):
    """Forecasting dataset backed by a local CSV file."""

    def __init__(
        self,
        input_len: int,
        output_len: int,
        mode: SplitName,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        *,
        seed: int | None = None,
        **params: Any,
    ) -> None:
        _ = seed
        csv_params = CSVDatasetParams.model_validate(params)

        self.input_len = input_len
        self.output_len = output_len
        self.target_cols = list(csv_params.target_cols)
        self.feature_cols = list(csv_params.feature_cols or [])
        self.target_dim = len(self.target_cols)
        self.feature_dim = len(self.feature_cols)
        self.input_dim = self.target_dim + self.feature_dim
        self.num_features = self.target_dim
        self.mode = mode
        self.path = Path(csv_params.path)
        self.timestamp_col = csv_params.timestamp_col
        self.missing_policy = csv_params.missing_policy
        self.sort_by_time = csv_params.sort_by_time

        frame = self._load_frame()
        min_length = input_len + output_len
        split_slices = compute_time_split_slices(
            len(frame),
            train_ratio,
            val_ratio,
            test_ratio,
            min_length=min_length,
        )
        selected = split_slices.for_mode(mode)
        self.split_start = selected.start
        self.split_end = selected.stop
        raw_split_frame = frame.iloc[self.split_start : self.split_end].copy()
        self._validate_feature_missing(raw_split_frame)
        self._split_frame = self._handle_missing(raw_split_frame).reset_index(drop=True)
        self._validate_split_min_length(
            row_count=len(self._split_frame),
            min_length=min_length,
            selected_ratio={"train": train_ratio, "val": val_ratio, "test": test_ratio}[mode],
        )
        self._target_values = torch.tensor(
            self._split_frame[self.target_cols].to_numpy(dtype="float32"),
            dtype=torch.float32,
        )
        self._feature_values = (
            torch.tensor(
                self._split_frame[self.feature_cols].to_numpy(dtype="float32"),
                dtype=torch.float32,
            )
            if self.feature_dim > 0
            else None
        )
        self._values = self._target_values
        self._starts = list(range(0, max(0, len(self._target_values) - min_length + 1)))

    @property
    def split_timestamps(self) -> list[str]:
        """Return split timestamps as ISO-like strings when timestamps are configured."""

        if self.timestamp_col is None:
            return []
        values = self._split_frame[self.timestamp_col]
        return [str(value) for value in values.tolist()]

    def split_metadata(self) -> DatasetSplitMetadata:
        """Return structured metadata for the selected split."""

        timestamps = self.split_timestamps
        return DatasetSplitMetadata(
            mode=self.mode,
            split_start=self.split_start,
            split_end=self.split_end,
            row_count=len(self._split_frame),
            window_count=len(self),
            start_timestamp=timestamps[0] if timestamps else None,
            end_timestamp=timestamps[-1] if timestamps else None,
        )

    def __len__(self) -> int:
        """Return number of forecasting windows."""

        return len(self._starts)

    def __getitem__(self, index: int) -> ForecastBatch:
        """Return one forecasting window."""

        start = self._starts[index]
        x_end = start + self.input_len
        y_end = x_end + self.output_len
        target_x = self._target_values[start:x_end]
        y = self._target_values[x_end:y_end]
        if self._feature_values is None:
            return {"x": target_x, "y": y}
        feature_x = self._feature_values[start:x_end]
        return {
            "x": torch.cat([target_x, feature_x], dim=-1),
            "y": y,
            "target_x": target_x,
            "feature_x": feature_x,
            "metadata": {
                "target_cols": self.target_cols,
                "feature_cols": self.feature_cols,
                "input_dim": self.input_dim,
                "target_dim": self.target_dim,
                "feature_dim": self.feature_dim,
            },
        }

    def scaler_fit_values(self) -> torch.Tensor:
        """Return values from this split for scaler fitting."""

        if self._values.numel() == 0:
            msg = f"{self.mode} split has no values for scaler fitting"
            raise ValueError(msg)
        return self._values

    def feature_scaler_fit_values(self) -> torch.Tensor:
        """Return feature values from this split for feature scaler fitting."""

        if self._feature_values is None or self._feature_values.numel() == 0:
            msg = "dataset has no feature values for scaler fitting"
            raise ValueError(msg)
        return self._feature_values

    def _load_frame(self) -> pd.DataFrame:
        if not self.path.exists():
            msg = f"CSV dataset file does not exist: {self.path}"
            raise FileNotFoundError(msg)
        frame = pd.read_csv(self.path)
        missing_targets = [column for column in self.target_cols if column not in frame.columns]
        if missing_targets:
            msg = f"CSV target columns are missing: {missing_targets}"
            raise ValueError(msg)
        missing_features = [column for column in self.feature_cols if column not in frame.columns]
        if missing_features:
            msg = f"CSV feature columns are missing: {missing_features}"
            raise ValueError(msg)
        if self.timestamp_col is not None:
            if self.timestamp_col not in frame.columns:
                msg = f"CSV timestamp column is missing: {self.timestamp_col}"
                raise ValueError(msg)
            frame[self.timestamp_col] = pd.to_datetime(frame[self.timestamp_col], errors="raise")
            if frame[self.timestamp_col].duplicated().any():
                msg = f"CSV timestamp column contains duplicate values: {self.timestamp_col}"
                raise ValueError(msg)
            if self.sort_by_time:
                frame = frame.sort_values(self.timestamp_col).reset_index(drop=True)

        for column in self.target_cols:
            try:
                frame[column] = pd.to_numeric(frame[column], errors="raise")
            except ValueError as exc:
                msg = f"CSV target column {column!r} must be numeric"
                raise ValueError(msg) from exc

        for column in self.feature_cols:
            try:
                frame[column] = pd.to_numeric(frame[column], errors="raise")
            except ValueError as exc:
                msg = f"CSV feature column {column!r} must be numeric"
                raise ValueError(msg) from exc

        return frame.reset_index(drop=True)

    def _validate_feature_missing(self, frame: pd.DataFrame) -> None:
        if not self.feature_cols:
            return
        missing_count = int(frame[self.feature_cols].isna().sum().sum())
        if missing_count:
            msg = f"{self.mode} split feature columns contain missing values"
            raise ValueError(msg)

    def _handle_missing(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing_count = int(frame[self.target_cols].isna().sum().sum())
        if missing_count == 0:
            return frame.reset_index(drop=True)
        if self.missing_policy == "error":
            msg = f"{self.mode} split target columns contain {missing_count} missing values"
            raise ValueError(msg)
        if self.missing_policy == "drop":
            return frame.dropna(subset=self.target_cols).reset_index(drop=True)
        if self.missing_policy == "forward_fill":
            filled = frame.copy()
            filled[self.target_cols] = filled[self.target_cols].ffill()
            remaining = int(filled[self.target_cols].isna().sum().sum())
            if remaining:
                msg = (
                    f"{self.mode} split forward_fill left missing target values "
                    "at the start of the split"
                )
                raise ValueError(msg)
            return filled.reset_index(drop=True)
        if self.missing_policy == "zero_fill":
            filled = frame.copy()
            filled[self.target_cols] = filled[self.target_cols].fillna(0.0)
            return filled.reset_index(drop=True)
        msg = f"Unsupported missing_policy: {self.missing_policy}"
        raise ValueError(msg)

    def _validate_split_min_length(
        self,
        *,
        row_count: int,
        min_length: int,
        selected_ratio: float,
    ) -> None:
        if selected_ratio == 0 and row_count == 0:
            return
        if row_count < min_length:
            msg = (
                f"{self.mode} split has {row_count} rows after applying "
                f"missing_policy={self.missing_policy!r}; requires at least {min_length} rows"
            )
            raise ValueError(msg)
