"""Validated parameter model for CSV datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

MissingPolicy = Literal["error", "drop", "forward_fill", "zero_fill"]


class CSVDatasetParams(BaseModel):
    """Strictly validated CSV dataset parameters."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    timestamp_col: str | None = None
    target_cols: list[str]
    missing_policy: MissingPolicy = "error"
    sort_by_time: bool = True
    feature_cols: list[str] | None = None

    @field_validator("path", mode="before")
    @classmethod
    def validate_path(cls, value: object) -> object:
        """Require a value that can be interpreted as a local filesystem path."""

        if isinstance(value, str | Path):
            return value
        msg = "path must be a filesystem path"
        raise ValueError(msg)

    @field_validator("timestamp_col", mode="before")
    @classmethod
    def validate_timestamp_col(cls, value: object) -> object:
        """Require timestamp_col to be a string or null."""

        if value is None or isinstance(value, str):
            return value
        msg = "timestamp_col must be a string or None"
        raise ValueError(msg)

    @field_validator("target_cols", mode="before")
    @classmethod
    def validate_target_cols(cls, value: object) -> object:
        """Require target_cols to be a non-empty list of strings."""

        if isinstance(value, str):
            msg = "target_cols must be a non-empty list of strings"
            raise ValueError(msg)
        if not isinstance(value, list):
            msg = "target_cols must be a non-empty list of strings"
            raise ValueError(msg)
        if not value:
            msg = "target_cols must be a non-empty list of strings"
            raise ValueError(msg)
        if not all(isinstance(item, str) and item for item in value):
            msg = "target_cols must be a non-empty list of strings"
            raise ValueError(msg)
        return value

    @field_validator("sort_by_time", mode="before")
    @classmethod
    def validate_sort_by_time(cls, value: object) -> object:
        """Require sort_by_time to be an actual boolean."""

        if isinstance(value, bool):
            return value
        msg = "sort_by_time must be a boolean"
        raise ValueError(msg)

    @field_validator("feature_cols", mode="before")
    @classmethod
    def validate_feature_cols_shape(cls, value: object) -> object:
        """Require feature_cols to be absent, null, or a list of strings."""

        if value is None:
            return value
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            msg = "feature_cols must be a list of strings or None"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def reject_exogenous_features(self) -> CSVDatasetParams:
        """Reject exogenous features until model/data support is implemented."""

        if self.feature_cols:
            msg = "exogenous feature_cols are not supported yet"
            raise ValueError(msg)
        return self
