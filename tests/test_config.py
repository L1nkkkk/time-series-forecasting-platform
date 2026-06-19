from __future__ import annotations

import pytest
from pydantic import ValidationError

from ts_platform.config.loader import load_config
from ts_platform.config.schema import PlatformConfig


def test_load_valid_config_defaults() -> None:
    config = load_config("configs/examples/simple_forecast.yaml")

    assert isinstance(config, PlatformConfig)
    assert config.experiment.name == "simple_forecast"
    assert config.training.epochs == 2
    assert config.evaluation.metrics == ["mae", "mse", "rmse", "mape", "wape"]


def test_missing_required_field_has_clear_error(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
experiment:
  name: bad
data:
  name: synthetic
  input_len: 4
model:
  name: linear
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="output_len"):
        load_config(path)


def test_split_ratio_validation(tmp_path) -> None:
    path = tmp_path / "bad_ratio.yaml"
    path.write_text(
        """
experiment:
  name: bad
data:
  name: synthetic
  input_len: 4
  output_len: 2
  train_ratio: 0.5
  val_ratio: 0.2
  test_ratio: 0.2
model:
  name: linear
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="must equal 1.0"):
        load_config(path)


def test_experiment_name_rejects_path_separator(tmp_path) -> None:
    path = tmp_path / "bad_name.yaml"
    path.write_text(
        """
experiment:
  name: bad/name
data:
  name: synthetic
  input_len: 4
  output_len: 2
model:
  name: linear
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="experiment.name must be a safe path component"):
        load_config(path)


def test_experiment_name_rejects_parent_reference(tmp_path) -> None:
    path = tmp_path / "bad_parent.yaml"
    path.write_text(
        """
experiment:
  name: bad..name
data:
  name: synthetic
  input_len: 4
  output_len: 2
model:
  name: linear
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="experiment.name must be a safe path component"):
        load_config(path)


def test_experiment_name_rejects_absolute_path(tmp_path) -> None:
    path = tmp_path / "bad_absolute.yaml"
    path.write_text(
        """
experiment:
  name: /tmp/bad
data:
  name: synthetic
  input_len: 4
  output_len: 2
model:
  name: linear
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="experiment.name must be a safe path component"):
        load_config(path)
