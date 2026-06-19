from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from ts_platform.config.compare_loader import load_compare_config


def _compare_payload() -> dict[str, object]:
    return {
        "experiment": {"name": "compare_test", "output_dir": "runs", "overwrite": True},
        "data": {
            "name": "synthetic",
            "input_len": 4,
            "output_len": 2,
            "batch_size": 4,
            "params": {"length": 40, "num_features": 1},
        },
        "models": [{"name": "naive"}, {"name": "linear", "alias": "linear_a"}],
        "training": {"epochs": 1},
        "evaluation": {"metrics": ["mae", "mse"], "include_scaled_metrics": False},
    }


def _write_compare_config(tmp_path, payload: dict[str, object]):
    path = tmp_path / "compare.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_load_valid_compare_config(tmp_path) -> None:
    path = _write_compare_config(tmp_path, _compare_payload())

    config = load_compare_config(path)

    assert config.experiment.name == "compare_test"
    assert len(config.models) == 2
    assert config.primary_metric == "mae"
    assert config.continue_on_error is True


def test_compare_config_requires_at_least_two_models(tmp_path) -> None:
    payload = _compare_payload()
    payload["models"] = [{"name": "naive"}]
    path = _write_compare_config(tmp_path, payload)

    with pytest.raises(ValidationError, match="at least 2"):
        load_compare_config(path)


def test_compare_config_primary_metric_must_be_in_metrics(tmp_path) -> None:
    payload = _compare_payload()
    payload["primary_metric"] = "rmse"
    path = _write_compare_config(tmp_path, payload)

    with pytest.raises(ValidationError, match="primary_metric must be one of evaluation.metrics"):
        load_compare_config(path)


def test_compare_model_alias_must_be_safe(tmp_path) -> None:
    payload = _compare_payload()
    payload["models"] = [{"name": "naive"}, {"name": "linear", "alias": "../linear"}]
    path = _write_compare_config(tmp_path, payload)

    with pytest.raises(ValidationError, match="model alias must be a safe path component"):
        load_compare_config(path)


def test_compare_config_rejects_extra_fields(tmp_path) -> None:
    payload = _compare_payload()
    payload["unexpected"] = True
    path = _write_compare_config(tmp_path, payload)

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_compare_config(path)
