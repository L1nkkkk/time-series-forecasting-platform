from __future__ import annotations

import pytest

from tests.helpers import tiny_config
from ts_platform.config.schema import ModelConfig
from ts_platform.runner.trainer import Trainer


def test_runner_smoke(tmp_path) -> None:
    config = tiny_config(tmp_path, name="smoke")

    result = Trainer(config).run()

    assert result.checkpoint_path.exists()
    assert (result.run_dir / "config_snapshot.yaml").exists()
    assert (result.run_dir / "environment.json").exists()
    assert (result.run_dir / "results.json").exists()
    assert set(result.test_metrics["original"]) == {"mae", "mse", "rmse", "mape", "wape"}
    assert "scaled" not in result.test_metrics


@pytest.mark.parametrize(
    ("model_name", "params"),
    (
        ("rnn", {"hidden_size": 8}),
        ("gru", {"hidden_size": 8}),
        ("lstm", {"hidden_size": 8}),
        ("tcn", {"hidden_channels": 8, "num_layers": 2}),
    ),
)
def test_new_model_training_smoke(tmp_path, model_name: str, params: dict[str, object]) -> None:
    config = tiny_config(tmp_path, name=f"{model_name}_smoke")
    config = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "input_len": 4,
                    "output_len": 2,
                    "batch_size": 4,
                    "params": {"length": 48, "num_features": 1, "noise_std": 0.0},
                }
            ),
            "model": ModelConfig(name=model_name, params=params),
        }
    )

    result = Trainer(config).run()

    assert result.checkpoint_path.exists()
    assert (result.run_dir / "results.json").exists()
    assert "original" in result.test_metrics
