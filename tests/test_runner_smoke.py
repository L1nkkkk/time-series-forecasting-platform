from __future__ import annotations

import json

import pytest

from tests.helpers import tiny_config
from ts_platform.config.schema import EarlyStoppingConfig, LRSchedulerConfig, ModelConfig
from ts_platform.runner import trainer as trainer_module
from ts_platform.runner.checkpoint import load_checkpoint
from ts_platform.runner.trainer import Trainer


def test_runner_smoke(tmp_path) -> None:
    config = tiny_config(tmp_path, name="smoke")

    result = Trainer(config).run()

    assert result.checkpoint_path.exists()
    assert (result.run_dir / "config_snapshot.yaml").exists()
    assert (result.run_dir / "environment.json").exists()
    assert (result.run_dir / "results.json").exists()
    assert (result.run_dir / "forecast_samples.json").exists()
    assert result.forecast_samples["sample_count"] > 0
    assert result.forecast_samples["samples"][0]["actual"]
    assert result.forecast_samples["samples"][0]["predicted"]
    assert set(result.test_metrics["original"]) == {"mae", "mse", "rmse", "mape", "wape"}
    assert "scaled" not in result.test_metrics


@pytest.mark.parametrize(
    ("model_name", "params"),
    (
        ("rnn", {"hidden_size": 8}),
        ("gru", {"hidden_size": 8}),
        ("lstm", {"hidden_size": 8}),
        ("tcn", {"hidden_channels": 8, "num_layers": 2}),
        ("transformer", {"d_model": 16, "num_heads": 4, "num_layers": 1, "dim_feedforward": 32}),
        ("nbeats", {"hidden_size": 16, "num_blocks": 2, "num_layers": 2}),
        ("dlinear", {"kernel_size": 3}),
        ("nlinear", {}),
        ("patchtst", {"patch_len": 2, "stride": 1, "d_model": 16}),
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


def test_training_controls_write_best_checkpoint_and_progress(tmp_path) -> None:
    config = tiny_config(tmp_path, name="training_controls", epochs=4)
    training = config.training.model_copy(
        update={
            "early_stopping": EarlyStoppingConfig(
                enabled=True,
                patience=1,
                min_delta=1_000_000.0,
                mode="min",
            ),
            "lr_scheduler": LRSchedulerConfig(name="step", step_size=1, gamma=0.5),
            "gradient_clip_norm": 1.0,
            "best_checkpoint_metric": "mae",
        }
    )
    config = config.model_copy(update={"training": training})

    result = Trainer(config).run()
    progress = json.loads((result.run_dir / "progress.json").read_text(encoding="utf-8"))
    checkpoint = load_checkpoint(result.checkpoint_path)

    assert len(result.history) == 2
    assert result.best_epoch == 1
    assert result.best_checkpoint_path is not None
    assert result.best_checkpoint_path.exists()
    assert result.history[0]["learning_rate"] == 0.01
    assert result.history[1]["learning_rate"] == 0.005
    assert progress["early_stopped"] is True
    assert progress["best_metric"]["name"] == "mae"
    assert checkpoint["epoch"] == result.best_epoch


def test_target_duration_pacing_updates_progress_without_real_sleep(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_time = [0.0]
    sleeps: list[float] = []

    def fake_monotonic() -> float:
        return current_time[0]

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        current_time[0] += seconds

    monkeypatch.setattr(trainer_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(trainer_module.time, "sleep", fake_sleep)
    config = tiny_config(tmp_path, name="paced_training", epochs=2)
    config = config.model_copy(
        update={"training": config.training.model_copy(update={"target_duration_minutes": 0.02})}
    )

    result = Trainer(config).run()
    progress = json.loads((result.run_dir / "progress.json").read_text(encoding="utf-8"))

    assert sleeps
    assert sum(sleeps) == pytest.approx(1.2)
    assert progress["target_duration_minutes"] == 0.02
    assert progress["target_duration_seconds"] == pytest.approx(1.2)
    assert progress["target_epoch_seconds"] == pytest.approx(0.6)
    assert progress["elapsed_seconds"] == pytest.approx(1.2)
    assert progress["estimated_remaining_seconds"] == 0
    assert progress["progress_percent"] == 100
