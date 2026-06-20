from __future__ import annotations

from ts_platform.config.loader import load_config
from ts_platform.runner.trainer import Trainer


def test_csv_training_smoke(tmp_path) -> None:
    config = load_config("configs/examples/csv_forecast.yaml")
    config.experiment.output_dir = tmp_path

    result = Trainer(config).run()

    assert result.checkpoint_path.exists()
    assert (result.run_dir / "results.json").exists()
    assert result.test_metrics["original"]
    assert "scaled" in result.test_metrics


def test_feature_aware_csv_example_config_runs(tmp_path) -> None:
    config = load_config("configs/examples/csv_feature_forecast.yaml")
    config.experiment.output_dir = tmp_path

    result = Trainer(config).run()

    assert result.checkpoint_path.exists()
    assert (result.run_dir / "results.json").exists()
    assert result.data_metadata == {
        "input_dim": 3,
        "target_dim": 1,
        "feature_dim": 2,
        "target_cols": ["value"],
        "feature_cols": ["temperature", "holiday"],
        "feature_aware": True,
    }
    assert result.test_metrics["original"]
