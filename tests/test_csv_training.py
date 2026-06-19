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
