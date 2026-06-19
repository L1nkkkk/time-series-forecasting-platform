from __future__ import annotations

from tests.helpers import tiny_config
from ts_platform.runner.trainer import Trainer


def test_runner_smoke(tmp_path) -> None:
    config = tiny_config(tmp_path, name="smoke")

    result = Trainer(config).run()

    assert result.checkpoint_path.exists()
    assert (result.run_dir / "config_snapshot.yaml").exists()
    assert (result.run_dir / "environment.json").exists()
    assert (result.run_dir / "results.json").exists()
    assert set(result.test_metrics["original"]) == {"mae", "mse", "rmse", "mape", "wape"}
    assert "scaled" in result.test_metrics
