from __future__ import annotations

from ts_platform.config.schema import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PlatformConfig,
    ScalerConfig,
    TrainingConfig,
)
from ts_platform.runner.trainer import Trainer


def test_runner_smoke(tmp_path) -> None:
    config = PlatformConfig(
        experiment=ExperimentConfig(
            name="smoke",
            output_dir=tmp_path,
            seed=7,
            overwrite=True,
        ),
        data=DataConfig(
            name="synthetic",
            input_len=6,
            output_len=2,
            batch_size=8,
            scaler=ScalerConfig(name="standard"),
            params={"length": 80, "num_features": 1, "noise_std": 0.0},
        ),
        model=ModelConfig(name="linear"),
        training=TrainingConfig(epochs=1, learning_rate=0.01, device="cpu"),
    )

    result = Trainer(config).run()

    assert result.checkpoint_path.exists()
    assert (result.run_dir / "config_snapshot.yaml").exists()
    assert (result.run_dir / "environment.json").exists()
    assert (result.run_dir / "results.json").exists()
    assert set(result.test_metrics) == {"mae", "mse", "rmse", "mape", "wape"}
