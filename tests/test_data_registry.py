from __future__ import annotations

from ts_platform.config.loader import load_config
from ts_platform.data.loaders import build_dataset
from ts_platform.data.registry import DATASET_REGISTRY


def test_dataset_registry_contains_synthetic() -> None:
    assert "synthetic" in DATASET_REGISTRY.names()
    assert "csv" in DATASET_REGISTRY.names()


def test_synthetic_dataset_split_and_batch_shape() -> None:
    config = load_config("configs/examples/simple_forecast.yaml")
    dataset = build_dataset(config.data, "train", config.experiment.seed)
    sample = dataset[0]

    assert len(dataset) > 0
    assert sample["x"].shape == (config.data.input_len, 2)
    assert sample["y"].shape == (config.data.output_len, 2)
    assert dataset.scaler_fit_values().shape[-1] == 2
