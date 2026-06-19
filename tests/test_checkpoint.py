from __future__ import annotations

import pytest
import torch

from tests.helpers import tiny_config
from ts_platform.models.registry import build_model
from ts_platform.runner.checkpoint import (
    load_checkpoint,
    restore_model_from_checkpoint,
    restore_scaler_from_checkpoint,
    save_checkpoint,
)
from ts_platform.runner.trainer import Trainer
from ts_platform.scaler.registry import build_scaler


def test_checkpoint_schema(tmp_path) -> None:
    config = tiny_config(tmp_path)
    dataset_num_features = 1
    built_model = build_model(
        config.model,
        input_len=config.data.input_len,
        output_len=config.data.output_len,
        num_features=dataset_num_features,
    )
    optimizer = torch.optim.Adam(built_model.parameters(), lr=config.training.learning_rate)
    scaler = build_scaler(config.data.scaler).fit(
        torch.arange(10, dtype=torch.float32).reshape(10, 1)
    )
    path = save_checkpoint(
        tmp_path / "checkpoint.pt",
        model=built_model,
        optimizer=optimizer,
        epoch=1,
        metrics={"test_metrics": {"original": {"mae": 1.0}}},
        config=config,
        scaler=scaler,
        environment={"python": "test"},
    )

    payload = load_checkpoint(path)

    assert payload["schema_version"] == 1
    assert payload["epoch"] == 1
    assert payload["config"]["experiment"]["name"] == config.experiment.name
    assert payload["model"]["name"] == "linear"
    assert payload["scaler"]["name"] == "standard"
    assert payload["optimizer"]["name"] == "adam"
    assert payload["environment"] == {"python": "test"}


def test_checkpoint_load_rejects_unknown_schema(tmp_path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"schema_version": 999}, path)

    with pytest.raises(ValueError, match="Unsupported checkpoint schema_version"):
        load_checkpoint(path)


def test_checkpoint_restore_model_and_scaler(tmp_path) -> None:
    config = tiny_config(tmp_path)
    result = Trainer(config).run()
    payload = load_checkpoint(result.checkpoint_path)

    model = restore_model_from_checkpoint(payload, config.model)
    scaler = restore_scaler_from_checkpoint(payload, config.data.scaler)
    x = torch.zeros(2, config.data.input_len, 1)
    values = torch.randn(5, 1)

    assert model(x).shape == (2, config.data.output_len, 1)
    assert torch.allclose(scaler.inverse_transform(scaler.transform(values)), values, atol=1e-5)
