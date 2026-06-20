from __future__ import annotations

import pytest
import torch

from tests.helpers import tiny_config, tiny_feature_config
from ts_platform.data.transforms import FeatureAwareScalerBundle
from ts_platform.models.registry import build_model
from ts_platform.runner.checkpoint import (
    load_checkpoint,
    restore_model_from_checkpoint,
    restore_scaler_from_checkpoint,
    restore_scalers_from_checkpoint,
    save_checkpoint,
    validate_checkpoint_for_training,
)
from ts_platform.runner.trainer import Trainer
from ts_platform.scaler.registry import build_scaler


def _feature_checkpoint(tmp_path, *, name: str = "feature_checkpoint"):
    result = Trainer(tiny_feature_config(tmp_path, name=name)).run()
    return result, load_checkpoint(result.checkpoint_path)


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

    assert payload["schema_version"] == 2
    assert payload["epoch"] == 1
    assert payload["config"]["experiment"]["name"] == config.experiment.name
    assert payload["model"]["name"] == "linear"
    assert payload["model"]["input_dim"] == 1
    assert payload["model"]["target_dim"] == 1
    assert payload["data"]["input_dim"] == 1
    assert payload["data"]["target_dim"] == 1
    assert payload["scaler"]["name"] == "standard"
    assert payload["target_scaler"]["name"] == "standard"
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


def test_checkpoint_v2_saves_input_target_dimensions(tmp_path) -> None:
    _, payload = _feature_checkpoint(tmp_path)

    assert payload["schema_version"] == 2
    assert payload["model"]["input_dim"] == 3
    assert payload["model"]["target_dim"] == 1
    assert payload["model"]["num_features"] == 1
    assert payload["data"]["input_dim"] == 3
    assert payload["data"]["target_dim"] == 1
    assert payload["data"]["feature_dim"] == 2


def test_checkpoint_v2_saves_target_and_feature_columns(tmp_path) -> None:
    _, payload = _feature_checkpoint(tmp_path)

    assert payload["data"]["target_cols"] == ["value"]
    assert payload["data"]["feature_cols"] == ["temperature", "holiday"]


def test_checkpoint_v2_saves_target_and_feature_scalers(tmp_path) -> None:
    _, payload = _feature_checkpoint(tmp_path)

    assert payload["target_scaler"]["name"] == "standard"
    assert payload["feature_scaler"]["name"] == "standard"
    assert "state" in payload["target_scaler"]
    assert "state" in payload["feature_scaler"]
    assert "scaler" not in payload


def test_restore_model_from_v2_checkpoint_feature_aware(tmp_path) -> None:
    _, payload = _feature_checkpoint(tmp_path)

    model = restore_model_from_checkpoint(payload)

    assert model.input_dim == 3
    assert model.target_dim == 1
    assert model(torch.zeros(2, 4, 3)).shape == (2, 2, 1)


def test_restore_scalers_from_v2_checkpoint_feature_aware(tmp_path) -> None:
    _, payload = _feature_checkpoint(tmp_path)

    scaler = restore_scalers_from_checkpoint(payload)

    assert isinstance(scaler, FeatureAwareScalerBundle)
    assert scaler.features is not None
    assert scaler.target.state_dict()["mean"].shape[-1] == 1
    assert scaler.features.state_dict()["mean"].shape[-1] == 2


def test_validate_checkpoint_rejects_feature_column_mismatch(tmp_path) -> None:
    _, payload = _feature_checkpoint(tmp_path)
    config = tiny_feature_config(tmp_path, feature_cols=["holiday", "temperature"])

    with pytest.raises(ValueError, match="feature_cols"):
        validate_checkpoint_for_training(
            payload,
            config,
            input_dim=3,
            target_dim=1,
            target_cols=["value"],
            feature_cols=["holiday", "temperature"],
        )


def test_validate_checkpoint_rejects_input_dim_mismatch(tmp_path) -> None:
    _, payload = _feature_checkpoint(tmp_path)
    config = tiny_feature_config(tmp_path, feature_cols=["temperature"])

    with pytest.raises(ValueError, match="input_dim"):
        validate_checkpoint_for_training(
            payload,
            config,
            input_dim=2,
            target_dim=1,
            target_cols=["value"],
            feature_cols=["temperature"],
        )


def test_v1_checkpoint_target_only_compatibility(tmp_path) -> None:
    config = tiny_config(tmp_path, name="v1_compat")
    model = build_model(
        config.model,
        input_len=config.data.input_len,
        output_len=config.data.output_len,
        num_features=1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    scaler = build_scaler(config.data.scaler).fit(torch.arange(10, dtype=torch.float32).view(10, 1))
    path = tmp_path / "v1_checkpoint.pt"
    torch.save(
        {
            "schema_version": 1,
            "epoch": 1,
            "config": config.model_dump(mode="json"),
            "model": {
                "name": config.model.name,
                "params": config.model.params,
                "input_len": model.input_len,
                "output_len": model.output_len,
                "num_features": model.num_features,
                "state_dict": model.state_dict(),
            },
            "optimizer": {
                "name": config.training.optimizer,
                "state_dict": optimizer.state_dict(),
            },
            "scaler": {
                "name": config.data.scaler.name,
                "params": config.data.scaler.params,
                "state": scaler.state_dict(),
            },
            "metrics": {},
            "environment": {},
        },
        path,
    )

    payload = load_checkpoint(path)
    validate_checkpoint_for_training(payload, config, num_features=1)
    restored_model = restore_model_from_checkpoint(payload, config.model)
    restored_scaler = restore_scaler_from_checkpoint(payload, config.data.scaler)

    assert payload["schema_version"] == 1
    assert restored_model(torch.zeros(2, config.data.input_len, 1)).shape == (
        2,
        config.data.output_len,
        1,
    )
    assert restored_scaler.fitted


def test_feature_aware_checkpoint_resume(tmp_path) -> None:
    first = Trainer(tiny_feature_config(tmp_path, name="feature_resume_first", epochs=1)).run()
    resume_config = tiny_feature_config(
        tmp_path,
        name="feature_resume_second",
        epochs=2,
        resume_from=first.checkpoint_path,
    )

    resumed = Trainer(resume_config).run()
    payload = load_checkpoint(resumed.checkpoint_path)

    assert payload["epoch"] == 2
    assert payload["model"]["input_dim"] == 3
    assert payload["data"]["feature_cols"] == ["temperature", "holiday"]


def test_feature_aware_resume_rejects_feature_cols_mismatch(tmp_path) -> None:
    first = Trainer(tiny_feature_config(tmp_path, name="feature_resume_cols_a")).run()
    resume_config = tiny_feature_config(
        tmp_path,
        name="feature_resume_cols_b",
        epochs=2,
        resume_from=first.checkpoint_path,
        feature_cols=["holiday", "temperature"],
    )

    with pytest.raises(ValueError, match="feature_cols"):
        Trainer(resume_config).run()


def test_feature_aware_resume_rejects_target_cols_mismatch(tmp_path) -> None:
    first = Trainer(tiny_feature_config(tmp_path, name="feature_resume_target_a")).run()
    resume_config = tiny_feature_config(
        tmp_path,
        name="feature_resume_target_b",
        epochs=2,
        resume_from=first.checkpoint_path,
        target_cols=["temperature"],
        feature_cols=["value", "holiday"],
    )

    with pytest.raises(ValueError, match="target_cols"):
        Trainer(resume_config).run()
