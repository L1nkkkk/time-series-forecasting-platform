from __future__ import annotations

import json

import pytest
import torch

from tests.helpers import tiny_config, tiny_feature_config
from ts_platform.data.transforms import FeatureAwareScalerBundle
from ts_platform.runner.model_export import (
    load_model_export,
    restore_model_from_export,
    restore_scalers_from_export,
)
from ts_platform.runner.predictor import predict_from_model_export
from ts_platform.runner.trainer import Trainer


def test_training_writes_model_export(tmp_path) -> None:
    config = tiny_config(tmp_path, name="model_export")
    result = Trainer(config).run()

    payload = result.to_dict()

    assert result.model_export_path.exists()
    assert result.model_export_metadata_path.exists()
    assert payload["model_export_path"] == str(result.model_export_path)
    assert payload["model_export_metadata_path"] == str(result.model_export_metadata_path)


def test_model_export_restores_model_for_inference(tmp_path) -> None:
    config = tiny_config(tmp_path, name="model_export_restore")
    result = Trainer(config).run()

    payload = load_model_export(result.model_export_path)
    model = restore_model_from_export(payload)
    scaler = restore_scalers_from_export(payload)

    x = torch.zeros(2, config.data.input_len, 1)

    assert "optimizer" not in payload
    assert model(x).shape == (2, config.data.output_len, 1)
    assert scaler.fitted


def test_feature_aware_model_export_restores_scaler_bundle(tmp_path) -> None:
    result = Trainer(tiny_feature_config(tmp_path, name="model_export_feature")).run()

    payload = load_model_export(result.model_export_path)
    model = restore_model_from_export(payload)
    scaler = restore_scalers_from_export(payload)

    assert payload["data"]["feature_aware"] is True
    assert model(torch.zeros(2, 4, 3)).shape == (2, 2, 1)
    assert isinstance(scaler, FeatureAwareScalerBundle)


def test_model_export_predicts_original_scale_values(tmp_path) -> None:
    config = tiny_config(tmp_path, name="model_export_predict")
    result = Trainer(config).run()

    payload = predict_from_model_export(
        result.model_export_path,
        values=[[[0.0] for _ in range(config.data.input_len)]],
    )

    assert payload["format"] == "ts_platform_prediction"
    assert payload["model"]["output_len"] == config.data.output_len
    assert len(payload["prediction"]) == 1
    assert len(payload["prediction"][0]) == config.data.output_len
    assert len(payload["prediction"][0][0]) == 1


def test_feature_aware_model_export_predicts_from_combined_input(tmp_path) -> None:
    config = tiny_feature_config(tmp_path, name="model_export_feature_predict")
    result = Trainer(config).run()

    payload = predict_from_model_export(
        result.model_export_path,
        values=[[[0.0, 20.0, 0.0] for _ in range(config.data.input_len)]],
    )

    assert payload["data"]["feature_aware"] is True
    assert payload["model"]["input_dim"] == 3
    assert len(payload["prediction"][0]) == config.data.output_len


def test_model_export_prediction_rejects_wrong_shape(tmp_path) -> None:
    config = tiny_config(tmp_path, name="model_export_bad_shape")
    result = Trainer(config).run()

    with pytest.raises(ValueError, match="input length"):
        predict_from_model_export(result.model_export_path, values=[[[0.0]]])


def test_model_export_metadata_omits_weight_tensors(tmp_path) -> None:
    result = Trainer(tiny_config(tmp_path, name="model_export_metadata")).run()

    metadata = json.loads(result.model_export_metadata_path.read_text(encoding="utf-8"))

    assert metadata["format"] == "ts_platform_model_export"
    assert metadata["model"]["name"] == "linear"
    assert "state_dict" not in metadata["model"]
    assert metadata["target_scaler"]["fitted"] is True
