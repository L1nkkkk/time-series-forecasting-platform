from __future__ import annotations

import json

import pytest

from tests.helpers import tiny_config, tiny_feature_config
from ts_platform.experiment.logger import setup_experiment_logger
from ts_platform.experiment.recorder import ExperimentRecorder
from ts_platform.runner.checkpoint import load_checkpoint
from ts_platform.runner.trainer import Trainer


def test_resume_training_from_checkpoint(tmp_path) -> None:
    first = Trainer(tiny_config(tmp_path, name="resume_first", epochs=1)).run()
    resume_config = tiny_config(
        tmp_path,
        name="resume_second",
        epochs=2,
        resume_from=first.checkpoint_path,
    )

    resumed = Trainer(resume_config).run()
    payload = load_checkpoint(resumed.checkpoint_path)
    results = json.loads((resumed.run_dir / "results.json").read_text(encoding="utf-8"))

    assert payload["epoch"] == 2
    assert results["resumed_from"] == str(first.checkpoint_path)
    assert resumed.test_metrics["original"]


def test_run_dir_unique_without_overwrite(tmp_path) -> None:
    first = ExperimentRecorder(tmp_path, "demo", overwrite=False).prepare()
    second = ExperimentRecorder(tmp_path, "demo", overwrite=False).prepare()

    assert first != second
    assert first.parent == second.parent == tmp_path / "demo"


def test_experiment_recorder_rejects_run_dir_escape(tmp_path) -> None:
    with pytest.raises(ValueError, match="experiment run_dir escapes root_dir"):
        ExperimentRecorder(tmp_path / "root", "../escape", overwrite=True)


def test_overwrite_does_not_leave_stale_artifacts(tmp_path) -> None:
    first = ExperimentRecorder(tmp_path, "demo", overwrite=True).prepare()
    stale = first / "stale.txt"
    stale.write_text("old", encoding="utf-8")

    second = ExperimentRecorder(tmp_path, "demo", overwrite=True).prepare()

    assert first == second == tmp_path / "demo" / "latest"
    assert not stale.exists()


def test_overwrite_closes_existing_run_logger(tmp_path) -> None:
    first = ExperimentRecorder(tmp_path, "demo", overwrite=True).prepare()
    logger = setup_experiment_logger(first)
    logger.info("old run")
    assert logger.handlers

    second = ExperimentRecorder(tmp_path, "demo", overwrite=True).prepare()

    assert first == second == tmp_path / "demo" / "latest"
    assert logger.handlers == []


def test_results_include_run_metadata(tmp_path) -> None:
    result = Trainer(tiny_config(tmp_path, name="metadata")).run()
    results = json.loads((result.run_dir / "results.json").read_text(encoding="utf-8"))

    assert results["run_id"] == result.run_id
    assert results["created_at"] == result.created_at
    assert results["experiment_name"] == "metadata"
    assert results["run_dir"] == str(result.run_dir)


def test_feature_aware_results_include_data_metadata(tmp_path) -> None:
    result = Trainer(tiny_feature_config(tmp_path, name="feature_results")).run()

    assert result.data_metadata == {
        "input_dim": 3,
        "target_dim": 1,
        "feature_dim": 2,
        "target_cols": ["value"],
        "feature_cols": ["temperature", "holiday"],
        "feature_aware": True,
    }


def test_target_only_results_include_data_metadata_or_remain_compatible(tmp_path) -> None:
    result = Trainer(tiny_config(tmp_path, name="target_results")).run()
    results = json.loads((result.run_dir / "results.json").read_text(encoding="utf-8"))

    assert results["data_metadata"] == {
        "input_dim": 1,
        "target_dim": 1,
        "feature_dim": 0,
        "target_cols": [],
        "feature_cols": [],
        "feature_aware": False,
    }
    assert results["test_metrics"] == result.test_metrics


def test_results_json_feature_aware_metadata(tmp_path) -> None:
    result = Trainer(tiny_feature_config(tmp_path, name="feature_results_json")).run()
    results = json.loads((result.run_dir / "results.json").read_text(encoding="utf-8"))

    assert results["data_metadata"]["input_dim"] == 3
    assert results["data_metadata"]["target_dim"] == 1
    assert results["data_metadata"]["feature_dim"] == 2
    assert results["data_metadata"]["target_cols"] == ["value"]
    assert results["data_metadata"]["feature_cols"] == ["temperature", "holiday"]
    assert results["data_metadata"]["feature_aware"] is True


def test_zero_validation_split(tmp_path) -> None:
    result = Trainer(tiny_config(tmp_path, name="zero_val", val_ratio=0.0)).run()
    results = json.loads((result.run_dir / "results.json").read_text(encoding="utf-8"))

    assert result.validation_metrics is None
    assert results["validation_metrics"] is None
    assert result.test_metrics["original"]
    assert all("validation_metrics" not in row for row in result.history)


def test_training_reproducible_with_same_seed(tmp_path) -> None:
    first = Trainer(tiny_config(tmp_path, name="repro_a", seed=123)).run()
    second = Trainer(tiny_config(tmp_path, name="repro_b", seed=123)).run()

    assert first.test_metrics["original"] == pytest.approx(second.test_metrics["original"])


def test_include_scaled_metrics_default(tmp_path) -> None:
    default_result = Trainer(tiny_config(tmp_path, name="scaled_default")).run()
    explicit_config = tiny_config(tmp_path, name="scaled_enabled")
    explicit_config.evaluation.include_scaled_metrics = True
    explicit_result = Trainer(explicit_config).run()

    assert "scaled" not in default_result.test_metrics
    assert "scaled" in explicit_result.test_metrics
