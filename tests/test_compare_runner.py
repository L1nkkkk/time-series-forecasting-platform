from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from ts_platform.config.compare_schema import CompareConfig, CompareModelConfig
from ts_platform.config.schema import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ScalerConfig,
    TrainingConfig,
)
from ts_platform.runner.comparer import CompareRunner


def _compare_config(
    output_dir: Path,
    *,
    name: str = "compare_unit",
    models: list[CompareModelConfig] | None = None,
    continue_on_error: bool = True,
) -> CompareConfig:
    return CompareConfig(
        experiment=ExperimentConfig(name=name, output_dir=output_dir, overwrite=True, seed=11),
        data=DataConfig(
            name="synthetic",
            input_len=4,
            output_len=2,
            batch_size=4,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            scaler=ScalerConfig(name="standard"),
            params={"length": 48, "num_features": 1, "noise_std": 0.0},
        ),
        models=models
        or [
            CompareModelConfig(name="naive"),
            CompareModelConfig(name="moving_average", params={"window_size": 2}),
        ],
        training=TrainingConfig(epochs=1, learning_rate=0.01, device="cpu"),
        evaluation=EvaluationConfig(metrics=["mae", "mse"], include_scaled_metrics=False),
        primary_metric="mae",
        continue_on_error=continue_on_error,
    )


def test_compare_runner_writes_artifacts(tmp_path) -> None:
    result = CompareRunner(_compare_config(tmp_path)).run()

    assert result.compare_run_dir == tmp_path / "compare_unit" / "latest"
    assert result.leaderboard_json_path.exists()
    assert result.leaderboard_csv_path.exists()
    assert (result.compare_run_dir / "compare_config_snapshot.yaml").exists()
    assert (result.compare_run_dir / "environment.json").exists()
    assert len(result.rows) == 2


def test_compare_runner_creates_one_trainer_run_per_model(tmp_path) -> None:
    result = CompareRunner(_compare_config(tmp_path)).run()

    assert (result.compare_run_dir / "models" / "001_naive" / "latest" / "results.json").exists()
    assert (
        result.compare_run_dir / "models" / "002_moving_average" / "latest" / "results.json"
    ).exists()


def test_compare_runner_preserves_shared_data_config(tmp_path) -> None:
    result = CompareRunner(_compare_config(tmp_path)).run()
    snapshot_path = (
        result.compare_run_dir / "models" / "001_naive" / "latest" / "config_snapshot.yaml"
    )
    snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))

    assert snapshot["data"]["name"] == "synthetic"
    assert snapshot["data"]["input_len"] == 4
    assert snapshot["data"]["output_len"] == 2
    assert snapshot["data"]["params"] == {"length": 48, "num_features": 1, "noise_std": 0.0}


def test_compare_leaderboard_ranks_by_primary_metric(tmp_path) -> None:
    result = CompareRunner(_compare_config(tmp_path)).run()
    success_rows = [row for row in result.rows if row["status"] == "success"]

    values = [row["primary_metric_value"] for row in success_rows]
    ranks = [row["rank"] for row in success_rows]

    assert values == sorted(values)
    assert ranks == list(range(1, len(success_rows) + 1))


def test_compare_leaderboard_json_and_csv_have_same_rows(tmp_path) -> None:
    result = CompareRunner(_compare_config(tmp_path)).run()
    json_rows = json.loads(result.leaderboard_json_path.read_text(encoding="utf-8"))
    with result.leaderboard_csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))

    assert len(json_rows) == len(csv_rows) == len(result.rows)
    assert [row["model_alias"] for row in json_rows] == [row["model_alias"] for row in csv_rows]
    assert [row["status"] for row in json_rows] == [row["status"] for row in csv_rows]
    assert [str(row["rank"]) for row in json_rows] == [row["rank"] for row in csv_rows]


def test_compare_failed_model_recorded_when_continue_on_error(tmp_path) -> None:
    config = _compare_config(
        tmp_path,
        models=[CompareModelConfig(name="naive"), CompareModelConfig(name="unknown_model")],
    )

    result = CompareRunner(config).run()

    failed = [row for row in result.rows if row["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["rank"] is None
    assert "unknown registry item" in failed[0]["error"]
    assert any(row["status"] == "success" for row in result.rows)


def test_compare_stops_on_failure_when_continue_on_error_false(tmp_path) -> None:
    config = _compare_config(
        tmp_path,
        models=[CompareModelConfig(name="unknown_model"), CompareModelConfig(name="naive")],
        continue_on_error=False,
    )

    with pytest.raises(RuntimeError, match="compare model '001_unknown_model' failed"):
        CompareRunner(config).run()


def test_compare_rejects_unknown_model_clearly(tmp_path) -> None:
    config = _compare_config(
        tmp_path,
        models=[CompareModelConfig(name="unknown_one"), CompareModelConfig(name="unknown_two")],
    )

    result = CompareRunner(config).run()

    assert all(row["status"] == "failed" for row in result.rows)
    assert all("unknown registry item" in row["error"] for row in result.rows)
