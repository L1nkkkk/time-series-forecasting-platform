from __future__ import annotations

import pytest

from tests.helpers import tiny_config
from ts_platform.api.services.experiment_store import (
    ExperimentArtifactNotFoundError,
    ExperimentStore,
    UnsafePathComponentError,
)
from ts_platform.config.compare_schema import CompareConfig, CompareModelConfig
from ts_platform.config.schema import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ScalerConfig,
    TrainingConfig,
)
from ts_platform.runner.comparer import CompareRunner
from ts_platform.runner.trainer import Trainer


def _compare_config(tmp_path, *, name: str = "store_compare") -> CompareConfig:
    return CompareConfig(
        experiment=ExperimentConfig(name=name, output_dir=tmp_path, overwrite=True, seed=11),
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
        models=[
            CompareModelConfig(name="naive"),
            CompareModelConfig(name="moving_average", params={"window_size": 2}),
        ],
        training=TrainingConfig(epochs=1, learning_rate=0.01, device="cpu"),
        evaluation=EvaluationConfig(metrics=["mae", "mse"], include_scaled_metrics=False),
        primary_metric="mae",
        continue_on_error=True,
    )


def test_experiment_store_rejects_unsafe_experiment_name(tmp_path) -> None:
    store = ExperimentStore(tmp_path)

    with pytest.raises(UnsafePathComponentError, match="experiment_name"):
        store.read_results("../escape", "latest")


def test_experiment_store_rejects_unsafe_run_id(tmp_path) -> None:
    store = ExperimentStore(tmp_path)

    with pytest.raises(UnsafePathComponentError, match="run_id"):
        store.read_results("safe_experiment", "../escape")


def test_experiment_store_prevents_path_escape(tmp_path) -> None:
    runs_root = tmp_path / "runs"
    experiment_dir = runs_root / "safe_experiment"
    outside_dir = tmp_path / "outside"
    experiment_dir.mkdir(parents=True)
    outside_dir.mkdir()
    (outside_dir / "results.json").write_text("{}", encoding="utf-8")
    link_path = experiment_dir / "linked"
    try:
        link_path.symlink_to(outside_dir, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable in this environment")

    store = ExperimentStore(runs_root)

    with pytest.raises(UnsafePathComponentError, match="escapes runs root"):
        store.read_results("safe_experiment", "linked")


def test_experiment_store_reads_train_results(tmp_path) -> None:
    result = Trainer(tiny_config(tmp_path, name="store_train")).run()
    store = ExperimentStore(tmp_path)

    payload = store.read_results("store_train", "latest")

    assert payload["run_id"] == result.run_id
    assert payload["experiment_name"] == "store_train"
    assert payload["test_metrics"]["original"]


def test_experiment_store_reads_train_artifacts(tmp_path) -> None:
    result = Trainer(tiny_config(tmp_path, name="store_train_artifacts")).run()
    store = ExperimentStore(tmp_path)

    payload = store.read_artifacts("store_train_artifacts", "latest")

    assert payload["run_type"] == "train"
    assert payload["run_id"] == result.run_id
    assert any(artifact["name"] == "results" for artifact in payload["artifacts"])


def test_experiment_store_reads_compare_results(tmp_path) -> None:
    result = CompareRunner(_compare_config(tmp_path)).run()
    store = ExperimentStore(tmp_path)

    payload = store.read_results("store_compare", "latest")

    assert payload["run_type"] == "compare"
    assert payload["compare_run_id"] == result.compare_run_id
    assert payload["success_count"] == 2
    assert payload["failed_count"] == 0


def test_experiment_store_reads_compare_artifacts(tmp_path) -> None:
    result = CompareRunner(_compare_config(tmp_path, name="store_compare_artifacts")).run()
    store = ExperimentStore(tmp_path)

    payload = store.read_artifacts("store_compare_artifacts", result.compare_run_id)

    assert payload["run_type"] == "compare"
    assert payload["compare_run_id"] == result.compare_run_id
    assert any(artifact["name"] == "leaderboard_json" for artifact in payload["artifacts"])


def test_experiment_store_reads_compare_leaderboard(tmp_path) -> None:
    CompareRunner(_compare_config(tmp_path)).run()
    store = ExperimentStore(tmp_path)

    rows = store.read_leaderboard("store_compare", "latest")

    assert len(rows) == 2
    assert all(isinstance(row["model_params"], dict) for row in rows)


def test_experiment_store_missing_results_is_clear_error(tmp_path) -> None:
    store = ExperimentStore(tmp_path)

    with pytest.raises(ExperimentArtifactNotFoundError, match="does not exist"):
        store.read_results("missing", "latest")


def test_experiment_store_artifacts_missing_is_clear_error(tmp_path) -> None:
    run_dir = tmp_path / "missing_artifacts" / "latest"
    run_dir.mkdir(parents=True)
    (run_dir / "results.json").write_text(
        '{"experiment_name": "missing_artifacts", "run_id": "latest"}',
        encoding="utf-8",
    )
    store = ExperimentStore(tmp_path)

    with pytest.raises(ExperimentArtifactNotFoundError, match="artifacts.json"):
        store.read_artifacts("missing_artifacts", "latest")


def test_experiment_store_artifacts_rejects_unsafe_path_component(tmp_path) -> None:
    store = ExperimentStore(tmp_path)

    with pytest.raises(UnsafePathComponentError, match="experiment_name"):
        store.read_artifacts("bad name", "latest")
