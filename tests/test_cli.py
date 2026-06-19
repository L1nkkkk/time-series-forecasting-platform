from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from tests.helpers import tiny_config
from ts_platform.api.services.experiment_store import UnsafePathComponentError
from ts_platform.cli.main import main


def test_cli_train_runs(tmp_path, capsys) -> None:
    config = tiny_config(tmp_path, name="cli")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_path)])
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    assert exit_code == 0
    assert payload["run_dir"]
    assert payload["checkpoint_path"]
    assert payload["test_metrics"]["original"]
    assert Path(payload["checkpoint_path"]).exists()
    assert (tmp_path / "cli" / "latest" / "results.json").exists()


def test_cli_list_datasets(capsys) -> None:
    exit_code = main(["list-datasets"])
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    assert exit_code == 0
    assert {"synthetic", "csv"}.issubset(set(payload["names"]))
    assert any(item["name"] == "synthetic" for item in payload["datasets"])


def test_cli_list_datasets_with_catalog(capsys) -> None:
    exit_code = main(["list-datasets", "--catalog", "configs/datasets/local_csv.yaml"])
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    assert exit_code == 0
    assert any(item["name"] == "tiny_csv" for item in payload["datasets"])


def test_cli_list_models(capsys) -> None:
    exit_code = main(["list-models"])
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    assert exit_code == 0
    assert {"naive", "linear", "mlp"}.issubset(set(payload["models"]))


def test_cli_still_allows_custom_output_dir(tmp_path, capsys) -> None:
    custom_output_dir = tmp_path / "custom_runs"
    config = tiny_config(custom_output_dir, name="cli_custom")
    config_path = tmp_path / "custom_config.yaml"
    config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_path)])
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    assert exit_code == 0
    assert Path(payload["run_dir"]).is_relative_to(custom_output_dir)
    assert (custom_output_dir / "cli_custom" / "latest" / "results.json").exists()


def test_cli_train_rejects_unsafe_experiment_name(tmp_path) -> None:
    config = tiny_config(tmp_path, name="cli_safe")
    payload = config.model_dump(mode="json")
    payload["experiment"]["name"] = "../escape"
    config_path = tmp_path / "bad_config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValidationError, match="experiment.name must be a safe path component"):
        main(["train", "--config", str(config_path)])


def _write_compare_cli_config(tmp_path, *, bad: bool = False) -> Path:
    payload = {
        "experiment": {"name": "cli_compare", "output_dir": str(tmp_path), "overwrite": True},
        "data": {
            "name": "synthetic",
            "input_len": 4,
            "output_len": 2,
            "batch_size": 4,
            "params": {"length": 48, "num_features": 1, "noise_std": 0.0},
        },
        "models": [{"name": "naive"}, {"name": "moving_average", "params": {"window_size": 2}}],
        "training": {"epochs": 1, "learning_rate": 0.01, "device": "cpu"},
        "evaluation": {"metrics": ["mae", "mse"], "include_scaled_metrics": False},
        "primary_metric": "mae",
    }
    if bad:
        payload["models"] = [{"name": "naive"}]
    config_path = tmp_path / ("bad_compare.yaml" if bad else "compare.yaml")
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def test_cli_compare_runs(tmp_path, capsys) -> None:
    config_path = _write_compare_cli_config(tmp_path)

    exit_code = main(["compare", "--config", str(config_path)])
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    assert exit_code == 0
    assert len(payload["rows"]) == 2
    assert Path(payload["leaderboard_json_path"]).exists()


def test_cli_compare_outputs_leaderboard_paths(tmp_path, capsys) -> None:
    config_path = _write_compare_cli_config(tmp_path)

    exit_code = main(["compare", "--config", str(config_path)])
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    assert exit_code == 0
    assert Path(payload["leaderboard_json_path"]).is_file()
    assert Path(payload["leaderboard_csv_path"]).is_file()
    assert Path(payload["compare_run_dir"]).is_dir()


def test_cli_compare_rejects_bad_config(tmp_path) -> None:
    config_path = _write_compare_cli_config(tmp_path, bad=True)

    with pytest.raises(ValidationError, match="at least 2"):
        main(["compare", "--config", str(config_path)])


def test_cli_show_results(tmp_path, capsys) -> None:
    config = tiny_config(tmp_path, name="cli_show")
    config_path = tmp_path / "show_config.yaml"
    config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    main(["train", "--config", str(config_path)])
    capsys.readouterr()

    exit_code = main(
        [
            "show-results",
            "--experiment",
            "cli_show",
            "--run",
            "latest",
            "--runs-root",
            str(tmp_path),
        ]
    )
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    assert exit_code == 0
    assert payload["experiment_name"] == "cli_show"
    assert payload["test_metrics"]["original"]


def test_cli_show_leaderboard(tmp_path, capsys) -> None:
    config_path = _write_compare_cli_config(tmp_path)
    main(["compare", "--config", str(config_path)])
    capsys.readouterr()

    exit_code = main(
        [
            "show-leaderboard",
            "--experiment",
            "cli_compare",
            "--run",
            "latest",
            "--runs-root",
            str(tmp_path),
        ]
    )
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    assert exit_code == 0
    assert len(payload) == 2
    assert all(isinstance(row["model_params"], dict) for row in payload)


def test_cli_show_artifacts(tmp_path, capsys) -> None:
    config_path = _write_compare_cli_config(tmp_path)
    main(["compare", "--config", str(config_path)])
    capsys.readouterr()

    exit_code = main(
        [
            "show-artifacts",
            "--experiment",
            "cli_compare",
            "--run",
            "latest",
            "--runs-root",
            str(tmp_path),
        ]
    )
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    assert exit_code == 0
    assert payload["run_type"] == "compare"
    assert any(artifact["name"] == "leaderboard_json" for artifact in payload["artifacts"])


def test_cli_show_results_rejects_unsafe_path_component(tmp_path) -> None:
    with pytest.raises(UnsafePathComponentError, match="experiment_name"):
        main(
            [
                "show-results",
                "--experiment",
                "bad name",
                "--run",
                "latest",
                "--runs-root",
                str(tmp_path),
            ]
        )


def test_cli_show_artifacts_rejects_unsafe_path_component(tmp_path) -> None:
    with pytest.raises(UnsafePathComponentError, match="experiment_name"):
        main(
            [
                "show-artifacts",
                "--experiment",
                "bad name",
                "--run",
                "latest",
                "--runs-root",
                str(tmp_path),
            ]
        )
