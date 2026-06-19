from __future__ import annotations

import json
from pathlib import Path

import yaml

from tests.helpers import tiny_config
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
