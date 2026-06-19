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
