from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from tests.helpers import tiny_config
from ts_platform.api.jobs.sqlite_store import SQLiteJobStore
from ts_platform.api.jobs.store import JobStateConflictError, JobStore, UnsafeJobIdError
from ts_platform.api.services.artifact_service import ArtifactAccessForbiddenError
from ts_platform.api.services.experiment_store import (
    ExperimentArtifactNotFoundError,
    UnsafePathComponentError,
)
from ts_platform.cli.main import main
from ts_platform.config.loader import load_config


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


def test_cli_profile_dataset(capsys) -> None:
    exit_code = main(
        [
            "profile-dataset",
            "--path",
            "tests/fixtures/tiny_series.csv",
            "--target-cols",
            "value",
            "--timestamp-col",
            "timestamp",
            "--input-len",
            "8",
            "--output-len",
            "2",
        ]
    )
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    assert exit_code == 0
    assert payload["exists"] is True
    assert payload["row_count"] == 90
    assert payload["can_build_windows"] is True
    assert payload["warnings"] == []


def test_cli_profile_dataset_missing_file(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "profile-dataset",
            "--path",
            str(tmp_path / "missing.csv"),
            "--target-cols",
            "value",
            "--timestamp-col",
            "timestamp",
        ]
    )
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    assert exit_code == 0
    assert payload["exists"] is False
    assert payload["can_build_windows"] is False
    assert "missing file" in payload["warnings"]


def test_cli_profile_dataset_detects_insufficient_rows(tmp_path, capsys) -> None:
    path = tmp_path / "short.csv"
    path.write_text("timestamp,value\n2024-01-01,1\n2024-01-02,2\n", encoding="utf-8")

    exit_code = main(
        [
            "profile-dataset",
            "--path",
            str(path),
            "--target-cols",
            "value",
            "--timestamp-col",
            "timestamp",
            "--input-len",
            "2",
            "--output-len",
            "2",
        ]
    )
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    assert exit_code == 0
    assert payload["can_build_windows"] is False
    assert "insufficient rows for windows" in payload["warnings"]


def test_cli_profile_catalog(capsys) -> None:
    exit_code = main(
        [
            "profile-catalog",
            "--catalog",
            "configs/datasets/local_csv.yaml",
            "--input-len",
            "8",
            "--output-len",
            "2",
        ]
    )
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    assert exit_code == 0
    assert len(payload["profiles"]) == 1
    assert payload["profiles"][0]["name"] == "tiny_csv"
    assert payload["profiles"][0]["exists"] is True


def test_cli_profile_catalog_skips_unsupported_dataset_type(tmp_path, capsys) -> None:
    catalog_path = tmp_path / "synthetic_catalog.yaml"
    catalog_path.write_text(
        "datasets:\n"
        "  - name: synthetic_entry\n"
        "    dataset_type: synthetic\n"
        "    domain: demo\n"
        "    description: Synthetic metadata\n",
        encoding="utf-8",
    )

    exit_code = main(["profile-catalog", "--catalog", str(catalog_path)])
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    assert exit_code == 0
    assert payload["profiles"][0]["name"] == "synthetic_entry"
    assert "unsupported dataset_type: synthetic" in payload["profiles"][0]["warnings"]


def test_cli_profile_catalog_reports_warnings(tmp_path, capsys) -> None:
    catalog_path = tmp_path / "missing_file_catalog.yaml"
    catalog_path.write_text(
        "datasets:\n"
        "  - name: missing_file\n"
        "    dataset_type: csv\n"
        "    domain: demo\n"
        "    description: Missing file metadata\n"
        f"    path: {tmp_path / 'missing.csv'}\n"
        "    target_cols: [value]\n",
        encoding="utf-8",
    )

    exit_code = main(["profile-catalog", "--catalog", str(catalog_path)])
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    assert exit_code == 0
    assert "missing file" in payload["profiles"][0]["warnings"]


def test_cli_make_config_from_catalog(tmp_path, capsys) -> None:
    output_path = tmp_path / "tiny_csv_config.yaml"

    exit_code = main(
        [
            "make-config-from-catalog",
            "--catalog",
            "configs/datasets/local_csv.yaml",
            "--dataset",
            "tiny_csv",
            "--output",
            str(output_path),
            "--input-len",
            "8",
            "--output-len",
            "2",
            "--model",
            "linear",
            "--epochs",
            "1",
        ]
    )
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    assert exit_code == 0
    assert output_path.exists()
    assert payload["output"] == str(output_path)
    assert payload["config"]["experiment"]["name"] == "train_tiny_csv_linear"
    assert payload["config"]["data"]["params"]["path"] == "tests/fixtures/tiny_series.csv"


def test_make_config_from_catalog_output_loads(tmp_path, capsys) -> None:
    output_path = tmp_path / "loadable_config.yaml"
    main(
        [
            "make-config-from-catalog",
            "--catalog",
            "configs/datasets/local_csv.yaml",
            "--dataset",
            "tiny_csv",
            "--output",
            str(output_path),
            "--input-len",
            "8",
            "--output-len",
            "2",
            "--model",
            "linear",
            "--epochs",
            "1",
        ]
    )
    capsys.readouterr()

    config = load_config(output_path)

    assert config.experiment.name == "train_tiny_csv_linear"
    assert config.data.name == "csv"
    assert config.data.params["target_cols"] == ["value"]
    assert config.training.epochs == 1


def test_make_config_from_catalog_rejects_missing_dataset(tmp_path) -> None:
    with pytest.raises(KeyError, match="unknown dataset metadata: missing"):
        main(
            [
                "make-config-from-catalog",
                "--catalog",
                "configs/datasets/local_csv.yaml",
                "--dataset",
                "missing",
                "--output",
                str(tmp_path / "config.yaml"),
                "--input-len",
                "8",
                "--output-len",
                "2",
                "--model",
                "linear",
                "--epochs",
                "1",
            ]
        )


def test_make_config_from_catalog_rejects_non_csv_dataset(tmp_path) -> None:
    catalog_path = tmp_path / "synthetic_catalog.yaml"
    catalog_path.write_text(
        "datasets:\n"
        "  - name: synthetic_entry\n"
        "    dataset_type: synthetic\n"
        "    domain: demo\n"
        "    description: Synthetic metadata\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="only supports csv"):
        main(
            [
                "make-config-from-catalog",
                "--catalog",
                str(catalog_path),
                "--dataset",
                "synthetic_entry",
                "--output",
                str(tmp_path / "config.yaml"),
                "--input-len",
                "8",
                "--output-len",
                "2",
                "--model",
                "linear",
                "--epochs",
                "1",
            ]
        )


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


def _write_cli_artifact_run(tmp_path) -> Path:
    run_dir = tmp_path / "cli_artifact" / "latest"
    run_dir.mkdir(parents=True)
    leaderboard_path = run_dir / "leaderboard.json"
    leaderboard_path.write_text('[{"rank": 1, "model": "naive"}]', encoding="utf-8")
    checkpoint_path = run_dir / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    (run_dir / "artifacts.json").write_text(
        json.dumps(
            {
                "run_type": "compare",
                "experiment_name": "cli_artifact",
                "compare_run_id": "latest",
                "compare_run_dir": str(run_dir),
                "artifacts": [
                    {
                        "name": "leaderboard_json",
                        "kind": "json",
                        "path": str(leaderboard_path),
                        "description": "Leaderboard",
                    },
                    {
                        "name": "checkpoint",
                        "kind": "checkpoint",
                        "path": str(checkpoint_path),
                        "description": "Checkpoint",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return run_dir


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


def test_cli_show_artifact_prints_content(tmp_path, capsys) -> None:
    _write_cli_artifact_run(tmp_path)

    exit_code = main(
        [
            "show-artifact",
            "--experiment",
            "cli_artifact",
            "--run",
            "latest",
            "--artifact",
            "leaderboard_json",
            "--runs-root",
            str(tmp_path),
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert stdout == '[{"rank": 1, "model": "naive"}]'


def test_cli_show_artifact_writes_output_file(tmp_path, capsys) -> None:
    _write_cli_artifact_run(tmp_path)
    output_path = tmp_path / "downloads" / "leaderboard.json"

    exit_code = main(
        [
            "show-artifact",
            "--experiment",
            "cli_artifact",
            "--run",
            "latest",
            "--artifact",
            "leaderboard_json",
            "--runs-root",
            str(tmp_path),
            "--output",
            str(output_path),
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert stdout == ""
    assert output_path.read_text(encoding="utf-8") == '[{"rank": 1, "model": "naive"}]'


def test_cli_show_artifact_rejects_unknown_artifact(tmp_path) -> None:
    _write_cli_artifact_run(tmp_path)

    with pytest.raises(ExperimentArtifactNotFoundError, match="not registered"):
        main(
            [
                "show-artifact",
                "--experiment",
                "cli_artifact",
                "--run",
                "latest",
                "--artifact",
                "missing",
                "--runs-root",
                str(tmp_path),
            ]
        )


def test_cli_show_artifact_rejects_unsafe_artifact_name(tmp_path) -> None:
    _write_cli_artifact_run(tmp_path)

    with pytest.raises(UnsafePathComponentError, match="artifact_name"):
        main(
            [
                "show-artifact",
                "--experiment",
                "cli_artifact",
                "--run",
                "latest",
                "--artifact",
                "../secret",
                "--runs-root",
                str(tmp_path),
            ]
        )


def test_cli_show_artifact_rejects_checkpoint_by_default(tmp_path) -> None:
    _write_cli_artifact_run(tmp_path)

    with pytest.raises(ArtifactAccessForbiddenError, match="checkpoint"):
        main(
            [
                "show-artifact",
                "--experiment",
                "cli_artifact",
                "--run",
                "latest",
                "--artifact",
                "checkpoint",
                "--runs-root",
                str(tmp_path),
            ]
        )


def test_cli_show_artifact_rejects_cross_run_artifact_path(tmp_path) -> None:
    run_dir = tmp_path / "cli_artifact" / "latest"
    other_run_dir = tmp_path / "other_cli_artifact" / "latest"
    run_dir.mkdir(parents=True)
    other_run_dir.mkdir(parents=True)
    other_artifact_path = other_run_dir / "secret.json"
    other_artifact_path.write_text('{"secret": true}', encoding="utf-8")
    (run_dir / "artifacts.json").write_text(
        json.dumps(
            {
                "run_type": "compare",
                "experiment_name": "cli_artifact",
                "compare_run_id": "latest",
                "compare_run_dir": str(run_dir),
                "artifacts": [
                    {
                        "name": "secret",
                        "kind": "json",
                        "path": str(other_artifact_path),
                        "description": "Cross-run artifact",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(UnsafePathComponentError, match="escapes run directory"):
        main(
            [
                "show-artifact",
                "--experiment",
                "cli_artifact",
                "--run",
                "latest",
                "--artifact",
                "secret",
                "--runs-root",
                str(tmp_path),
            ]
        )


def test_cli_show_artifact_rejects_tampered_manifest_run_dir(tmp_path) -> None:
    run_dir = tmp_path / "cli_artifact" / "latest"
    other_run_dir = tmp_path / "other_cli_artifact" / "latest"
    run_dir.mkdir(parents=True)
    other_run_dir.mkdir(parents=True)
    other_artifact_path = other_run_dir / "secret.json"
    other_artifact_path.write_text('{"secret": true}', encoding="utf-8")
    (run_dir / "artifacts.json").write_text(
        json.dumps(
            {
                "run_type": "compare",
                "experiment_name": "cli_artifact",
                "compare_run_id": "latest",
                "compare_run_dir": str(other_run_dir),
                "artifacts": [
                    {
                        "name": "secret",
                        "kind": "json",
                        "path": str(other_artifact_path),
                        "description": "Tampered compare_run_dir artifact",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(UnsafePathComponentError, match="escapes run directory"):
        main(
            [
                "show-artifact",
                "--experiment",
                "cli_artifact",
                "--run",
                "latest",
                "--artifact",
                "secret",
                "--runs-root",
                str(tmp_path),
            ]
        )


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


def test_cli_list_jobs(tmp_path, capsys) -> None:
    store = JobStore(tmp_path / "jobs")
    job = store.create_job("train", "cli_job_list", {"config": True})

    exit_code = main(["list-jobs", "--jobs-root", str(tmp_path / "jobs")])
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    assert exit_code == 0
    assert [item["job_id"] for item in payload["jobs"]] == [job.job_id]


def test_cli_show_job(tmp_path, capsys) -> None:
    store = JobStore(tmp_path / "jobs")
    job = store.create_job("compare", "cli_job_show", {"config": True})

    exit_code = main(["show-job", "--job-id", job.job_id, "--jobs-root", str(tmp_path / "jobs")])
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    assert exit_code == 0
    assert payload["job_id"] == job.job_id
    assert payload["job_type"] == "compare"


def test_cli_list_jobs_sqlite_backend(tmp_path, capsys) -> None:
    store = SQLiteJobStore(tmp_path / "jobs", tmp_path / "jobs.sqlite3")
    job = store.create_job("train", "cli_sqlite_list", {"config": True})

    exit_code = main(
        [
            "list-jobs",
            "--job-backend",
            "sqlite",
            "--jobs-root",
            str(tmp_path / "jobs"),
            "--sqlite-db",
            str(tmp_path / "jobs.sqlite3"),
        ]
    )
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    assert exit_code == 0
    assert [item["job_id"] for item in payload["jobs"]] == [job.job_id]


def test_cli_show_job_sqlite_backend(tmp_path, capsys) -> None:
    store = SQLiteJobStore(tmp_path / "jobs", tmp_path / "jobs.sqlite3")
    job = store.create_job("compare", "cli_sqlite_show", {"config": True})

    exit_code = main(
        [
            "show-job",
            "--job-id",
            job.job_id,
            "--job-backend",
            "sqlite",
            "--jobs-root",
            str(tmp_path / "jobs"),
            "--sqlite-db",
            str(tmp_path / "jobs.sqlite3"),
        ]
    )
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    assert exit_code == 0
    assert payload["job_id"] == job.job_id
    assert payload["job_type"] == "compare"


def test_cli_show_job_events(tmp_path, capsys) -> None:
    store = SQLiteJobStore(tmp_path / "jobs", tmp_path / "jobs.sqlite3")
    job = store.create_job("train", "cli_events", {"config": True})

    exit_code = main(
        [
            "show-job-events",
            "--job-id",
            job.job_id,
            "--jobs-root",
            str(tmp_path / "jobs"),
            "--sqlite-db",
            str(tmp_path / "jobs.sqlite3"),
        ]
    )
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    assert exit_code == 0
    assert [event["event_type"] for event in payload] == ["job_created"]


def test_cli_show_job_attempts(tmp_path, capsys) -> None:
    store = SQLiteJobStore(tmp_path / "jobs", tmp_path / "jobs.sqlite3")
    job = store.create_job("train", "cli_attempts", {"config": True})
    claimed = store.claim_next_queued_job(worker_id="cli_worker")
    assert claimed is not None

    exit_code = main(
        [
            "show-job-attempts",
            "--job-id",
            job.job_id,
            "--jobs-root",
            str(tmp_path / "jobs"),
            "--sqlite-db",
            str(tmp_path / "jobs.sqlite3"),
        ]
    )
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    assert exit_code == 0
    assert payload[0]["attempt_id"] == claimed.attempt_id
    assert payload[0]["worker_id"] == "cli_worker"


def test_cli_show_job_events_rejects_unsafe_job_id(tmp_path) -> None:
    with pytest.raises(UnsafeJobIdError, match="job_id"):
        main(
            [
                "show-job-events",
                "--job-id",
                "../escape",
                "--jobs-root",
                str(tmp_path / "jobs"),
                "--sqlite-db",
                str(tmp_path / "jobs.sqlite3"),
            ]
        )


def test_cli_list_stale_jobs(tmp_path, capsys) -> None:
    store = SQLiteJobStore(tmp_path / "jobs", tmp_path / "jobs.sqlite3")
    job = store.create_job("train", "cli_stale", {"config": True})
    old = "2000-01-01T00:00:00+00:00"
    store.update_job(
        replace(job, status="running", started_at=old, updated_at=old),
        touch=False,
    )

    exit_code = main(
        [
            "list-stale-jobs",
            "--jobs-root",
            str(tmp_path / "jobs"),
            "--sqlite-db",
            str(tmp_path / "jobs.sqlite3"),
            "--older-than-seconds",
            "60",
        ]
    )
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    assert exit_code == 0
    assert [item["job_id"] for item in payload["jobs"]] == [job.job_id]
    assert store.get_job(job.job_id).status == "running"


def test_cli_mark_stale_timeout(tmp_path, capsys) -> None:
    store = SQLiteJobStore(tmp_path / "jobs", tmp_path / "jobs.sqlite3")
    job = store.create_job("train", "cli_stale_timeout", {"config": True})
    old = "2000-01-01T00:00:00+00:00"
    store.update_job(
        replace(job, status="running", started_at=old, updated_at=old),
        touch=False,
    )

    exit_code = main(
        [
            "mark-stale-timeout",
            "--jobs-root",
            str(tmp_path / "jobs"),
            "--sqlite-db",
            str(tmp_path / "jobs.sqlite3"),
            "--older-than-seconds",
            "60",
            "--reason",
            "cli timeout",
        ]
    )
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    assert exit_code == 0
    assert [item["job_id"] for item in payload["timed_out"]] == [job.job_id]
    assert store.get_job(job.job_id).status == "timed_out"
    assert store.get_job(job.job_id).error == "cli timeout"


def test_cli_retry_job(tmp_path, capsys) -> None:
    store = SQLiteJobStore(tmp_path / "jobs", tmp_path / "jobs.sqlite3")
    job = store.create_job("train", "cli_retry", {"config": True})
    store.mark_failed(job.job_id, "boom")

    exit_code = main(
        [
            "retry-job",
            "--job-id",
            job.job_id,
            "--jobs-root",
            str(tmp_path / "jobs"),
            "--sqlite-db",
            str(tmp_path / "jobs.sqlite3"),
            "--max-attempts",
            "3",
        ]
    )
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    assert exit_code == 0
    assert payload["status"] == "queued"
    assert payload["error"] is None
    assert store.get_job(job.job_id).status == "queued"


def test_cli_retry_job_respects_max_attempts(tmp_path) -> None:
    store = SQLiteJobStore(tmp_path / "jobs", tmp_path / "jobs.sqlite3")
    job = store.create_job("train", "cli_retry_max", {"config": True})
    claimed = store.claim_next_queued_job(worker_id="cli_worker")
    assert claimed is not None
    store.mark_failed(job.job_id, "boom")
    store.mark_attempt_failed(claimed.attempt_id, "boom")

    with pytest.raises(JobStateConflictError, match="max attempts"):
        main(
            [
                "retry-job",
                "--job-id",
                job.job_id,
                "--jobs-root",
                str(tmp_path / "jobs"),
                "--sqlite-db",
                str(tmp_path / "jobs.sqlite3"),
                "--max-attempts",
                "1",
            ]
        )


def test_cli_retry_job_rejects_unsafe_job_id(tmp_path) -> None:
    with pytest.raises(UnsafeJobIdError, match="job_id"):
        main(
            [
                "retry-job",
                "--job-id",
                "../escape",
                "--jobs-root",
                str(tmp_path / "jobs"),
                "--sqlite-db",
                str(tmp_path / "jobs.sqlite3"),
            ]
        )


def test_cli_list_jobs_rejects_unknown_backend(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["list-jobs", "--job-backend", "memory"])

    assert exc_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_cli_worker_once_idle(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "worker-once",
            "--jobs-root",
            str(tmp_path / "jobs"),
            "--sqlite-db",
            str(tmp_path / "jobs.sqlite3"),
            "--runs-root",
            str(tmp_path / "runs"),
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert json.loads(stdout) == {"status": "idle"}


def test_cli_worker_once_runs_train_job(tmp_path, capsys) -> None:
    store = SQLiteJobStore(tmp_path / "jobs", tmp_path / "jobs.sqlite3")
    config = tiny_config(tmp_path / "requested", name="cli_worker_train")
    job = store.create_job("train", config.experiment.name, config.model_dump(mode="json"))

    exit_code = main(
        [
            "worker-once",
            "--jobs-root",
            str(tmp_path / "jobs"),
            "--sqlite-db",
            str(tmp_path / "jobs.sqlite3"),
            "--runs-root",
            str(tmp_path / "runs"),
            "--worker-id",
            "cli_worker",
        ]
    )
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    assert exit_code == 0
    assert payload["job_id"] == job.job_id
    assert payload["status"] == "succeeded"
    assert Path(payload["result_path"]).is_file()


def test_cli_worker_once_rejects_unsafe_worker_id(tmp_path) -> None:
    with pytest.raises(ValueError, match="worker_id"):
        main(
            [
                "worker-once",
                "--jobs-root",
                str(tmp_path / "jobs"),
                "--sqlite-db",
                str(tmp_path / "jobs.sqlite3"),
                "--worker-id",
                "bad worker",
            ]
        )


def test_cli_worker_loop_idle(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "worker-loop",
            "--jobs-root",
            str(tmp_path / "jobs"),
            "--sqlite-db",
            str(tmp_path / "jobs.sqlite3"),
            "--runs-root",
            str(tmp_path / "runs"),
            "--sleep-seconds",
            "0",
        ]
    )
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    assert exit_code == 0
    assert payload == {
        "worker_id": "local_worker",
        "processed": 0,
        "idle_cycles": 1,
        "jobs": [],
    }


def test_cli_worker_loop_processes_one_job(tmp_path, capsys) -> None:
    store = SQLiteJobStore(tmp_path / "jobs", tmp_path / "jobs.sqlite3")
    config = tiny_config(tmp_path / "requested", name="cli_worker_loop")
    job = store.create_job("train", config.experiment.name, config.model_dump(mode="json"))

    exit_code = main(
        [
            "worker-loop",
            "--jobs-root",
            str(tmp_path / "jobs"),
            "--sqlite-db",
            str(tmp_path / "jobs.sqlite3"),
            "--runs-root",
            str(tmp_path / "runs"),
            "--worker-id",
            "loop_worker",
            "--sleep-seconds",
            "0",
        ]
    )
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    assert exit_code == 0
    assert payload["worker_id"] == "loop_worker"
    assert payload["processed"] == 1
    assert payload["idle_cycles"] == 0
    assert payload["jobs"][0]["job_id"] == job.job_id
    assert payload["jobs"][0]["status"] == "succeeded"


def test_cli_worker_loop_respects_max_jobs(tmp_path, capsys) -> None:
    store = SQLiteJobStore(tmp_path / "jobs", tmp_path / "jobs.sqlite3")
    first_config = tiny_config(tmp_path / "requested", name="cli_worker_loop_first")
    second_config = tiny_config(tmp_path / "requested", name="cli_worker_loop_second")
    first = store.create_job(
        "train", first_config.experiment.name, first_config.model_dump(mode="json")
    )
    second = store.create_job(
        "train",
        second_config.experiment.name,
        second_config.model_dump(mode="json"),
    )

    exit_code = main(
        [
            "worker-loop",
            "--jobs-root",
            str(tmp_path / "jobs"),
            "--sqlite-db",
            str(tmp_path / "jobs.sqlite3"),
            "--runs-root",
            str(tmp_path / "runs"),
            "--max-jobs",
            "1",
            "--sleep-seconds",
            "0",
        ]
    )
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    statuses = [store.get_job(job.job_id).status for job in (first, second)]

    assert exit_code == 0
    assert payload["processed"] == 1
    assert statuses.count("succeeded") == 1
    assert statuses.count("queued") == 1


def test_cli_worker_loop_rejects_invalid_max_jobs(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["worker-loop", "--max-jobs", "0"])

    assert exc_info.value.code == 2
    assert "--max-jobs must be >= 1" in capsys.readouterr().err


def test_cli_worker_loop_rejects_unsafe_worker_id(tmp_path) -> None:
    with pytest.raises(ValueError, match="worker_id"):
        main(
            [
                "worker-loop",
                "--jobs-root",
                str(tmp_path / "jobs"),
                "--sqlite-db",
                str(tmp_path / "jobs.sqlite3"),
                "--worker-id",
                "bad worker",
            ]
        )
