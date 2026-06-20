"""CLI entry point."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from ts_platform.api.jobs.base import JobStoreProtocol
from ts_platform.api.jobs.factory import build_job_store
from ts_platform.api.jobs.retry import RetryPolicy
from ts_platform.api.jobs.sqlite_store import SQLiteJobStore
from ts_platform.api.jobs.worker import JobWorker
from ts_platform.api.services.artifact_service import ArtifactService
from ts_platform.api.services.experiment_store import ExperimentStore
from ts_platform.api.settings import APISettings
from ts_platform.config.schema import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    PlatformConfig,
    ScalerConfig,
    TrainingConfig,
)
from ts_platform.data import (
    DATASET_CATALOG,
    DATASET_REGISTRY,
    DatasetMetadata,
    load_dataset_catalog,
    profile_csv_dataset,
    register_dataset_catalog,
)
from ts_platform.data.profile import DatasetProfile
from ts_platform.models.registry import registered_model_names
from ts_platform.runner.comparer import CompareRunner
from ts_platform.runner.trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(prog="ts-platform")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="Run a training config")
    train_parser.add_argument("--config", required=True, help="Path to YAML or JSON config")

    compare_parser = subparsers.add_parser("compare", help="Run a multi-model compare config")
    compare_parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML or JSON compare config",
    )

    datasets_parser = subparsers.add_parser("list-datasets", help="List registered datasets")
    datasets_parser.add_argument(
        "--catalog",
        action="append",
        default=[],
        help="Optional local dataset catalog YAML to load before listing",
    )

    profile_dataset_parser = subparsers.add_parser(
        "profile-dataset",
        help="Profile one local CSV dataset",
    )
    profile_dataset_parser.add_argument("--path", required=True, help="Local CSV path")
    profile_dataset_parser.add_argument(
        "--target-cols",
        nargs="+",
        required=True,
        help="One or more target column names",
    )
    profile_dataset_parser.add_argument("--timestamp-col", help="Optional timestamp column")
    profile_dataset_parser.add_argument("--input-len", type=int, help="Optional input length")
    profile_dataset_parser.add_argument("--output-len", type=int, help="Optional output length")
    profile_dataset_parser.add_argument("--name", help="Optional dataset name")

    profile_catalog_parser = subparsers.add_parser(
        "profile-catalog",
        help="Profile CSV entries in a local dataset catalog",
    )
    profile_catalog_parser.add_argument("--catalog", required=True, help="Catalog YAML path")
    profile_catalog_parser.add_argument("--input-len", type=int, help="Optional input length")
    profile_catalog_parser.add_argument("--output-len", type=int, help="Optional output length")

    make_config_parser = subparsers.add_parser(
        "make-config-from-catalog",
        help="Generate a training config from one catalog entry",
    )
    make_config_parser.add_argument("--catalog", required=True, help="Catalog YAML path")
    make_config_parser.add_argument("--dataset", required=True, help="Catalog dataset name")
    make_config_parser.add_argument("--output", required=True, help="Output YAML path")
    make_config_parser.add_argument("--input-len", required=True, type=int, help="Input length")
    make_config_parser.add_argument("--output-len", required=True, type=int, help="Output length")
    make_config_parser.add_argument("--model", required=True, help="Registered model name")
    make_config_parser.add_argument("--epochs", required=True, type=int, help="Training epochs")
    make_config_parser.add_argument("--batch-size", type=int, default=8, help="Batch size")

    subparsers.add_parser("list-models", help="List registered models")

    show_results_parser = subparsers.add_parser("show-results", help="Show run results JSON")
    show_results_parser.add_argument("--experiment", required=True, help="Experiment name")
    show_results_parser.add_argument("--run", default="latest", help="Run id or latest")
    show_results_parser.add_argument(
        "--runs-root",
        default="runs",
        help="Runs root to read from",
    )

    show_leaderboard_parser = subparsers.add_parser(
        "show-leaderboard",
        help="Show compare leaderboard JSON",
    )
    show_leaderboard_parser.add_argument("--experiment", required=True, help="Experiment name")
    show_leaderboard_parser.add_argument("--run", default="latest", help="Run id or latest")
    show_leaderboard_parser.add_argument(
        "--runs-root",
        default="runs",
        help="Runs root to read from",
    )

    show_artifacts_parser = subparsers.add_parser(
        "show-artifacts",
        help="Show run artifact manifest JSON",
    )
    show_artifacts_parser.add_argument("--experiment", required=True, help="Experiment name")
    show_artifacts_parser.add_argument("--run", default="latest", help="Run id or latest")
    show_artifacts_parser.add_argument(
        "--runs-root",
        default="runs",
        help="Runs root to read from",
    )

    show_artifact_parser = subparsers.add_parser(
        "show-artifact",
        help="Show one run artifact file",
    )
    show_artifact_parser.add_argument("--experiment", required=True, help="Experiment name")
    show_artifact_parser.add_argument("--run", default="latest", help="Run id or latest")
    show_artifact_parser.add_argument("--artifact", required=True, help="Artifact name")
    show_artifact_parser.add_argument(
        "--runs-root",
        default="runs",
        help="Runs root to read from",
    )
    show_artifact_parser.add_argument(
        "--output",
        help="Optional file path to write the artifact content to",
    )

    list_jobs_parser = subparsers.add_parser("list-jobs", help="List local API jobs")
    list_jobs_parser.add_argument(
        "--job-backend",
        choices=("json", "sqlite"),
        default="json",
        help="Job backend to inspect",
    )
    list_jobs_parser.add_argument(
        "--jobs-root",
        default="runs/jobs",
        help="Jobs root to read from",
    )
    list_jobs_parser.add_argument(
        "--sqlite-db",
        default="runs/jobs.sqlite3",
        help="SQLite jobs database to read when --job-backend sqlite",
    )

    show_job_parser = subparsers.add_parser("show-job", help="Show one local API job")
    show_job_parser.add_argument("--job-id", required=True, help="Job id to read")
    show_job_parser.add_argument(
        "--job-backend",
        choices=("json", "sqlite"),
        default="json",
        help="Job backend to inspect",
    )
    show_job_parser.add_argument(
        "--jobs-root",
        default="runs/jobs",
        help="Jobs root to read from",
    )
    show_job_parser.add_argument(
        "--sqlite-db",
        default="runs/jobs.sqlite3",
        help="SQLite jobs database to read when --job-backend sqlite",
    )

    show_job_events_parser = subparsers.add_parser(
        "show-job-events",
        help="Show SQLite audit events for one job",
    )
    show_job_events_parser.add_argument("--job-id", required=True, help="Job id to read")
    show_job_events_parser.add_argument(
        "--jobs-root",
        default="runs/jobs",
        help="Jobs root to read from",
    )
    show_job_events_parser.add_argument(
        "--sqlite-db",
        default="runs/jobs.sqlite3",
        help="SQLite jobs database to read",
    )

    show_job_attempts_parser = subparsers.add_parser(
        "show-job-attempts",
        help="Show SQLite worker attempts for one job",
    )
    show_job_attempts_parser.add_argument("--job-id", required=True, help="Job id to read")
    show_job_attempts_parser.add_argument(
        "--jobs-root",
        default="runs/jobs",
        help="Jobs root to read from",
    )
    show_job_attempts_parser.add_argument(
        "--sqlite-db",
        default="runs/jobs.sqlite3",
        help="SQLite jobs database to read",
    )

    list_stale_jobs_parser = subparsers.add_parser(
        "list-stale-jobs",
        help="List stale running SQLite jobs without changing state",
    )
    list_stale_jobs_parser.add_argument(
        "--sqlite-db",
        default="runs/jobs.sqlite3",
        help="SQLite jobs database to inspect",
    )
    list_stale_jobs_parser.add_argument(
        "--jobs-root",
        default="runs/jobs",
        help="Jobs root containing request snapshots",
    )
    list_stale_jobs_parser.add_argument(
        "--older-than-seconds",
        type=int,
        default=3600,
        help="Activity age threshold for stale running jobs",
    )

    mark_stale_timeout_parser = subparsers.add_parser(
        "mark-stale-timeout",
        help="Mark stale running SQLite jobs timed out",
    )
    mark_stale_timeout_parser.add_argument(
        "--sqlite-db",
        default="runs/jobs.sqlite3",
        help="SQLite jobs database to update",
    )
    mark_stale_timeout_parser.add_argument(
        "--jobs-root",
        default="runs/jobs",
        help="Jobs root containing request snapshots",
    )
    mark_stale_timeout_parser.add_argument(
        "--older-than-seconds",
        type=int,
        default=3600,
        help="Activity age threshold for stale running jobs",
    )
    mark_stale_timeout_parser.add_argument(
        "--reason",
        help="Optional timeout reason to store on each job",
    )

    retry_job_parser = subparsers.add_parser(
        "retry-job",
        help="Requeue a failed, timed out, or cancelled SQLite job",
    )
    retry_job_parser.add_argument("--job-id", required=True, help="Job id to retry")
    retry_job_parser.add_argument(
        "--sqlite-db",
        default="runs/jobs.sqlite3",
        help="SQLite jobs database to update",
    )
    retry_job_parser.add_argument(
        "--jobs-root",
        default="runs/jobs",
        help="Jobs root containing request snapshots",
    )
    retry_job_parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum allowed attempts before rejecting retry",
    )

    worker_once_parser = subparsers.add_parser(
        "worker-once",
        help="Claim and run one queued SQLite job",
    )
    worker_once_parser.add_argument(
        "--sqlite-db",
        default="runs/jobs.sqlite3",
        help="SQLite jobs database to claim from",
    )
    worker_once_parser.add_argument(
        "--jobs-root",
        default="runs/jobs",
        help="Jobs root containing request snapshots",
    )
    worker_once_parser.add_argument(
        "--runs-root",
        default="runs",
        help="Runs root to write train or compare results",
    )
    worker_once_parser.add_argument(
        "--worker-id",
        default="local_worker",
        help="Safe worker id recorded in job attempts",
    )

    worker_loop_parser = subparsers.add_parser(
        "worker-loop",
        help="Claim and run queued SQLite jobs with finite local bounds",
    )
    worker_loop_parser.add_argument(
        "--sqlite-db",
        default="runs/jobs.sqlite3",
        help="SQLite jobs database to claim from",
    )
    worker_loop_parser.add_argument(
        "--jobs-root",
        default="runs/jobs",
        help="Jobs root containing request snapshots",
    )
    worker_loop_parser.add_argument(
        "--runs-root",
        default="runs",
        help="Runs root to write train or compare results",
    )
    worker_loop_parser.add_argument(
        "--worker-id",
        default="local_worker",
        help="Safe worker id recorded in job attempts",
    )
    worker_loop_parser.add_argument(
        "--max-jobs",
        type=int,
        default=1,
        help="Maximum jobs to process before exiting",
    )
    worker_loop_parser.add_argument(
        "--max-idle-cycles",
        type=int,
        default=1,
        help="Maximum idle polls before exiting",
    )
    worker_loop_parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Seconds to sleep between idle polls",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        training_result = Trainer.from_config_path(args.config).run()
        print(json.dumps(training_result.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "compare":
        compare_result = CompareRunner.from_config_path(args.config).run()
        print(json.dumps(compare_result.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "list-datasets":
        for catalog_path in args.catalog:
            register_dataset_catalog(catalog_path)
        payload = {"datasets": DATASET_CATALOG.list(), "names": DATASET_REGISTRY.names()}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "profile-dataset":
        profile = profile_csv_dataset(
            path=args.path,
            target_cols=args.target_cols,
            timestamp_col=args.timestamp_col,
            input_len=args.input_len,
            output_len=args.output_len,
            name=args.name,
        )
        print(json.dumps(profile.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "profile-catalog":
        profiles = [
            _profile_catalog_entry(
                metadata,
                input_len=args.input_len,
                output_len=args.output_len,
            )
            for metadata in load_dataset_catalog(args.catalog)
        ]
        print(json.dumps({"profiles": profiles}, indent=2, sort_keys=True))
        return 0
    if args.command == "make-config-from-catalog":
        metadata = _find_catalog_metadata(load_dataset_catalog(args.catalog), args.dataset)
        config = _config_from_catalog_metadata(
            metadata,
            input_len=args.input_len,
            output_len=args.output_len,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
        payload = {
            "output": str(output_path),
            "config": config.model_dump(mode="json"),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "list-models":
        print(json.dumps({"models": registered_model_names()}, indent=2, sort_keys=True))
        return 0
    if args.command == "show-results":
        results_payload = ExperimentStore(Path(args.runs_root)).read_results(
            args.experiment,
            args.run,
        )
        print(json.dumps(results_payload, indent=2, sort_keys=True))
        return 0
    if args.command == "show-leaderboard":
        leaderboard_payload = ExperimentStore(Path(args.runs_root)).read_leaderboard(
            args.experiment,
            args.run,
        )
        print(json.dumps(leaderboard_payload, indent=2, sort_keys=True))
        return 0
    if args.command == "show-artifacts":
        artifacts_payload = ExperimentStore(Path(args.runs_root)).read_artifacts(
            args.experiment,
            args.run,
        )
        print(json.dumps(artifacts_payload, indent=2, sort_keys=True))
        return 0
    if args.command == "show-artifact":
        artifact = ArtifactService(Path(args.runs_root)).resolve_artifact(
            args.experiment,
            args.run,
            args.artifact,
        )
        content = _read_text_artifact(artifact.path)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding="utf-8")
        else:
            print(content, end="")
        return 0
    if args.command == "list-jobs":
        jobs_payload = [job.to_dict() for job in _job_store_from_args(args).list_jobs()]
        print(json.dumps({"jobs": jobs_payload}, indent=2, sort_keys=True))
        return 0
    if args.command == "show-job":
        job_payload = _job_store_from_args(args).get_job(args.job_id).to_dict()
        print(json.dumps(job_payload, indent=2, sort_keys=True))
        return 0
    if args.command == "show-job-events":
        store = _sqlite_store_from_args(args)
        store.get_job(args.job_id)
        print(json.dumps(store.list_events(args.job_id), indent=2, sort_keys=True))
        return 0
    if args.command == "show-job-attempts":
        store = _sqlite_store_from_args(args)
        store.get_job(args.job_id)
        print(json.dumps(store.list_attempts(args.job_id), indent=2, sort_keys=True))
        return 0
    if args.command == "list-stale-jobs":
        if args.older_than_seconds <= 0:
            parser.error("--older-than-seconds must be > 0")
        store = _sqlite_store_from_args(args)
        jobs_payload = [
            job.to_dict()
            for job in store.list_stale_running_jobs(
                older_than_seconds=args.older_than_seconds,
            )
        ]
        print(json.dumps({"jobs": jobs_payload}, indent=2, sort_keys=True))
        return 0
    if args.command == "mark-stale-timeout":
        if args.older_than_seconds <= 0:
            parser.error("--older-than-seconds must be > 0")
        store = _sqlite_store_from_args(args)
        timed_out_payload = [
            job.to_dict()
            for job in store.mark_stale_running_jobs_timed_out(
                older_than_seconds=args.older_than_seconds,
                reason=args.reason,
            )
        ]
        print(json.dumps({"timed_out": timed_out_payload}, indent=2, sort_keys=True))
        return 0
    if args.command == "retry-job":
        if args.max_attempts < 1:
            parser.error("--max-attempts must be >= 1")
        store = _sqlite_store_from_args(args)
        policy = RetryPolicy(max_attempts=args.max_attempts)
        job_payload = store.retry_job(args.job_id, policy=policy).to_dict()
        print(json.dumps(job_payload, indent=2, sort_keys=True))
        return 0
    if args.command == "worker-once":
        store = SQLiteJobStore(Path(args.jobs_root), Path(args.sqlite_db))
        worker = JobWorker(
            store=store,
            runs_root=Path(args.runs_root),
            worker_id=args.worker_id,
        )
        job = worker.run_once()
        payload = {"status": "idle"} if job is None else job.to_dict()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "worker-loop":
        if args.max_jobs < 1:
            parser.error("--max-jobs must be >= 1")
        if args.max_idle_cycles < 1:
            parser.error("--max-idle-cycles must be >= 1")
        if args.sleep_seconds < 0:
            parser.error("--sleep-seconds must be >= 0")
        store = SQLiteJobStore(Path(args.jobs_root), Path(args.sqlite_db))
        worker = JobWorker(
            store=store,
            runs_root=Path(args.runs_root),
            worker_id=args.worker_id,
        )
        loop_jobs_payload: list[dict[str, object]] = []
        processed = 0
        idle_cycles = 0
        while processed < args.max_jobs and idle_cycles < args.max_idle_cycles:
            job = worker.run_once()
            if job is None:
                idle_cycles += 1
                if idle_cycles < args.max_idle_cycles:
                    time.sleep(args.sleep_seconds)
                continue
            processed += 1
            loop_jobs_payload.append(job.to_dict())
        payload = {
            "worker_id": worker.worker_id,
            "processed": processed,
            "idle_cycles": idle_cycles,
            "jobs": loop_jobs_payload,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


def _read_text_artifact(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        msg = f"artifact is not UTF-8 text: {path}"
        raise ValueError(msg) from exc


def _profile_catalog_entry(
    metadata: DatasetMetadata,
    *,
    input_len: int | None,
    output_len: int | None,
) -> dict[str, Any]:
    if metadata.dataset_type != "csv":
        return _profile_stub(metadata, warning=f"unsupported dataset_type: {metadata.dataset_type}")
    if metadata.path is None:
        return _profile_stub(metadata, warning="missing path")
    if metadata.target_cols is None:
        return _profile_stub(metadata, warning="missing target_cols")
    return profile_csv_dataset(
        path=metadata.path,
        target_cols=metadata.target_cols,
        timestamp_col=metadata.timestamp_col,
        input_len=input_len,
        output_len=output_len,
        name=metadata.name,
    ).to_dict()


def _profile_stub(metadata: DatasetMetadata, *, warning: str) -> dict[str, Any]:
    return DatasetProfile(
        name=metadata.name,
        dataset_type=metadata.dataset_type,
        path=metadata.path or "",
        exists=False,
        row_count=0,
        column_count=0,
        columns=[],
        timestamp_col=metadata.timestamp_col,
        start_timestamp=None,
        end_timestamp=None,
        target_cols=metadata.target_cols or [],
        target_missing_counts={},
        target_dtypes={},
        duplicate_timestamp_count=None,
        inferred_frequency=None,
        min_required_rows=None,
        can_build_windows=False,
        warnings=[warning],
    ).to_dict()


def _find_catalog_metadata(metadata: list[DatasetMetadata], name: str) -> DatasetMetadata:
    normalized = name.strip().lower()
    for item in reversed(metadata):
        if item.name.strip().lower() == normalized:
            return item
    msg = f"unknown dataset metadata: {normalized}"
    raise KeyError(msg)


def _config_from_catalog_metadata(
    metadata: DatasetMetadata,
    *,
    input_len: int,
    output_len: int,
    model_name: str,
    epochs: int,
    batch_size: int,
) -> PlatformConfig:
    if metadata.dataset_type != "csv":
        msg = "make-config-from-catalog only supports csv datasets"
        raise ValueError(msg)
    if metadata.path is None:
        msg = "csv catalog entry must include path"
        raise ValueError(msg)
    if metadata.target_cols is None:
        msg = "csv catalog entry must include target_cols"
        raise ValueError(msg)
    data_params: dict[str, Any] = {
        "path": metadata.path,
        "target_cols": metadata.target_cols,
        "missing_policy": "error",
        "sort_by_time": True,
    }
    if metadata.timestamp_col is not None:
        data_params["timestamp_col"] = metadata.timestamp_col
    return PlatformConfig(
        experiment=ExperimentConfig(
            name=f"train_{metadata.name}_{model_name}",
            output_dir=Path("runs"),
            overwrite=True,
        ),
        data=DataConfig(
            name="csv",
            input_len=input_len,
            output_len=output_len,
            batch_size=batch_size,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            scaler=ScalerConfig(name="standard"),
            params=data_params,
        ),
        model=ModelConfig(name=model_name),
        training=TrainingConfig(
            epochs=epochs,
            learning_rate=0.01,
            device="cpu",
            optimizer="adam",
            loss="mse",
            checkpoint_every=1,
        ),
        evaluation=EvaluationConfig(
            metrics=["mae", "mse", "rmse", "wape"],
            include_scaled_metrics=False,
        ),
    )


def _job_store_from_args(args: argparse.Namespace) -> JobStoreProtocol:
    return build_job_store(
        APISettings(
            job_backend=args.job_backend,
            jobs_root=Path(args.jobs_root),
            sqlite_jobs_db_path=Path(args.sqlite_db),
        )
    )


def _sqlite_store_from_args(args: argparse.Namespace) -> SQLiteJobStore:
    return SQLiteJobStore(Path(args.jobs_root), Path(args.sqlite_db))


if __name__ == "__main__":
    raise SystemExit(main())
