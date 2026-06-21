"""Job inspection commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from ts_platform.api.jobs.base import JobStoreProtocol
from ts_platform.api.jobs.factory import build_job_store
from ts_platform.api.jobs.sqlite_store import SQLiteJobStore
from ts_platform.api.settings import APISettings
from ts_platform.cli.utils import print_json


def register(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register job inspection commands."""

    list_jobs_parser = subparsers.add_parser("list-jobs", help="List local API jobs")
    _add_job_store_args(list_jobs_parser)
    list_jobs_parser.set_defaults(handler=handle_list_jobs)

    show_job_parser = subparsers.add_parser("show-job", help="Show one local API job")
    show_job_parser.add_argument("--job-id", required=True, help="Job id to read")
    _add_job_store_args(show_job_parser)
    show_job_parser.set_defaults(handler=handle_show_job)

    show_job_events_parser = subparsers.add_parser(
        "show-job-events",
        help="Show SQLite audit events for one job",
    )
    show_job_events_parser.add_argument("--job-id", required=True, help="Job id to read")
    _add_sqlite_job_store_args(show_job_events_parser)
    show_job_events_parser.set_defaults(handler=handle_show_job_events)

    show_job_attempts_parser = subparsers.add_parser(
        "show-job-attempts",
        help="Show SQLite worker attempts for one job",
    )
    show_job_attempts_parser.add_argument("--job-id", required=True, help="Job id to read")
    _add_sqlite_job_store_args(show_job_attempts_parser)
    show_job_attempts_parser.set_defaults(handler=handle_show_job_attempts)


def handle_list_jobs(args: argparse.Namespace) -> int:
    """List local API jobs."""

    jobs_payload = [job.to_dict() for job in _job_store_from_args(args).list_jobs()]
    print_json({"jobs": jobs_payload})
    return 0


def handle_show_job(args: argparse.Namespace) -> int:
    """Show one local API job."""

    job_payload = _job_store_from_args(args).get_job(args.job_id).to_dict()
    print_json(job_payload)
    return 0


def handle_show_job_events(args: argparse.Namespace) -> int:
    """Show SQLite audit events for one job."""

    store = sqlite_store_from_args(args)
    store.get_job(args.job_id)
    print_json(store.list_events(args.job_id))
    return 0


def handle_show_job_attempts(args: argparse.Namespace) -> int:
    """Show SQLite worker attempts for one job."""

    store = sqlite_store_from_args(args)
    store.get_job(args.job_id)
    print_json(store.list_attempts(args.job_id))
    return 0


def sqlite_store_from_args(args: argparse.Namespace) -> SQLiteJobStore:
    """Build a SQLite job store from CLI args."""

    return SQLiteJobStore(Path(args.jobs_root), Path(args.sqlite_db))


def _job_store_from_args(args: argparse.Namespace) -> JobStoreProtocol:
    return build_job_store(
        APISettings(
            job_backend=args.job_backend,
            jobs_root=Path(args.jobs_root),
            sqlite_jobs_db_path=Path(args.sqlite_db),
        )
    )


def _add_job_store_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--job-backend",
        choices=("json", "sqlite"),
        default="json",
        help="Job backend to inspect",
    )
    _add_sqlite_job_store_args(parser)


def _add_sqlite_job_store_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--jobs-root",
        default="runs/jobs",
        help="Jobs root to read from",
    )
    parser.add_argument(
        "--sqlite-db",
        default="runs/jobs.sqlite3",
        help="SQLite jobs database to read when --job-backend sqlite",
    )
