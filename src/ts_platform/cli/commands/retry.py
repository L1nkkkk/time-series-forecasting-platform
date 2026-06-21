"""SQLite retry and timeout maintenance commands."""

from __future__ import annotations

import argparse

from ts_platform.api.jobs.retry import RetryPolicy
from ts_platform.cli.commands.jobs import sqlite_store_from_args
from ts_platform.cli.utils import print_json


def register(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register SQLite retry and timeout commands."""

    list_stale_jobs_parser = subparsers.add_parser(
        "list-stale-jobs",
        help="List stale running SQLite jobs without changing state",
    )
    _add_stale_args(list_stale_jobs_parser)
    list_stale_jobs_parser.set_defaults(
        handler=handle_list_stale_jobs, parser=list_stale_jobs_parser
    )

    mark_stale_timeout_parser = subparsers.add_parser(
        "mark-stale-timeout",
        help="Mark stale running SQLite jobs timed out",
    )
    _add_stale_args(mark_stale_timeout_parser)
    mark_stale_timeout_parser.add_argument(
        "--reason",
        help="Optional timeout reason to store on each job",
    )
    mark_stale_timeout_parser.set_defaults(
        handler=handle_mark_stale_timeout,
        parser=mark_stale_timeout_parser,
    )

    retry_job_parser = subparsers.add_parser(
        "retry-job",
        help="Requeue a failed, timed out, or cancelled SQLite job",
    )
    retry_job_parser.add_argument("--job-id", required=True, help="Job id to retry")
    _add_sqlite_job_store_args(retry_job_parser)
    retry_job_parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum allowed attempts before rejecting retry",
    )
    retry_job_parser.set_defaults(handler=handle_retry_job, parser=retry_job_parser)


def handle_list_stale_jobs(args: argparse.Namespace) -> int:
    """List stale running SQLite jobs without changing state."""

    if args.older_than_seconds <= 0:
        args.parser.error("--older-than-seconds must be > 0")
    store = sqlite_store_from_args(args)
    jobs_payload = [
        job.to_dict()
        for job in store.list_stale_running_jobs(
            older_than_seconds=args.older_than_seconds,
        )
    ]
    print_json({"jobs": jobs_payload})
    return 0


def handle_mark_stale_timeout(args: argparse.Namespace) -> int:
    """Mark stale running SQLite jobs timed out."""

    if args.older_than_seconds <= 0:
        args.parser.error("--older-than-seconds must be > 0")
    store = sqlite_store_from_args(args)
    timed_out_payload = [
        job.to_dict()
        for job in store.mark_stale_running_jobs_timed_out(
            older_than_seconds=args.older_than_seconds,
            reason=args.reason,
        )
    ]
    print_json({"timed_out": timed_out_payload})
    return 0


def handle_retry_job(args: argparse.Namespace) -> int:
    """Requeue a failed, timed out, or cancelled SQLite job."""

    if args.max_attempts < 1:
        args.parser.error("--max-attempts must be >= 1")
    store = sqlite_store_from_args(args)
    policy = RetryPolicy(max_attempts=args.max_attempts)
    job_payload = store.retry_job(args.job_id, policy=policy).to_dict()
    print_json(job_payload)
    return 0


def _add_stale_args(parser: argparse.ArgumentParser) -> None:
    _add_sqlite_job_store_args(parser)
    parser.add_argument(
        "--older-than-seconds",
        type=int,
        default=3600,
        help="Activity age threshold for stale running jobs",
    )


def _add_sqlite_job_store_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sqlite-db",
        default="runs/jobs.sqlite3",
        help="SQLite jobs database to inspect",
    )
    parser.add_argument(
        "--jobs-root",
        default="runs/jobs",
        help="Jobs root containing request snapshots",
    )
