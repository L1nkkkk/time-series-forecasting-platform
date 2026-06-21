"""Local SQLite worker commands."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ts_platform.api.jobs.sqlite_store import SQLiteJobStore
from ts_platform.api.jobs.worker import JobWorker
from ts_platform.cli.utils import print_json


def register(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register local worker commands."""

    worker_once_parser = subparsers.add_parser(
        "worker-once",
        help="Claim and run one queued SQLite job",
    )
    _add_worker_store_args(worker_once_parser)
    worker_once_parser.set_defaults(handler=handle_worker_once)

    worker_loop_parser = subparsers.add_parser(
        "worker-loop",
        help="Claim and run queued SQLite jobs with finite local bounds",
    )
    _add_worker_store_args(worker_loop_parser)
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
    worker_loop_parser.set_defaults(handler=handle_worker_loop, parser=worker_loop_parser)


def handle_worker_once(args: argparse.Namespace) -> int:
    """Claim and run one queued SQLite job."""

    worker = _worker_from_args(args)
    job = worker.run_once()
    payload = {"status": "idle"} if job is None else job.to_dict()
    print_json(payload)
    return 0


def handle_worker_loop(args: argparse.Namespace) -> int:
    """Claim and run queued SQLite jobs with finite local bounds."""

    if args.max_jobs < 1:
        args.parser.error("--max-jobs must be >= 1")
    if args.max_idle_cycles < 1:
        args.parser.error("--max-idle-cycles must be >= 1")
    if args.sleep_seconds < 0:
        args.parser.error("--sleep-seconds must be >= 0")
    worker = _worker_from_args(args)
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
    print_json(payload)
    return 0


def _worker_from_args(args: argparse.Namespace) -> JobWorker:
    store = SQLiteJobStore(Path(args.jobs_root), Path(args.sqlite_db))
    return JobWorker(
        store=store,
        runs_root=Path(args.runs_root),
        worker_id=args.worker_id,
    )


def _add_worker_store_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sqlite-db",
        default="runs/jobs.sqlite3",
        help="SQLite jobs database to claim from",
    )
    parser.add_argument(
        "--jobs-root",
        default="runs/jobs",
        help="Jobs root containing request snapshots",
    )
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Runs root to write train or compare results",
    )
    parser.add_argument(
        "--worker-id",
        default="local_worker",
        help="Safe worker id recorded in job attempts",
    )
