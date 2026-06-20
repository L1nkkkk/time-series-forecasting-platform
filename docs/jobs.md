# Local Jobs

Phase 5 adds a lightweight local job layer for API-driven train and compare
work. It is designed for demos and tests, not production scheduling. Phase 8A
adds an optional SQLite-backed job store prototype while keeping the JSON store
as the default backend. Phase 8B adds a local `worker-once` process prototype
that can claim and execute queued SQLite jobs outside the API process. Phase
8C exposes SQLite events and attempts for inspection, adds a finite
`worker-loop` CLI, and records minimal worker heartbeats. Phase 8D adds
explicit retry and timeout state transitions for the SQLite prototype without
adding an automatic scheduler.

## Storage Layout

The default backend is still the JSON job store. Jobs are stored under the API
runs root:

```text
runs/
  jobs/
    20260619T120000Z_a1b2c3/
      job.json
      request_config.json
```

`request_config.json` stores the validated request payload. `job.json` stores
the current job metadata and is updated as the local runner progresses.

`runs/jobs/<job_id>` is internal metadata. `ExperimentStore.list_experiments()`
skips it so jobs do not appear as incomplete experiment runs.

The optional SQLite backend stores job metadata in:

```text
runs/jobs.sqlite3
```

It still writes `request_config.json` under `runs/jobs/<job_id>/` so request
payload snapshots remain inspectable on disk. It does not write a compatibility
`job.json` copy; the SQLite `jobs` table is the source of truth for metadata.

| Backend | Default | Metadata source | Request snapshot | Notes |
| --- | --- | --- | --- | --- |
| JSON | Yes | `runs/jobs/<job_id>/job.json` | `runs/jobs/<job_id>/request_config.json` | Simple local store used by existing behavior and tests |
| SQLite | No | `runs/jobs.sqlite3` table `jobs` | `runs/jobs/<job_id>/request_config.json` | Phase 8A/8B prototype for durable metadata, audit events, attempts, and local worker claiming |

## Job Ids

Job ids use the same visible shape as run ids:

```text
YYYYMMDDTHHMMSSZ_<6 hex chars>
```

Example:

```text
20260619T120000Z_a1b2c3
```

The id must also be a safe path component. Path separators, whitespace, `..`,
absolute paths, and path escapes are rejected.

## Job Record Schema

Each `job.json` contains:

```json
{
  "job_id": "20260619T120000Z_a1b2c3",
  "job_type": "train",
  "status": "queued",
  "created_at": "2026-06-19T12:00:00+00:00",
  "updated_at": "2026-06-19T12:00:00+00:00",
  "started_at": null,
  "finished_at": null,
  "experiment_name": "api_train",
  "run_id": null,
  "compare_run_id": null,
  "result_path": null,
  "leaderboard_json_path": null,
  "artifacts_path": null,
  "error": null,
  "config_snapshot_path": "runs/jobs/20260619T120000Z_a1b2c3/request_config.json"
}
```

Compare jobs use `job_type: compare` and populate `compare_run_id` plus
`leaderboard_json_path` after success.

## Status Transitions

Supported statuses:

- `queued`
- `running`
- `succeeded`
- `failed`
- `cancel_requested`
- `cancelled`
- `retrying`
- `timed_out`

Normal success path:

```text
queued -> running -> succeeded
```

Failure path:

```text
queued -> running -> failed
```

Explicit timeout path:

```text
running -> timed_out
cancel_requested -> timed_out
```

Explicit retry path:

```text
failed -> retrying -> queued
timed_out -> retrying -> queued
cancelled -> retrying -> queued
```

Cancellation:

```text
queued -> cancelled
running -> cancel_requested
```

Running cancellation is best effort. The local runner does not force-kill a
Python thread, so a job marked `cancel_requested` may still finish and record a
succeeded or failed result.

`timed_out` is assigned only by an explicit API, CLI, or store call after stale
inspection. `retrying` is a short-lived transition recorded while a retryable
terminal job is requeued; the final visible state after retry is `queued`.

## Runner

`JobRunner` uses `ThreadPoolExecutor` with `max_workers=1` by default. It
creates the job through a `JobStoreProtocol`, submits work to the executor, and
delegates the actual execution to:

- `train_with_safe_output_dir` for train jobs
- `compare_with_safe_output_dir` for compare jobs

Both services overwrite `experiment.output_dir` with the safe API runs root,
so API callers cannot choose arbitrary output locations.

`JsonJobStore` implements the default JSON backend. `JobStore` remains a
backward-compatible alias for `JsonJobStore`. `SQLiteJobStore` implements the
same protocol and can be injected into `JobRunner` or selected through
`APISettings.job_backend = "sqlite"`.

## Execution Modes

`APISettings.job_execution_mode` controls how submitted API jobs are executed:

- `in_process` is the default. `POST /jobs/train` and `POST /jobs/compare`
  create a job and immediately submit it to the in-process `ThreadPoolExecutor`.
- `external_worker` requires `job_backend = "sqlite"`. The API creates only a
  `queued` job record and request snapshot. A separate worker process must
  claim and execute the job.

The `/jobs` endpoint surface and `JobRecord` response shape are unchanged in
both modes.

## Worker

`JobWorker` is the Phase 8B local worker prototype. It calls
`SQLiteJobStore.claim_next_queued_job(worker_id=...)`, reads
`request_config.json`, validates it as `PlatformConfig` or `CompareConfig`,
then delegates to the same safe API execution services:

- `train_with_safe_output_dir`
- `compare_with_safe_output_dir`

The worker records a `job_attempts` row on claim, marks the attempt
`succeeded` or `failed`, and updates the job to `succeeded` or `failed`.
`JobWorker.run_once()` records a heartbeat immediately after claim and again
before a job is marked succeeded or failed. This is a minimal activity marker,
not a periodic monitor for long-running training calls.

`worker-once` processes exactly one job and exits. `worker-loop` repeatedly
calls the same worker path with finite bounds such as `--max-jobs` and
`--max-idle-cycles`, then exits with a JSON summary. It is not a daemon and
does not implement retries, timeout handling, stale recovery, signal handling,
or process supervision.

## Lifecycle

The FastAPI app registers a lifespan shutdown hook that calls
`shutdown_job_runner()`. Shutdown closes the local executor with `wait=False`
and resets the module-level runner singleton to `None`. A later jobs request can
create a fresh `JobRunner`.

This cleanup is process-local. If the API process stops while a job is running,
the platform does not recover or resume that interrupted job on restart.

## API Endpoints

```text
POST /jobs/train
POST /jobs/compare
GET /jobs
GET /jobs/stale
GET /jobs/{job_id}
GET /jobs/{job_id}/events
GET /jobs/{job_id}/attempts
GET /jobs/{job_id}/result
GET /jobs/{job_id}/logs
POST /jobs/{job_id}/cancel
POST /jobs/{job_id}/timeout
POST /jobs/{job_id}/retry
```

`POST /jobs/train` and `POST /jobs/compare` return immediately with a
`JobRecord`. They do not wait for training or compare completion.

`GET /jobs/{job_id}/result` returns `results.json` only when the job status is
`succeeded`. For unfinished, failed, or cancelled jobs it returns HTTP 409 with
the current status and error field.

`GET /jobs/{job_id}/logs` returns a JSON wrapper with job status, error, and
the `train.log` text when a completed train run exposes one. Compare parent
runs may not have a parent log.

Unsafe job ids return HTTP 400. Missing jobs return HTTP 404.

`GET /jobs/{job_id}/events` and `GET /jobs/{job_id}/attempts` are SQLite-only
observability endpoints. They return JSON arrays of event or attempt rows. If
the configured backend is JSON, they return HTTP 400 with a clear
`job events require sqlite backend` or `job attempts require sqlite backend`
message.

`GET /jobs/stale?older_than_seconds=3600` is SQLite-only and lists stale
`running` or `cancel_requested` jobs without changing their status. Non-positive
thresholds return HTTP 400.

`POST /jobs/{job_id}/timeout` is SQLite-only and explicitly marks a
`running` or `cancel_requested` job as `timed_out`. Terminal jobs are returned
unchanged. The latest running attempt, when present, is marked failed with the
timeout reason.

`POST /jobs/{job_id}/retry?max_attempts=3` is SQLite-only and requeues
`failed`, `timed_out`, or `cancelled` jobs. It rejects `queued`, `running`,
`cancel_requested`, and `succeeded` jobs with HTTP 409. It also returns HTTP 409
when the existing attempt count is already at `max_attempts`.

Corrupt `job.json` metadata is skipped by `GET /jobs` so one damaged record
does not hide other jobs. Reading the corrupt job directly with
`GET /jobs/{job_id}` returns HTTP 500 and the metadata should be cleaned up
manually.

When the SQLite backend is selected, the `/jobs` response shape and status
semantics are unchanged. SQLite schema or database errors are returned as job
store errors instead of silently falling back to JSON.

## SQLite Events

The SQLite backend writes a minimal audit trail to `job_events`:

- `job_created`
- `job_running`
- `job_succeeded`
- `job_failed`
- `job_cancelled`
- `cancel_requested`
- `job_claimed`
- `heartbeat`
- `attempt_succeeded`
- `attempt_failed`
- `job_timed_out`
- `job_retrying`
- `job_requeued`

Events are exposed through the SQLite-only API endpoint and the
`show-job-events` CLI command. This is intentionally small: it records
lifecycle transitions, local attempts, explicit timeouts, and explicit retries,
but does not add automatic retry scheduling.

`job_attempts` stores:

- `attempt_id`
- `job_id`
- `status`
- `worker_id`
- `started_at`
- `finished_at`
- `heartbeat_at`
- `error`

Attempts are exposed through the SQLite-only API endpoint and the
`show-job-attempts` CLI command.

`SQLiteJobStore.list_stale_running_jobs(older_than_seconds=...)` can inspect
jobs in `running` or `cancel_requested` status whose latest activity timestamp
is older than a threshold. It prefers the latest attempt `heartbeat_at` and
falls back to the job `updated_at` when there is no attempt. The method is
read-only; it does not reset, retry, fail, or cancel stale jobs.

`SQLiteJobStore.mark_stale_running_jobs_timed_out(...)` calls the same stale
inspection and explicitly marks those jobs `timed_out`. It does not retry them.
`SQLiteJobStore.retry_job(...)` requeues retryable terminal jobs when the
configured `RetryPolicy.max_attempts` has not been reached. `RetryPolicy`
defaults to `max_attempts=3` and `stale_after_seconds=3600`.

## CLI

The CLI supports read-only job inspection:

```bash
py -m ts_platform.cli.main list-jobs
py -m ts_platform.cli.main show-job --job-id 20260619T120000Z_a1b2c3
py -m ts_platform.cli.main list-jobs --job-backend sqlite --sqlite-db runs/jobs.sqlite3
py -m ts_platform.cli.main show-job --job-backend sqlite --sqlite-db runs/jobs.sqlite3 --job-id 20260619T120000Z_a1b2c3
py -m ts_platform.cli.main show-job-events --sqlite-db runs/jobs.sqlite3 --jobs-root runs/jobs --job-id 20260619T120000Z_a1b2c3
py -m ts_platform.cli.main show-job-attempts --sqlite-db runs/jobs.sqlite3 --jobs-root runs/jobs --job-id 20260619T120000Z_a1b2c3
py -m ts_platform.cli.main list-stale-jobs --sqlite-db runs/jobs.sqlite3 --jobs-root runs/jobs --older-than-seconds 3600
py -m ts_platform.cli.main mark-stale-timeout --sqlite-db runs/jobs.sqlite3 --jobs-root runs/jobs --older-than-seconds 3600
py -m ts_platform.cli.main retry-job --sqlite-db runs/jobs.sqlite3 --jobs-root runs/jobs --job-id 20260619T120000Z_a1b2c3 --max-attempts 3
py -m ts_platform.cli.main worker-once --sqlite-db runs/jobs.sqlite3 --jobs-root runs/jobs --runs-root runs --worker-id local_worker
py -m ts_platform.cli.main worker-loop --sqlite-db runs/jobs.sqlite3 --jobs-root runs/jobs --runs-root runs --worker-id local_worker --max-jobs 1 --max-idle-cycles 1 --sleep-seconds 1.0
```

CLI job submission is intentionally not implemented. A CLI command is a
one-shot process, so an in-process `ThreadPoolExecutor` would not continue
running after the command exits. `worker-once` is an execution command for
already queued SQLite jobs; it is not a general submit command. `worker-loop`
is the finite local polling variant for SQLite jobs and prints
`worker_id`, `processed`, `idle_cycles`, and processed job records as JSON.
The retry and timeout commands are explicit SQLite maintenance commands; they
do not run queued jobs.

## Limitations

- Jobs are local to one API process.
- Jobs are not recovered if the process stops mid-run.
- There is no automatic retry scheduler.
- There is no unbounded daemon worker loop.
- There is no automatic timeout sweep or stale-heartbeat recovery loop.
- Worker heartbeat recording is minimal and only happens at claim, success, and
  failure boundaries.
- Running threads are not force-killed by cancellation.
- Shutdown closes the local executor but does not resume interrupted running
  work.
- The default job metadata backend is JSON on local disk.
- The SQLite backend makes metadata durable in one local database file, but it
  does not by itself provide full worker crash recovery or restart-safe
  execution.

Future production hardening should move execution to a durable worker or queue,
add heartbeat/retry semantics, make cancellation cooperative at the runner
boundary, and add artifact download controls separately from metadata lookup.
