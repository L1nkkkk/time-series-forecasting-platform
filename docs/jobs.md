# Local Jobs

Phase 5 adds a lightweight local job layer for API-driven train and compare
work. It is designed for demos and tests, not production scheduling. Phase 8A
adds an optional SQLite-backed job store prototype while keeping the JSON store
as the default backend. Phase 8B adds a local `worker-once` process prototype
that can claim and execute queued SQLite jobs outside the API process.

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

Normal success path:

```text
queued -> running -> succeeded
```

Failure path:

```text
queued -> running -> failed
```

Cancellation:

```text
queued -> cancelled
running -> cancel_requested
```

Running cancellation is best effort. The local runner does not force-kill a
Python thread, so a job marked `cancel_requested` may still finish and record a
succeeded or failed result.

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
`worker-once` processes exactly one job and exits. It is not a daemon and does
not implement polling loops, retries, timeout handling, or process supervision.

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
GET /jobs/{job_id}
GET /jobs/{job_id}/result
GET /jobs/{job_id}/logs
POST /jobs/{job_id}/cancel
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

Events are not exposed through a public API endpoint in Phase 8A/8B. They are
available through the store for tests and local debugging. This is intentionally
small: it records lifecycle transitions and local attempts, but does not yet
add retries, timeouts, stale heartbeat handling, or recovery semantics.

`job_attempts` stores:

- `attempt_id`
- `job_id`
- `status`
- `worker_id`
- `started_at`
- `finished_at`
- `heartbeat_at`
- `error`

## CLI

The CLI supports read-only job inspection:

```bash
py -m ts_platform.cli.main list-jobs
py -m ts_platform.cli.main show-job --job-id 20260619T120000Z_a1b2c3
py -m ts_platform.cli.main list-jobs --job-backend sqlite --sqlite-db runs/jobs.sqlite3
py -m ts_platform.cli.main show-job --job-backend sqlite --sqlite-db runs/jobs.sqlite3 --job-id 20260619T120000Z_a1b2c3
py -m ts_platform.cli.main worker-once --sqlite-db runs/jobs.sqlite3 --jobs-root runs/jobs --runs-root runs --worker-id local_worker
```

CLI job submission is intentionally not implemented. A CLI command is a
one-shot process, so an in-process `ThreadPoolExecutor` would not continue
running after the command exits. `worker-once` is an execution command for
already queued SQLite jobs; it is not a general submit command.

## Limitations

- Jobs are local to one API process.
- Jobs are not recovered if the process stops mid-run.
- There is no retry scheduler.
- There is no daemon worker loop.
- There is no timeout or stale-heartbeat recovery.
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
