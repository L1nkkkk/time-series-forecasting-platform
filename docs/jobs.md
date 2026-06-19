# Local Jobs

Phase 5 adds a lightweight local job layer for API-driven train and compare
work. It is designed for demos and tests, not production scheduling.

## Storage Layout

Jobs are stored under the API runs root:

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
creates the job through `JobStore`, submits work to the executor, and delegates
the actual execution to:

- `train_with_safe_output_dir` for train jobs
- `compare_with_safe_output_dir` for compare jobs

Both services overwrite `experiment.output_dir` with the safe API runs root,
so API callers cannot choose arbitrary output locations.

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

## CLI

The CLI supports read-only job inspection:

```bash
py -m ts_platform.cli.main list-jobs
py -m ts_platform.cli.main show-job --job-id 20260619T120000Z_a1b2c3
```

CLI job submission is intentionally not implemented. A CLI command is a
one-shot process, so an in-process `ThreadPoolExecutor` would not continue
running after the command exits.

## Limitations

- Jobs are local to one API process.
- Jobs are not recovered if the process stops mid-run.
- There is no retry scheduler.
- Running threads are not force-killed by cancellation.
- Job metadata is JSON on local disk, not a database.

Future production hardening should move execution to a durable worker or queue,
add heartbeat/retry semantics, make cancellation cooperative at the runner
boundary, and add artifact download controls separately from metadata lookup.
