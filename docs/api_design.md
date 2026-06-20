# API Design

The MVP API keeps the original synchronous train and compare endpoints for
simple demos, and also exposes a lightweight local jobs layer for asynchronous
submission. The jobs layer is intentionally local and demo-grade: it uses an
in-process `ThreadPoolExecutor` and defaults to JSON metadata under the fixed
runs root. Phase 8A adds an optional SQLite job metadata backend without
changing the `/jobs` API surface. It still does not introduce Redis, Celery,
RabbitMQ, Kubernetes, or a separate worker process.

## Endpoints

### GET /health

Returns service health and version.

### GET /datasets

Returns registered datasets and catalog metadata. At app startup,
`configs/datasets/*.yaml` is loaded when present. Missing catalog files are
ignored. Damaged catalog files are logged as warnings and skipped so the demo
API can still start.

### GET /models

Returns registered model names.

### GET /experiments

Returns local experiment summaries discovered through `ExperimentStore` under
the fixed safe `runs` root.
The endpoint no longer accepts an arbitrary `root` query parameter for local
directory listing.

Response:

```json
{
  "experiments": [
    {
      "status": "complete",
      "run_type": "train",
      "experiment_name": "csv_forecast",
      "run_id": "20260620T000000Z_a1b2c3",
      "created_at": "2026-06-20T00:00:00+00:00",
      "run_dir": "runs/csv_forecast/latest",
      "checkpoint_path": "runs/csv_forecast/latest/checkpoint.pt",
      "test_metrics": {"original": {}}
    },
    {
      "status": "complete",
      "run_type": "compare",
      "experiment_name": "compare_forecast",
      "run_id": "20260620T000000Z_d4e5f6",
      "compare_run_id": "20260620T000000Z_d4e5f6",
      "created_at": "2026-06-20T00:00:00+00:00",
      "run_dir": "runs/compare_forecast/latest",
      "compare_run_dir": "runs/compare_forecast/latest",
      "primary_metric": "mae",
      "success_count": 5,
      "failed_count": 0,
      "leaderboard_json_path": "runs/compare_forecast/latest/leaderboard.json",
      "leaderboard_csv_path": "runs/compare_forecast/latest/leaderboard.csv"
    }
  ]
}
```

Runs with missing or damaged `results.json` are returned as
`status: incomplete` and `run_type: unknown` instead of crashing the endpoint.

### GET /experiments/{experiment_name}/{run_id}/results

Returns the stored `results.json` for a train run or compare parent run.
`run_id` can be `latest` or a recorded `run_id` / `compare_run_id`.

Train results include checkpoint path, history, validation metrics, and test
metrics. Compare results include:

- `run_type: compare`
- `experiment_name`
- `compare_run_id`
- `compare_run_dir`
- `created_at`
- `leaderboard_json_path`
- `leaderboard_csv_path`
- `primary_metric`
- `success_count`
- `failed_count`
- `rows`

### GET /experiments/{experiment_name}/{run_id}/leaderboard

Returns a compare run `leaderboard.json` array. This endpoint is meaningful for
compare runs; train runs return 404 because they do not have a leaderboard
artifact. `model_params` is returned as an object in JSON responses.

### GET /experiments/{experiment_name}/{run_id}/artifacts

Returns the stored `artifacts.json` manifest for a train or compare run.
`run_id` can be `latest` or a recorded `run_id` / `compare_run_id`. This
endpoint returns only manifest metadata and does not download arbitrary files.

### GET /experiments/{experiment_name}/{run_id}/artifacts/{artifact_name}

Downloads one file registered in the run `artifacts.json` manifest.
`artifact_name` is a safe path component such as `results`,
`leaderboard_json`, `leaderboard_csv`, or `train_log`; it is not a file path.
The endpoint looks up the matching manifest entry by exact `name`, resolves the
manifest path under the fixed runs root, then verifies the path also remains
inside the physical run directory resolved by `ExperimentStore` for the
requested `experiment_name` and `run_id`. Manifest `run_dir` and
`compare_run_dir` fields are returned as metadata but are not trusted as the
authorization boundary. A manifest entry that points to another run under the
same runs root is rejected even if the manifest directory metadata was
tampered. After path checks, the endpoint checks kind policy and file size, and
returns a `FileResponse` with the artifact media type and filename.

Default downloadable kinds are:

- `json`: `application/json`
- `yaml`: `text/yaml`
- `csv`: `text/csv`
- `log`: `text/plain`

Checkpoint artifacts are blocked by default. They are model-state binaries and
may be much larger or more sensitive than result metadata, so enabling them
requires an explicit `APISettings.allow_checkpoint_download` change.
`APISettings.artifact_allowed_kinds` controls downloadable non-checkpoint kinds,
and `APISettings.artifact_max_bytes` controls the size limit. Files larger than
the configured limit are rejected before response creation. The endpoint ignores
arbitrary path query parameters; clients can only request a manifest
`artifact_name`.

### POST /experiments/compare

Accepts a validated `CompareConfig` payload and runs a synchronous demo compare
job through `CompareRunner`. Future iterations should move compare execution to
a durable background worker. The endpoint does not allow clients to choose
arbitrary output directories. Any request `experiment.output_dir` value is
ignored and overwritten with the API safe runs root, currently `runs`.

Response is the compare result payload written to the parent `results.json`.

### POST /experiments/train

Accepts a validated config payload and runs a synchronous demo training job.
Future iterations should move this endpoint to a durable background worker.
The endpoint does not allow clients to choose arbitrary output directories.
Any request `experiment.output_dir` value is ignored and overwritten with the
API safe runs root, currently `runs`. CLI training still honors
`experiment.output_dir`.

Request body is the same shape as a YAML/JSON training config:

```json
{
  "experiment": {"name": "api_demo", "output_dir": "../../ignored", "overwrite": true},
  "data": {"name": "synthetic", "input_len": 6, "output_len": 2},
  "model": {"name": "linear"},
  "training": {"epochs": 1},
  "evaluation": {"metrics": ["mae", "mse"], "include_scaled_metrics": true}
}
```

The run is still written under `runs/api_demo/...`.

Response includes run metadata, artifact paths, history, and metrics:

```json
{
  "run_id": "20260620T000000Z_a1b2c3",
  "created_at": "2026-06-20T00:00:00+00:00",
  "run_dir": "runs/api_demo/latest",
  "experiment_name": "api_demo",
  "checkpoint_path": "runs/api_demo/latest/checkpoint.pt",
  "validation_metrics": {"original": {}, "scaled": {}},
  "test_metrics": {"original": {}, "scaled": {}}
}
```

Metrics under `original` are the default authoritative values. `scaled` is
present only when `evaluation.include_scaled_metrics` is true.

### POST /jobs/train

Accepts the same `PlatformConfig` request body as `POST /experiments/train`,
creates a local job under `runs/jobs/<job_id>/`, and returns immediately with a
`JobRecord`. The background job calls the same API training service, so
`experiment.output_dir` is still overwritten with the safe API runs root.
With the default JSON backend, job metadata is stored in
`runs/jobs/<job_id>/job.json`. With the SQLite backend, metadata is stored in
`runs/jobs.sqlite3` and the request snapshot still lives under
`runs/jobs/<job_id>/request_config.json`.

### POST /jobs/compare

Accepts the same `CompareConfig` request body as `POST /experiments/compare`,
creates a local compare job, and returns immediately with a `JobRecord`. The
background job delegates to the same safe compare service.

### GET /jobs

Returns persisted job records newest first by `created_at`. Corrupt `job.json`
metadata is skipped so a single damaged job does not prevent other jobs from
being listed. SQLite backend reads use the same response shape and return clear
job-store errors for SQLite database or schema failures.

```json
{
  "jobs": [
    {
      "job_id": "20260619T120000Z_a1b2c3",
      "job_type": "train",
      "status": "succeeded",
      "experiment_name": "api_train",
      "run_id": "20260619T120001Z_d4e5f6",
      "compare_run_id": null,
      "result_path": "runs/api_train/latest/results.json",
      "leaderboard_json_path": null,
      "artifacts_path": "runs/api_train/latest/artifacts.json",
      "error": null
    }
  ]
}
```

### GET /jobs/{job_id}

Returns one persisted job record. `job_id` must be a safe path component.
Corrupt metadata for the requested job returns HTTP 500 and requires manual
cleanup.

### GET /jobs/{job_id}/result

Returns the saved `results.json` for a succeeded job. Jobs that are still
`queued`, `running`, `cancel_requested`, `cancelled`, or `failed` return HTTP
409 with a detail payload containing the current status and error field. Missing
job ids return 404 and unsafe ids return 400.

### GET /jobs/{job_id}/logs

Returns a JSON wrapper with `job_id`, `status`, `error`, `log_path`, and `log`.
When the completed run has a `train.log`, the log text is included. Compare
parent runs may not have a parent log, so `log` can be `null`; failures are
still visible through the job `error` field.

### POST /jobs/{job_id}/cancel

Requests local cancellation. A queued job becomes `cancelled`. A running job
becomes `cancel_requested`; the platform does not force-kill the Python thread,
so the underlying training or compare call may still complete and record a
successful result. Terminal jobs return their current record unchanged.

Job statuses are:

- `queued`
- `running`
- `succeeded`
- `failed`
- `cancel_requested`
- `cancelled`

The FastAPI app closes the local JobRunner executor during lifespan shutdown
and resets the lazy singleton. A subsequent jobs request can create a fresh
runner, but interrupted running jobs are not recovered.

`APISettings.job_backend` selects the store backend. The default value is
`"json"`, preserving existing behavior. Setting it to `"sqlite"` builds
`SQLiteJobStore` through the job store factory. Unsupported values are rejected
with a clear backend error.

## Future Durable Jobs API Compatibility

The existing `/jobs` API is the stable external shape for asynchronous work.
The current runner is still a local `ThreadPoolExecutor`, and the backend can
be JSON or SQLite behind the same service boundary. Future implementations may
add a separate worker, Redis/RQ, or Celery after the SQLite boundary is proven.

Backend changes should preserve these endpoints:

- `POST /jobs/train`
- `POST /jobs/compare`
- `GET /jobs`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/result`
- `GET /jobs/{job_id}/logs`
- `POST /jobs/{job_id}/cancel`

Response shapes should remain compatible whenever possible. New durable queue
fields such as attempt id, worker id, heartbeat timestamp, retry count, or event
links should be added in a backward-compatible way. Existing fields such as
`job_id`, `job_type`, `status`, `experiment_name`, `run_id`,
`compare_run_id`, `result_path`, `artifacts_path`, `leaderboard_json_path`, and
`error` should keep their meaning.

## Error Handling

- Invalid configs return HTTP 422 through Pydantic validation.
- API training and compare override unsafe output roots instead of returning
  HTTP 400.
- Invalid `experiment_name` or `run_id` lookup path components return HTTP 400.
- Invalid artifact download `artifact_name` path components return HTTP 400.
- Artifact download paths that escape the current run directory return HTTP
  400.
- Invalid `job_id` path components return HTTP 400.
- Missing `results.json`, `leaderboard.json`, or `artifacts.json` artifacts
  return HTTP 404.
- Missing registered artifact names or missing artifact files return HTTP 404.
- Forbidden artifact kinds, including default checkpoint downloads, return HTTP
  403.
- Artifact files larger than the configured download limit return HTTP 413.
- Missing jobs return HTTP 404.
- Damaged result, leaderboard, or artifact manifest JSON returns HTTP 500.
- Damaged artifact manifest entries return HTTP 500.
- Corrupt `job.json` records are skipped by `GET /jobs`, but direct
  `GET /jobs/{job_id}` lookup returns HTTP 500.
- Job result lookup returns HTTP 409 until the job has succeeded, and also
  returns HTTP 409 for failed or cancelled jobs.
- Runtime training or compare failures should return clear HTTP 500 responses
  with a concise message and server-side logs.

## Run ID Format

`ExperimentRecorder.run_id` currently uses
`YYYYMMDDTHHMMSSZ_<6 hex chars>`, for example
`20260619T120000Z_a1b2c3`. Compare parent ids use the same format and are
reported as `compare_run_id`. API clients may pass either `latest` or a recorded
id to lookup endpoints.

## ExperimentStore

`api/services/experiment_store.py` centralizes result lookup instead of putting
filesystem logic in route functions. It validates `experiment_name` and
`run_id` with the same safe path component rules used by experiment configs,
resolves every candidate path, and verifies the resolved path remains inside the
fixed runs root.

The store supports direct directory lookup, `latest`, and lookup by recorded
`run_id` / `compare_run_id` when the physical directory is `latest`.
`resolve_run()` exposes the same physical run directory resolution to callers
that need a stable boundary. It reads `results.json`, `leaderboard.json`, and
`artifacts.json` with the same fixed-root safety checks.

`runs/jobs/<job_id>` is internal job metadata and is skipped by experiment
listing so job records do not appear as incomplete experiment runs.
