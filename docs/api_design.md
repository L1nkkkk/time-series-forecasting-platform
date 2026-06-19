# API Design

The MVP API is intentionally synchronous. It is designed to make later async job
execution straightforward without putting queue details into the trainer.

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

## Error Handling

- Invalid configs return HTTP 422 through Pydantic validation.
- API training and compare override unsafe output roots instead of returning
  HTTP 400.
- Invalid `experiment_name` or `run_id` lookup path components return HTTP 400.
- Missing `results.json` or `leaderboard.json` artifacts return HTTP 404.
- Damaged result or leaderboard JSON returns HTTP 500.
- Runtime training or compare failures should return clear HTTP 500 responses
  with a concise message and server-side logs.

## ExperimentStore

`api/services/experiment_store.py` centralizes result lookup instead of putting
filesystem logic in route functions. It validates `experiment_name` and
`run_id` with the same safe path component rules used by experiment configs,
resolves every candidate path, and verifies the resolved path remains inside the
fixed runs root.

The store supports direct directory lookup, `latest`, and lookup by recorded
`run_id` / `compare_run_id` when the physical directory is `latest`.
