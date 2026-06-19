# API Design

The MVP API is intentionally synchronous. It is designed to make later async job
execution straightforward without putting queue details into the trainer.

## Endpoints

### GET /health

Returns service health and version.

### GET /datasets

Returns registered datasets and catalog metadata.

### GET /models

Returns registered model names.

### GET /experiments

Returns local experiment summaries discovered under the fixed safe `runs` root.
The endpoint no longer accepts an arbitrary `root` query parameter for local
directory listing.

Response:

```json
{
  "experiments": [
    {
      "status": "complete",
      "experiment_name": "csv_forecast",
      "run_id": "20260620T000000Z_a1b2c3",
      "created_at": "2026-06-20T00:00:00+00:00",
      "run_dir": "runs/csv_forecast/latest",
      "checkpoint_path": "runs/csv_forecast/latest/checkpoint.pt",
      "test_metrics": {"original": {}}
    }
  ]
}
```

Runs with missing or damaged `results.json` are returned as
`status: incomplete` instead of crashing the endpoint.

### POST /experiments/train

Accepts a validated config payload and runs a synchronous demo training job.
Future iterations should move this endpoint to a durable background worker.

Request body is the same shape as a YAML/JSON training config:

```json
{
  "experiment": {"name": "api_demo", "output_dir": "runs", "overwrite": true},
  "data": {"name": "synthetic", "input_len": 6, "output_len": 2},
  "model": {"name": "linear"},
  "training": {"epochs": 1},
  "evaluation": {"metrics": ["mae", "mse"], "include_scaled_metrics": true}
}
```

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
- Runtime training failures should return clear HTTP 500 responses with a
  concise message and server-side logs.
