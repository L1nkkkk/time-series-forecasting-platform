# Artifact Manifests

Every completed train and compare run writes `artifacts.json` in the run
directory. The manifest is a small index of files produced by that run so APIs,
CLI commands, and downstream tools can discover outputs without guessing file
names.

The manifest is read-only metadata. The API returns the manifest only; it does
not provide arbitrary file download.

## Train Manifest

Train runs use this shape:

```json
{
  "run_type": "train",
  "experiment_name": "csv_forecast",
  "run_id": "20260619T120000Z_a1b2c3",
  "run_dir": "runs/csv_forecast/latest",
  "artifacts": [
    {
      "name": "results",
      "kind": "json",
      "path": "runs/csv_forecast/latest/results.json",
      "description": "Training result payload"
    },
    {
      "name": "checkpoint",
      "kind": "checkpoint",
      "path": "runs/csv_forecast/latest/checkpoint.pt",
      "description": "Final model checkpoint"
    },
    {
      "name": "config_snapshot",
      "kind": "yaml",
      "path": "runs/csv_forecast/latest/config_snapshot.yaml",
      "description": "Validated config snapshot"
    },
    {
      "name": "environment",
      "kind": "json",
      "path": "runs/csv_forecast/latest/environment.json",
      "description": "Runtime environment metadata"
    },
    {
      "name": "train_log",
      "kind": "log",
      "path": "runs/csv_forecast/latest/train.log",
      "description": "Training log"
    }
  ]
}
```

Optional missing files are skipped instead of failing the run.

## Compare Manifest

Compare parent runs use this shape:

```json
{
  "run_type": "compare",
  "experiment_name": "compare_forecast",
  "compare_run_id": "20260619T120000Z_a1b2c3",
  "compare_run_dir": "runs/compare_forecast/latest",
  "artifacts": [
    {
      "name": "results",
      "kind": "json",
      "path": "runs/compare_forecast/latest/results.json",
      "description": "Compare result payload"
    },
    {
      "name": "leaderboard_json",
      "kind": "json",
      "path": "runs/compare_forecast/latest/leaderboard.json",
      "description": "Leaderboard rows as JSON"
    },
    {
      "name": "leaderboard_csv",
      "kind": "csv",
      "path": "runs/compare_forecast/latest/leaderboard.csv",
      "description": "Leaderboard rows as CSV"
    },
    {
      "name": "compare_config_snapshot",
      "kind": "yaml",
      "path": "runs/compare_forecast/latest/compare_config_snapshot.yaml",
      "description": "Validated compare config snapshot"
    },
    {
      "name": "environment",
      "kind": "json",
      "path": "runs/compare_forecast/latest/environment.json",
      "description": "Runtime environment metadata"
    }
  ]
}
```

Compare manifests point to the parent compare outputs. Per-model train runs
under `models/<model_alias>/latest/` also write their own train manifests.

## Safety

Manifest builders verify each artifact path resolves inside the current run
directory before writing. `ExperimentStore` also validates `experiment_name` and
`run_id`, supports `latest`, and verifies resolved lookup paths remain under the
fixed runs root.

## Querying

CLI:

```bash
py -m ts_platform.cli.main show-artifacts --experiment compare_forecast --run latest
```

API:

```text
GET /experiments/{experiment_name}/{run_id}/artifacts
```
