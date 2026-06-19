# Time Series Forecasting Platform

This repository contains a configuration-driven MVP for a comprehensive time
series forecasting platform. It follows the same engineering ideas highlighted
by BasicTS: datasets, scalers, models, metrics, runners, and configs are
separate modules with small public interfaces and registry-based extension
points.

The current MVP focuses on a runnable local training loop:

- Synthetic forecasting dataset.
- Local CSV forecasting dataset with time-based splits.
- Naive last-value, moving-average, seasonal-naive, linear, and MLP forecasting
  models.
- Standard and min-max scalers.
- MAE, MSE, RMSE, MAPE, and WAPE metrics.
- Config snapshots, checkpoints, metrics output, and environment metadata.
- CLI and a synchronous FastAPI demo API.
- Original-scale evaluation metrics with optional scaled-space metrics.
- Versioned checkpoints that can restore model, scaler, and optimizer state.
- Strict CSV parameter validation, split-local missing-value handling, and
  dataset catalog discovery.
- Multi-model compare runs with `leaderboard.json` and `leaderboard.csv`.

No BasicTS code is copied into this project.

## Installation

Use Python 3.10 or newer.

```bash
py -m pip install -e ".[dev]"
```

On Unix-like shells, replace `py` with `python`.

## Quick Start

Run the example training config:

```bash
py -m ts_platform.cli.main train --config configs/examples/simple_forecast.yaml
```

The run writes logs, a checkpoint, a config snapshot, environment metadata, and
`results.json` under `runs/simple_forecast/latest/` because the example config
sets `overwrite: true`.

Run the CSV example:

```bash
py -m ts_platform.cli.main train --config configs/examples/csv_forecast.yaml
```

Run a multi-model compare:

```bash
py -m ts_platform.cli.main compare --config configs/examples/compare_forecast.yaml
```

The compare command writes one normal Trainer run per model under
`runs/compare_forecast/latest/models/` and produces
`leaderboard.json` plus `leaderboard.csv` in the compare run directory.

## Metrics

Validation and test metrics are reported on the original data scale by default.
Scaled-space metrics are not saved by default. When
`evaluation.include_scaled_metrics: true` is set, `results.json` also stores
scaled-space metrics:

```json
{
  "test_metrics": {
    "original": {"mae": 0.0},
    "scaled": {"mae": 0.0}
  }
}
```

## Configuration

Training is driven by YAML or JSON. The example config declares:

- `experiment`: run name, output directory, seed, overwrite behavior.
- `data`: dataset name, split ratios, sequence lengths, scaler, and dataset
  parameters.
- `model`: registered model name and model-specific parameters.
- `training`: epochs, learning rate, optimizer, device, and checkpoint policy.
- `evaluation`: metric names and whether to include scaled-space metrics.

See [configs/examples/simple_forecast.yaml](configs/examples/simple_forecast.yaml)
and [configs/examples/csv_forecast.yaml](configs/examples/csv_forecast.yaml).

`experiment.name` must be a safe path component containing only letters,
numbers, `_`, `-`, and `.`. Path separators, whitespace, `..`, absolute paths,
and names longer than 80 characters are rejected. `ExperimentRecorder` also
checks resolved paths so a run directory cannot escape the configured root.

## CSV Datasets

Use `data.name: csv` to train on a local CSV file:

```yaml
data:
  name: csv
  input_len: 8
  output_len: 2
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  params:
    path: tests/fixtures/tiny_series.csv
    timestamp_col: timestamp
    target_cols: [value]
    missing_policy: error
    sort_by_time: true
```

CSV parameters:

- `path`: local CSV path.
- `timestamp_col`: optional timestamp column. When present it is parsed as
  datetime, duplicate timestamps are rejected, and `sort_by_time: true` sorts
  rows before splitting.
- `target_cols`: one or more numeric target columns. Models receive only these
  columns in this phase. A plain string or empty list is rejected.
- `missing_policy`: `error`, `drop`, `forward_fill`, or `zero_fill`.
- `sort_by_time`: boolean flag to sort rows by timestamp before splitting.

CSV data uses time-based splitting: raw rows are split into train, validation,
and test periods first, then sliding windows are generated within each split.
This prevents validation/test rows from entering training windows. Scalers are
fit only from the training split via `train_dataset.scaler_fit_values()`.
Missing-value policies are also applied after the raw-row split and only inside
the selected split. For example, `forward_fill` cannot propagate the final train
row into the first validation row, and `drop` cannot change another split's row
boundaries.

Exogenous `feature_cols` are intentionally not supported yet; passing them
raises a clear error. That scope is deferred to a later phase.

Dataset catalog files such as
[configs/datasets/local_csv.yaml](configs/datasets/local_csv.yaml) describe
local datasets for discovery. They are metadata only; training still uses an
explicit experiment config. The API loads `configs/datasets/*.yaml` on startup
when present. The CLI can load a catalog explicitly:

```bash
py -m ts_platform.cli.main list-datasets --catalog configs/datasets/local_csv.yaml
```

Registering catalog metadata with an existing name overwrites the previous
metadata entry.

## Discovery Commands

List datasets and models as JSON for scripts:

```bash
py -m ts_platform.cli.main list-datasets
py -m ts_platform.cli.main list-models
```

## Compare Runs

Compare configs reuse the existing data, training, and evaluation config
sections, but replace `model` with a `models` list:

```yaml
models:
  - name: naive
  - name: moving_average
    params:
      window_size: 4
  - name: seasonal_naive
    params:
      season_length: 7
  - name: linear

primary_metric: mae
continue_on_error: true
```

`primary_metric` must be one of `evaluation.metrics`; when omitted, the first
evaluation metric is used. Successful rows are ranked ascending by this metric.
Failed model rows are preserved with `status: failed`, `rank: null`, and an
error message, so successful model results are not hidden.

Baseline model behavior:

- `moving_average`: averages the last `window_size` history steps, or all
  `input_len` steps when `window_size` is omitted, then repeats the average for
  the forecast horizon.
- `seasonal_naive`: cycles through the final `season_length` history steps
  until `output_len` predictions are produced.

See [docs/leaderboard_format.md](docs/leaderboard_format.md) for output
columns.

## Checkpoints and Resume

Checkpoints use schema version `1` and include:

- validated config snapshot
- model name, params, input/output lengths, feature count, and state dict
- optimizer name and state dict
- scaler name, params, and state dict
- metrics and environment metadata

Resume training by setting `training.resume_from` to a checkpoint path. The
configured `training.epochs` is the target final epoch, not the number of extra
epochs. For example, resuming an epoch-1 checkpoint with `epochs: 2` trains only
epoch 2. If the checkpoint epoch is already at or beyond the target, training is
skipped and evaluation still runs.

## Run Directory Strategy

- `overwrite: false` creates a unique run directory:
  `runs/<experiment_name>/<timestamp>_<short_id>/`.
- `overwrite: true` writes to `runs/<experiment_name>/latest/` and clears stale
  artifacts before running.

Every `results.json` includes `run_id`, `created_at`, `run_dir`, and
`experiment_name`.

## Validation Split

`val_ratio: 0` is allowed. In that case validation is skipped,
`validation_metrics` is `null`, and test metrics are still computed.

## Add a Dataset

1. Implement a `torch.utils.data.Dataset` compatible with
   `ForecastingDataset`.
2. Return batches with `x` shaped `[input_len, num_features]` and `y` shaped
   `[output_len, num_features]`.
3. Register the dataset with `DATASET_REGISTRY.register("name", DatasetClass)`.
4. Add catalog metadata through `DATASET_CATALOG.register(...)`.

See [examples/custom_dataset.py](examples/custom_dataset.py).

## Add a Model

1. Subclass `BaseForecastModel`.
2. Implement `forward(x)` for `x` shaped `[batch, input_len, num_features]`.
3. Return predictions shaped `[batch, output_len, num_features]`.
4. Register the model with `MODEL_REGISTRY.register("name", ModelClass)`.

## Run Tests

```bash
py -m pytest
ruff check .
ruff format --check .
mypy src
py -m ts_platform.cli.main train --config configs/examples/simple_forecast.yaml
py -m ts_platform.cli.main train --config configs/examples/csv_forecast.yaml
```

## API Demo

```bash
uvicorn ts_platform.api.app:create_app --factory --reload
```

Available endpoints:

- `GET /health`
- `GET /datasets`
- `GET /models`
- `GET /experiments`
- `POST /experiments/train`

The API keeps experiment discovery under the fixed safe `runs` root. For
`POST /experiments/train`, the API ignores any client-provided
`experiment.output_dir` and overwrites it with the same safe runs root. CLI
training still honors `experiment.output_dir` from the config.
