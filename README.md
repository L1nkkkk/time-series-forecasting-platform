# Time Series Forecasting Platform

This repository contains a configuration-driven MVP for a comprehensive time
series forecasting platform. It follows the same engineering ideas highlighted
by BasicTS: datasets, scalers, models, metrics, runners, and configs are
separate modules with small public interfaces and registry-based extension
points.

## Project Capability Summary

The current MVP focuses on a runnable local forecasting platform:

- Synthetic forecasting dataset.
- Local CSV forecasting dataset with time-based splits.
- Naive last-value, moving-average, seasonal-naive, linear, MLP, RNN, GRU,
  LSTM, and TCN forecasting models.
- Standard and min-max scalers.
- MAE, MSE, RMSE, MAPE, and WAPE metrics.
- Config snapshots, checkpoints, metrics output, and environment metadata.
- CLI and a synchronous FastAPI demo API.
- Original-scale evaluation metrics with optional scaled-space metrics.
- Versioned checkpoints that can restore model, scaler, and optimizer state.
- Strict CSV parameter validation, split-local missing-value handling, and
  dataset catalog discovery.
- CSV dataset profiling, catalog profiling, and config generation from catalog
  metadata.
- Multi-model compare runs with `leaderboard.json` and `leaderboard.csv`.
- Artifact manifests that make train and compare outputs discoverable.
- Safe manifest-based artifact downloads for JSON, YAML, CSV, and log files.
- Local asynchronous train/compare jobs for the demo API.
- Optional SQLite job metadata backend prototype for local jobs.
- Local `worker-once` prototype for queued SQLite jobs.
- SQLite job events/attempts inspection and finite `worker-loop` prototype.
- Explicit SQLite retry/timeout policy prototype for local jobs.
- Feature-aware CSV training and feature-aware compare with target-only
  metrics.

No BasicTS code is copied into this project.

## Demo Materials

- [docs/dashboard_demo.md](docs/dashboard_demo.md)
- [docs/demo_guide.md](docs/demo_guide.md)
- [docs/release_checklist.md](docs/release_checklist.md)
- [docs/final_report_outline.md](docs/final_report_outline.md)

For a visual local demo, start FastAPI and open `/ui`; see
[docs/dashboard_demo.md](docs/dashboard_demo.md).

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
`results.json` plus `artifacts.json` under `runs/simple_forecast/latest/`
because the example config sets `overwrite: true`.

Run the CSV example:

```bash
py -m ts_platform.cli.main train --config configs/examples/csv_forecast.yaml
```

Run the feature-aware CSV example:

```bash
py -m ts_platform.cli.main train --config configs/examples/csv_feature_forecast.yaml
```

Run a multi-model compare:

```bash
py -m ts_platform.cli.main compare --config configs/examples/compare_forecast.yaml
```

The compare command writes one normal Trainer run per model under
`runs/compare_forecast/latest/models/` and produces
`results.json`, `artifacts.json`, `leaderboard.json`, and `leaderboard.csv` in
the compare run directory. The compare-level `results.json` summarizes the
parent compare run, including success/failure counts, leaderboard paths, and the
same rows stored in `leaderboard.json`.

Run the lightweight model zoo compare:

```bash
py -m ts_platform.cli.main compare --config configs/examples/compare_model_zoo.yaml
```

Run the feature-aware compare smoke:

```bash
py -m ts_platform.cli.main compare --config configs/examples/compare_feature_forecast.yaml
```

The feature-aware compare config uses target and feature columns together. The
trainable models consume the full target-plus-feature history, while the
statistical baselines stay target-only and ignore the feature slice. Metrics
remain target-only for both feature-aware training and feature-aware compare.

## Dashboard Demo

Start the FastAPI app:

```bash
uvicorn ts_platform.api.app:create_app --factory
```

Open the local dashboard:

http://127.0.0.1:8000/ui

This is a lightweight local demo UI served by FastAPI with static HTML, CSS,
and vanilla JavaScript. It is not a production web UI. The dashboard calls the
existing API for health, datasets, models, experiments, jobs, results,
leaderboards, and artifacts, and it exposes whitelisted demo train/compare
buttons. See [docs/dashboard_demo.md](docs/dashboard_demo.md).

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
Feature-aware CSV training is shown in
[configs/examples/csv_feature_forecast.yaml](configs/examples/csv_feature_forecast.yaml).

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
- `feature_cols`: optional numeric input-only columns for the CSV dataset and
  batch layer. Feature-aware samples use `x = target history + feature history`
  and `y = target future only`.
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

Feature-aware CSV batches can be scaled directly with
`FeatureAwareScalerBundle`, which keeps target and feature scaler state
separate at the dataset wrapper layer. Trainer now builds separate fitted
target and feature scalers from the same `data.scaler` config, trains models on
`input_dim` tensors, and evaluates target-only forecasts on the original target
scale.

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

Profile one local CSV without training or cleaning it:

```bash
py -m ts_platform.cli.main profile-dataset --path tests/fixtures/tiny_series.csv --target-cols value --timestamp-col timestamp --input-len 8 --output-len 2
```

Profile CSV entries from a catalog:

```bash
py -m ts_platform.cli.main profile-catalog --catalog configs/datasets/local_csv.yaml --input-len 8 --output-len 2
```

Generate a training config from catalog metadata:

```bash
py -m ts_platform.cli.main make-config-from-catalog --catalog configs/datasets/local_csv.yaml --dataset tiny_csv --output /tmp/tiny_csv_generated.yaml --input-len 8 --output-len 2 --model linear --epochs 1
```

The generated config is a normal training YAML. It is not run automatically.

## Exogenous Features

Current CSV dataset construction supports `feature_cols` at the batch and
training layer. The end-to-end interface is documented in
[docs/exogenous_features_design.md](docs/exogenous_features_design.md), and
implementation is intentionally split into Phase 12 steps. Phase 12A adds
schema and compatibility infrastructure; Phase 12B adds CSV data-layer
`feature_cols` support; Phase 12C adds split target/feature scaler support;
Phase 12D adds feature-aware model forward support; Phase 12E adds Trainer,
evaluator, checkpoint, and results integration. Phase 12F adds feature-aware
compare/model-zoo smoke coverage.

## Discovery Commands

List datasets and models as JSON for scripts:

```bash
py -m ts_platform.cli.main list-datasets
py -m ts_platform.cli.main list-models
```

Read saved train or compare results as JSON:

```bash
py -m ts_platform.cli.main show-results --experiment compare_forecast --run latest
py -m ts_platform.cli.main show-leaderboard --experiment compare_forecast --run latest
py -m ts_platform.cli.main show-artifacts --experiment compare_forecast --run latest
py -m ts_platform.cli.main show-artifact --experiment compare_forecast --run latest --artifact leaderboard_json
py -m ts_platform.cli.main list-jobs
py -m ts_platform.cli.main list-jobs --job-backend sqlite --sqlite-db runs/jobs.sqlite3
py -m ts_platform.cli.main show-job --job-id 20260619T120000Z_a1b2c3
py -m ts_platform.cli.main show-job-events --job-id 20260619T120000Z_a1b2c3 --sqlite-db runs/jobs.sqlite3
py -m ts_platform.cli.main show-job-attempts --job-id 20260619T120000Z_a1b2c3 --sqlite-db runs/jobs.sqlite3
py -m ts_platform.cli.main list-stale-jobs --sqlite-db runs/jobs.sqlite3 --older-than-seconds 3600
py -m ts_platform.cli.main mark-stale-timeout --sqlite-db runs/jobs.sqlite3 --older-than-seconds 3600
py -m ts_platform.cli.main retry-job --job-id 20260619T120000Z_a1b2c3 --sqlite-db runs/jobs.sqlite3 --max-attempts 3
py -m ts_platform.cli.main worker-once --sqlite-db runs/jobs.sqlite3 --jobs-root runs/jobs --runs-root runs
py -m ts_platform.cli.main worker-loop --sqlite-db runs/jobs.sqlite3 --jobs-root runs/jobs --runs-root runs --max-jobs 1
```

`show-results` returns a train run `results.json` or a compare parent
`results.json`. `show-leaderboard` is meaningful for compare runs and reads
`leaderboard.json`. `show-artifacts` reads the run `artifacts.json` manifest.
`show-artifact` reads one named artifact that is registered in that manifest.
It prints JSON, YAML, CSV, and log artifacts to stdout by default, or writes
them to `--output` when provided. Checkpoints are intentionally blocked by
default in the CLI, and artifact files larger than 5 MiB are rejected.
`list-jobs` and `show-job` inspect local API job metadata under `runs/jobs` by
default. The default job backend is JSON. The Phase 8A SQLite prototype stores
metadata in `runs/jobs.sqlite3`; use `--job-backend sqlite` with
`--sqlite-db runs/jobs.sqlite3` for read-only CLI inspection of SQLite jobs.
`show-job-events` and `show-job-attempts` read SQLite observability rows as
JSON. `worker-once` claims and runs at most one queued SQLite job, then exits.
`worker-loop` repeats the same worker path with finite `--max-jobs` and
`--max-idle-cycles` bounds. `list-stale-jobs`, `mark-stale-timeout`, and
`retry-job` are explicit SQLite maintenance commands; they do not start a
scheduler or execute queued jobs.
CLI job submission is intentionally not provided because a one-shot CLI process
cannot keep an in-process background executor alive after exit.

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
- `rnn`, `gru`, and `lstm`: encode the input sequence and project the final
  hidden state directly to the full forecast horizon.
- `tcn`: applies a lightweight causal-ish Conv1d stack and projects the final
  hidden time step directly to the full forecast horizon.

See [docs/leaderboard_format.md](docs/leaderboard_format.md) for output
columns. In `leaderboard.json` and API responses, `model_params` is a JSON
object, and `target_cols`/`feature_cols` remain JSON arrays. In
`leaderboard.csv`, `model_params`, `target_cols`, and `feature_cols` are JSON
strings so the CSV cells remain portable.

## Checkpoints and Resume

Checkpoints use schema version `2` and include:

- validated config snapshot
- model name, params, input/output lengths, input/target dimensions, feature
  count, and state dict
- data target/feature column metadata and dimensions
- optimizer name and state dict
- target scaler name, params, and state dict
- feature scaler name, params, and state dict for feature-aware runs
- metrics and environment metadata

Schema version `1` target-only checkpoints remain loadable for compatibility.

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

`run_id` is formatted as `YYYYMMDDTHHMMSSZ_<6 hex chars>`, for example
`20260619T120000Z_a1b2c3`. Compare parent ids use the same format and are
reported as `compare_run_id`.

Every completed train or compare run also writes `artifacts.json`. Train
manifests include result, checkpoint, config snapshot, environment, and log
entries when those files exist. Compare manifests include compare results,
leaderboard JSON/CSV, compare config snapshot, and environment entries. See
[docs/artifacts.md](docs/artifacts.md). Manifest `run_dir` and
`compare_run_dir` values are compatibility metadata; artifact download
authorization uses the physical run directory resolved by `ExperimentStore`.

## Validation Split

`val_ratio: 0` is allowed. In that case validation is skipped,
`validation_metrics` is `null`, and test metrics are still computed.

## Add a Dataset

1. Implement a `torch.utils.data.Dataset` compatible with
   `ForecastingDataset`.
2. Return batches with `x` shaped `[input_len, input_dim]` and `y` shaped
   `[output_len, target_dim]`.
3. Register the dataset with `DATASET_REGISTRY.register("name", DatasetClass)`.
4. Add catalog metadata through `DATASET_CATALOG.register(...)`.

See [examples/custom_dataset.py](examples/custom_dataset.py).

## Add a Model

1. Subclass `BaseForecastModel`.
2. Implement `forward(x)` for `x` shaped `[batch, input_len, input_dim]`.
3. Return predictions shaped `[batch, output_len, target_dim]`.
4. Register the model with `MODEL_REGISTRY.register("name", ModelClass)`.

See [docs/model_zoo.md](docs/model_zoo.md) for built-in model parameters and
model extension notes.

## Run Tests

```bash
py -m pytest
ruff check .
ruff format --check .
mypy src
```

The full release smoke gate lives in [CONTRIBUTING.md](CONTRIBUTING.md) and
[docs/release_checklist.md](docs/release_checklist.md).

## API Demo

```bash
uvicorn ts_platform.api.app:create_app --factory --reload
```

Available endpoints:

- `GET /ui`
- `GET /demo/configs`
- `POST /demo/train/{demo_name}`
- `POST /demo/compare/{demo_name}`
- `GET /health`
- `GET /datasets`
- `GET /datasets/{dataset_name}`
- `GET /datasets/{dataset_name}/profile`
- `GET /models`
- `GET /experiments`
- `GET /experiments/{experiment_name}/{run_id}/results`
- `GET /experiments/{experiment_name}/{run_id}/artifacts`
- `GET /experiments/{experiment_name}/{run_id}/artifacts/{artifact_name}`
- `GET /experiments/{experiment_name}/{run_id}/leaderboard`
- `POST /experiments/train`
- `POST /experiments/compare`
- `POST /jobs/train`
- `POST /jobs/compare`
- `GET /jobs`
- `GET /jobs/stale`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/events`
- `GET /jobs/{job_id}/attempts`
- `GET /jobs/{job_id}/result`
- `GET /jobs/{job_id}/logs`
- `POST /jobs/{job_id}/cancel`
- `POST /jobs/{job_id}/timeout`
- `POST /jobs/{job_id}/retry`

The API keeps experiment discovery under the fixed safe `runs` root. For
`POST /experiments/train` and `POST /experiments/compare`, the API ignores any
client-provided `experiment.output_dir` and overwrites it with the same safe
runs root. CLI training and compare still honor `experiment.output_dir` from
their configs.

`GET /experiments` lists complete train runs, complete compare runs, and
incomplete run directories. Train summaries include checkpoint and test metric
metadata. Compare summaries include the parent compare run id, primary metric,
success/failure counts, and leaderboard paths.

`GET /datasets/{dataset_name}` returns catalog metadata. `GET
/datasets/{dataset_name}/profile` profiles only local CSV datasets already
described in the catalog and accepts only optional `input_len` and `output_len`
query parameters. It does not accept arbitrary paths and does not train.

Result lookup path parameters must be safe path components containing only
letters, numbers, `_`, `-`, and `.`. `run_id` also accepts `latest`. Path
separators, whitespace, `..`, absolute paths, and path escapes are rejected.
The artifacts endpoint returns only the manifest. The artifact download
endpoint accepts an `artifact_name`, not a path, and serves only files registered
in `artifacts.json`. The resolved file must stay inside both the fixed API runs
root and the physical run directory resolved by `ExperimentStore` for the
requested `experiment_name` and `run_id`. Manifest `run_dir` and
`compare_run_dir` values remain metadata only and cannot widen the download
boundary, so a tampered manifest cannot point to another run's file. The API
allows JSON, YAML, CSV, and log artifacts by default, rejects checkpoints
unless explicitly enabled through `APISettings`, and rejects files larger than
`APISettings.artifact_max_bytes` before returning a `FileResponse`.
`APISettings.artifact_allowed_kinds` controls the API downloadable kinds; the
CLI keeps its default safe policy and does not expose a checkpoint download
switch.

The `/jobs/*` endpoints add a lightweight local async layer on top of the same
safe train and compare services. A submitted job immediately returns a
`JobRecord` with status `queued` or `running`; background execution uses a
local `ThreadPoolExecutor`. By default the runner persists JSON metadata to
`runs/jobs/<job_id>/`. The optional SQLite prototype can store the same
metadata in `runs/jobs.sqlite3` behind the unchanged `/jobs` API while still
writing request snapshots under `runs/jobs/<job_id>/request_config.json`.
`APISettings.job_execution_mode = "external_worker"` makes SQLite-backed submit
endpoints queue only; `worker-once` then claims and executes one queued job.
SQLite-backed jobs also expose `GET /jobs/{job_id}/events` and
`GET /jobs/{job_id}/attempts`; JSON backend requests to those endpoints return
HTTP 400 because JSON jobs do not have event or attempt tables. `worker-loop`
is a finite local polling helper, not a daemon. Worker heartbeats are minimal
claim/success/failure markers. `GET /jobs/stale`, `POST /jobs/{job_id}/timeout`,
and `POST /jobs/{job_id}/retry` provide explicit SQLite retry/timeout
maintenance, but there is still no automatic retry scheduler or timeout sweep.
`GET /jobs/{job_id}/result` returns the saved `results.json` only after the job
has `succeeded`; unfinished, failed, or cancelled jobs return HTTP 409 with the
current status and error field. Cancelling a queued job marks it `cancelled`.
Cancelling a running job marks it `cancel_requested`; Python threads are not
force-killed, so the underlying run may still finish and record results. The
API closes the local executor on application shutdown and can lazily create a
new runner on the next jobs request, but interrupted running jobs are not
recovered. Corrupt `job.json` metadata is skipped by job listing and returns an
error when read directly. See [docs/jobs.md](docs/jobs.md).

## Production Hardening Roadmap

The current platform is a research/demo MVP. It now has local jobs, safe result
lookup, artifact manifests, and safe artifact downloads, but it is not a
multi-tenant production service. The production hardening design is documented
in:

- [docs/durable_queue_design.md](docs/durable_queue_design.md)
- [docs/deployment_design.md](docs/deployment_design.md)
- [docs/security_model.md](docs/security_model.md)
- [docs/roadmap.md](docs/roadmap.md)
