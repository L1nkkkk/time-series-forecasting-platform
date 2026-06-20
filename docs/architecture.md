# Architecture

## Design Lineage

This project references BasicTS at the idea level: separate dataset, scaler,
metric, runner, model, and config responsibilities, then bind them through a
configuration-driven runner. The implementation in this repository is original.

BasicTS is Apache-2.0 licensed. Because this MVP does not copy BasicTS source
code, no third-party notice is required beyond documenting the design reference.

## Module Responsibilities

- `config`: Load YAML or JSON and validate all user input with Pydantic.
- `data`: Define forecasting datasets, dataset registries, catalog metadata,
  catalog loading, split helpers, and transforms.
- `scaler`: Normalize and inverse-normalize tensors, and serialize scaler state
  for checkpoints.
- `models`: Define forecasting models and model registry.
- `metrics`: Calculate losses and evaluation metrics.
- `runner`: Orchestrate training, validation, testing, checkpoint save/restore,
  resume, evaluator calls, and multi-model compare runs.
- `experiment`: Create run directories, write logs, save configs, collect
  reproducibility metadata, record results, and write artifact manifests.
- `cli`: Parse command-line input and delegate to the runner.
- `api`: Expose platform endpoints, load discovery metadata, delegate training
  and compare work to small service layers, and read saved artifacts through
  `ExperimentStore`.
- `api/jobs`: Persist local job metadata and run train/compare jobs through a
  small in-process executor for the demo API.

## Component Flow

```text
YAML/JSON config
  -> Config loader and schema
  -> Dataset registry + catalog
  -> Split datasets by window policy for synthetic data or time policy for CSV
  -> Scaler fit on train split or restore from checkpoint
  -> Model registry
  -> Model/optimizer restore when resume_from is configured
  -> Trainer loop
  -> Evaluator inverse-transforms predictions for original-scale metrics
  -> Versioned checkpoint + config snapshot + results + artifacts.json
```

Compare runs add a thin orchestration layer:

```text
Compare YAML/JSON config
  -> Compare config loader and schema
  -> CompareRunner parent run directory
  -> One PlatformConfig per model
  -> Existing Trainer for each model
  -> parent results.json + leaderboard.json + leaderboard.csv + artifacts.json
```

Result lookup uses one fixed root:

```text
API / CLI lookup request
  -> ExperimentStore
  -> safe experiment_name and run_id validation
  -> resolved path stays under runs root
  -> train results.json, compare results.json, leaderboard.json, or artifacts.json
```

Local async jobs reuse the same safe execution services:

```text
POST /jobs/train or /jobs/compare
  -> JobStore creates runs/jobs/<job_id>/job.json and request_config.json
  -> JobRunner submits a ThreadPoolExecutor task
  -> training_service or compare_service overwrites output_dir with runs root
  -> Trainer or CompareRunner writes normal run artifacts
  -> JobStore records succeeded, failed, result paths, and errors
```

## Training Flow

1. Load and validate config.
2. Set random seed.
3. Build dataset from registry.
4. Split into train, validation, and test datasets.
5. For CSV datasets, strictly parse dataset parameters, validate local data,
   sort by time when configured, split raw rows into train/validation/test
   periods, apply missing policies inside the selected split, and then generate
   sliding windows inside each split.
6. If `resume_from` is set, load checkpoint, validate compatibility, and
   restore scaler/model/optimizer state. Otherwise fit scaler on the training
   split.
7. Wrap splits with transforms and build deterministic DataLoaders.
8. Build or restore model using sequence lengths and feature count.
9. Train from `checkpoint epoch + 1` through the target final epoch.
10. Evaluate validation metrics after every epoch when validation data exists.
11. Evaluate test metrics and record final results.
12. Save a schema-versioned checkpoint containing config, model, optimizer,
    scaler, metrics, and environment metadata.
13. Write `artifacts.json` after final result artifacts exist.

## Data Flow

Datasets yield dictionaries with:

- `x`: history tensor shaped `[input_len, num_features]`
- `y`: target tensor shaped `[output_len, num_features]`

DataLoader batches become:

- `x`: `[batch, input_len, num_features]`
- `y`: `[batch, output_len, num_features]`

Models must return predictions shaped like `y`.

Evaluation receives scaled model outputs and scaled targets. It computes
original-scale metrics by inverse-transforming both predictions and targets
before calling the metrics registry. When configured, scaled-space metrics are
also recorded under a separate `scaled` key.

## Checkpoint and Resume Boundaries

- `runner/checkpoint.py` owns checkpoint schema validation, save/load, model
  restore, scaler restore, and optimizer state restore.
- `scaler` implementations own `state_dict` and `load_state_dict`.
- `runner/evaluator.py` owns original-scale versus scaled-space metric
  calculation.
- `Trainer` coordinates these pieces but does not encode checkpoint schema
  details.

If `val_ratio` is `0`, validation is skipped and `validation_metrics` is
`null`. Test evaluation is still required.

## CSV Data Flow

`CSVForecastDataset` parses `DataConfig.params` through `CSVDatasetParams`,
loads a local CSV, validates target columns, optionally parses and sorts a
timestamp column, and uses raw-row time-based splits. The configured missing
value policy runs after selecting the current split, so train/validation/test
targets do not overlap and fill/drop behavior cannot cross split boundaries.

`split_metadata()` exposes split boundaries, post-policy row count, window
count, and optional timestamp range for tests, API use, and experiment
analysis.

The scaler fit path remains unchanged from the trainer perspective:
`train_dataset.scaler_fit_values()` returns only training-period target values.
This keeps validation and test periods out of scaler state.

Local catalog files are metadata documents loaded through
`data/catalog_loader.py`. They can register entries in `DATASET_CATALOG` for
discovery and API listing, but they do not replace explicit training configs.
The API loads `configs/datasets/*.yaml` during app creation and skips damaged
catalog files with a warning. `DatasetCatalog.register` overwrites metadata
with the same normalized name.

## API Training Boundary

`api/services/training_service.py` owns API-specific training policy. It copies
the validated `PlatformConfig`, overwrites `experiment.output_dir` with the
safe API runs root, and then calls `Trainer`. This preserves CLI behavior while
preventing API clients from writing runs to arbitrary paths.

`api/services/compare_service.py` applies the same output-root policy to
`CompareConfig` and delegates to `CompareRunner`. The compare endpoint remains
synchronous for the demo API.

`api/jobs/store.py` owns local job persistence under `runs/jobs`. It validates
job ids as safe path components, writes each `job.json` and
`request_config.json` atomically, lists jobs newest first, skips corrupt job
metadata during normal listing, and raises clear errors for missing, unsafe, or
directly-read corrupt job metadata.

`api/jobs/runner.py` owns in-process asynchronous execution. It uses a
`ThreadPoolExecutor` with a conservative default of one worker, marks jobs
queued/running/succeeded/failed, and records result, artifact, leaderboard, and
error fields. It does not duplicate training or compare logic; it delegates to
`training_service` and `compare_service`.

`api/routes/jobs.py` exposes submit, list, get, result, logs, and cancel
endpoints. It maps unsafe job ids to HTTP 400, missing jobs to HTTP 404, and
not-ready or failed result lookups to HTTP 409.

`api/app.py` registers a FastAPI lifespan hook that shuts down the local
JobRunner executor and clears the lazy singleton during app shutdown. The next
jobs request can create a fresh runner. This is cleanup only; interrupted
running jobs are not recovered.

`api/services/experiment_store.py` owns read-side artifact access for API and
CLI callers. It validates `experiment_name` and `run_id` as safe path
components, accepts `latest`, resolves all candidate paths, and rejects any
resolved path outside the fixed runs root. It can list train, compare, and
incomplete runs; read train or compare `results.json`; and read compare
`leaderboard.json`; and read train or compare `artifacts.json`.
The store skips `runs/jobs/<job_id>` because that tree is internal job metadata,
not an experiment run.

## Artifact Manifests

`experiment/artifacts.py` owns `ArtifactEntry`, `ArtifactManifest`, train
manifest construction, compare manifest construction, and manifest writing. It
keeps manifest path rules close to artifact creation and verifies every
artifact path resolves inside the current run directory before a manifest is
written.

`Trainer.run()` writes train `artifacts.json` after `results.json` and the
final checkpoint exist. The train manifest includes result, checkpoint, config
snapshot, environment, and log entries when those files exist.

`CompareRunner.run()` writes compare `artifacts.json` after parent
`results.json`, `leaderboard.json`, and `leaderboard.csv` exist. The compare
manifest includes parent result, leaderboard JSON/CSV, compare config snapshot,
and environment entries. Per-model runs are normal Trainer runs and write their
own train manifests.

## Experiment Name Safety

`ExperimentConfig.name` is validated as a single safe path component: letters,
numbers, `_`, `-`, and `.` only; no path separators, whitespace, `..`, absolute
paths, empty names, or names longer than 80 characters. `ExperimentRecorder`
performs defense-in-depth by resolving the computed run directory and verifying
that it remains under `root_dir`.

This validation applies to CLI, API, Trainer, and CompareRunner-created model
runs.

## Run ID Compatibility

Run ids are generated by `ExperimentRecorder` as
`YYYYMMDDTHHMMSSZ_<6 hex chars>`, for example
`20260619T120000Z_a1b2c3`. Compare parent run ids use the same format and are
exposed as `compare_run_id`. Treat this format as API-visible; changing it
requires a versioned compatibility plan.

## Compare Runner

`runner/comparer.py` owns Phase 3 compare orchestration. It creates a compare
parent directory, saves the compare config snapshot and environment metadata,
then expands each `CompareModelConfig` into a normal `PlatformConfig` with:

- shared data/training/evaluation config
- model-specific name and params
- safe model run name such as `001_naive`
- output root under `<compare_run_dir>/models`

Each model is executed through the existing `Trainer`; compare does not copy
training, scaling, checkpoint, evaluation, or metric logic. The authoritative
leaderboard metrics come from `TrainingResult.test_metrics["original"]`.

Successful rows are ranked ascending by `primary_metric`. Failed rows are kept
with `status: failed`, `rank: null`, and `error`, then appended after successful
rows. With `continue_on_error: false`, the first model failure aborts the compare
run with a clear exception.

After writing the leaderboard, `CompareRunner` writes parent `results.json` with
the compare run id, created timestamp, leaderboard paths, success/failure
counts, primary metric, and the same rows as `leaderboard.json`. JSON rows keep
`model_params` as an object; only CSV serializes that column to a JSON string.

## Registry Mechanism

Registries map string names to implementation classes or callables. This keeps
configuration files stable and prevents the trainer from importing every custom
implementation directly.

## Configuration-Driven Mechanism

All runnable experiment choices live in config files: dataset, split ratios,
scaler, model, optimizer, metrics, seed, and output location. This enables fair
comparisons because each run stores the exact config snapshot with its results.
