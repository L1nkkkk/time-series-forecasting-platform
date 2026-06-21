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
  catalog loading, dataset profiling, split helpers, and transforms.
- `scaler`: Normalize and inverse-normalize tensors, and serialize scaler state
  for checkpoints.
- `models`: Define forecasting models and model registry. Built-in models use
  the same `BaseForecastModel` contract, including naive/statistical baselines,
  linear/MLP baselines, recurrent baselines, and the lightweight TCN baseline.
- `metrics`: Calculate losses and evaluation metrics.
- `runner`: Orchestrate training, validation, testing, checkpoint save/restore,
  resume, evaluator calls, and multi-model compare runs.
- `experiment`: Create run directories, write logs, save configs, collect
  reproducibility metadata, record results, and write artifact manifests.
- `cli`: Parse command-line input and delegate to the runner.
- `api`: Expose platform endpoints, load discovery metadata, delegate training
  and compare work to small service layers, read saved metadata through
  `ExperimentStore`, and resolve safe artifact downloads through
  `ArtifactService`.
- `api/jobs`: Define a job store protocol, persist local job metadata through
  JSON or SQLite backends, and run train/compare jobs through a small
  in-process executor for the demo API.

## Component Flow

```text
YAML/JSON config
  -> Config loader and schema
  -> Dataset registry + catalog
  -> Split datasets by window policy for synthetic data or time policy for CSV
  -> Scaler fit on train split or restore from checkpoint
  -> Model registry builds a BaseForecastModel
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

Catalog profiling and config generation are read-side utilities:

```text
Catalog YAML or explicit CSV path
  -> Catalog loader validation or profile-dataset CLI args
  -> DatasetProfile for local CSV inspection
  -> optional make-config-from-catalog writes a normal PlatformConfig YAML
  -> user explicitly runs train or compare later
```

Result lookup uses one fixed root:

```text
API / CLI lookup request
  -> ExperimentStore
  -> safe experiment_name and run_id validation
  -> resolved path stays under runs root
  -> train results.json, compare results.json, leaderboard.json, or artifacts.json
```

Artifact downloads add one policy layer:

```text
API / CLI artifact request
  -> ArtifactService
  -> ExperimentStore resolves physical run directory
  -> ExperimentStore reads artifacts.json
  -> safe artifact_name exact match in manifest
  -> manifest path resolves under runs root
  -> manifest path resolves under resolved physical run directory
  -> kind, checkpoint, file existence, and size checks
  -> FileResponse or CLI text output
```

Local async jobs reuse the same safe execution services:

```text
POST /jobs/train or /jobs/compare
  -> JobStoreProtocol creates job metadata and request_config.json
  -> JsonJobStore writes runs/jobs/<job_id>/job.json by default
     or SQLiteJobStore writes runs/jobs.sqlite3 when configured
  -> in_process mode: JobRunner submits a ThreadPoolExecutor task
  -> external_worker mode: API leaves the job queued in SQLite
  -> worker-once claims the oldest queued SQLite job and creates an attempt
  -> worker-loop can repeat the same claim/execute path with finite bounds
  -> training_service or compare_service overwrites output_dir with runs root
  -> Trainer or CompareRunner writes normal run artifacts
  -> SQLite worker records heartbeats, attempts, events, result paths, and errors
  -> SQLite-only API/CLI observability reads job_events and job_attempts
  -> explicit timeout can mark stale running jobs timed_out
  -> explicit retry can requeue failed, timed_out, or cancelled jobs
```

## Production Evolution

The current API uses the local `JobRunner` for demo and research workloads. It
keeps job submission simple by running train and compare work in an in-process
`ThreadPoolExecutor`. The default store is JSON under `runs/jobs/<job_id>/`.
Phase 8A adds `SQLiteJobStore`, which stores the same `JobRecord` fields in
`runs/jobs.sqlite3` while keeping request config snapshots under
`runs/jobs/<job_id>/`. Phase 8B adds an external worker mode where the API only
queues SQLite jobs and a local `worker-once` process claims and executes them.
Phase 8C exposes SQLite events and attempts for read-side observability, adds
minimal worker heartbeat recording, adds stale running job inspection, and adds
a finite `worker-loop` CLI. Phase 8D adds explicit retry and timeout policy
operations. Automatic retry scheduling, retry backoff, worker supervision, and
automatic stale recovery remain future work.

Future durable queue work should replace the internal job backend without
breaking the `/jobs` API surface. The recommended path is SQLite-backed durable
queue first, then a separate worker process, then Redis/RQ or Celery if
multi-worker scheduling is needed, and Kubernetes Jobs only when resource
isolation or GPU scheduling becomes a near-term requirement.

`ExperimentStore` and `ArtifactService` should remain backend-agnostic. They
read stable run and artifact metadata and should not know whether a run was
started by the local runner, a SQLite worker, Redis/RQ, Celery, or Kubernetes.
Likewise, `Trainer` and `CompareRunner` should not depend on the API queue
implementation. Queue backend replacement should happen behind the jobs service
layer and must not affect core runner behavior.

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
8. Build or restore the model using sequence lengths and dataset dimensions
   (`input_dim` and `target_dim`), while preserving `num_features`
   compatibility for target-only configs.
9. Train from `checkpoint epoch + 1` through the target final epoch.
10. Evaluate validation metrics after every epoch when validation data exists.
11. Evaluate test metrics and record final results.
12. Save a schema-versioned checkpoint containing config, model, optimizer,
    scaler, metrics, and environment metadata.
13. Write `artifacts.json` after final result artifacts exist.

## Data Flow

Datasets expose `ForecastDimensions` through `dataset.dimensions`:

- `input_len`: history window length.
- `output_len`: forecast horizon length.
- `input_dim`: model input width.
- `target_dim`: forecast target width.
- `num_features`: target-only compatibility alias for `target_dim`.

In target-only runs, `input_dim == target_dim == num_features`. In
feature-aware CSV runs, `input_dim` includes target history plus feature
history, while `target_dim` remains the number of forecast target columns.
`Trainer` and `CompareRunner` build models with the explicit
`input_dim`/`target_dim` boundary and keep `num_features` as a target-only
compatibility alias.

Datasets yield `ForecastBatch` dictionaries with required fields:

- `x`: history tensor shaped `[input_len, input_dim]`
- `y`: target tensor shaped `[output_len, target_dim]`

Feature-aware CSV samples also expose optional `target_x`, `feature_x`, and
`metadata` fields. Runner code keeps consuming the required `x` and `y`
contract, while scaler, checkpoint, result, and leaderboard paths use the
metadata needed for safe feature-aware behavior.

DataLoader batches become:

- `x`: `[batch, input_len, input_dim]`
- `y`: `[batch, output_len, target_dim]`

Models must return predictions shaped like `y`. Built-in models produce direct
multi-step forecasts: one forward pass returns the whole output horizon shaped
`[batch, output_len, target_dim]`. The recurrent models encode the input
history and project the final hidden state to the horizon; the TCN encodes the
history with causal-ish temporal convolutions and projects the final hidden time
step. Neither path uses autoregressive decoding in the current model zoo.

Evaluation receives scaled model outputs and scaled targets. It computes
original-scale metrics by inverse-transforming predictions and targets through
the target scaler before calling the metrics registry. Feature columns never
enter `y`, inverse transforms, or metrics. When configured, scaled-space
metrics are also recorded under a separate `scaled` key.

## Feature-aware CSV Architecture

Phase 11 documented the exogenous feature interface, and Phase 12 implemented
it in staged pieces through data, scaler, model, trainer, checkpoint, and
compare integration.

Data layer responsibilities:

- Keep `target_cols` as the forecast target definition.
- Treat `feature_cols` as input-only CSV columns.
- Build `x` tensors from target history plus feature history.
- Keep `y` tensors target-only.
- Preserve split-local missing-value handling.

Scaler responsibilities:

- Fit a target scaler from training target values only.
- Fit a feature scaler from training feature values only.
- Apply inverse transforms only through the target scaler.
- Keep original-scale metrics target-only.

Model interface:

- `BaseForecastModel` supports both the legacy `num_features` constructor
  argument and the explicit `input_dim` / `target_dim` pair.
- `num_features` remains a target-only compatibility alias for `target_dim`.
- Trainable models consume the full `input_dim` history and output
  `target_dim`.
- Statistical baselines use `target_slice(x)` and ignore feature columns, so
  they remain target-only references in feature-aware runs.

Runner and evaluator behavior:

- `Trainer` builds split target/feature scalers for feature-aware datasets from
  the existing `data.scaler` config.
- `Trainer` passes dimensions and column metadata through model construction,
  results, and checkpoints.
- `Evaluator` receives the target scaler, so original-scale metrics remain
  target-only metrics.
- Compare leaderboards include feature-aware metadata and continue ranking
  target metrics.

Checkpoint behavior:

- Checkpoint schema v2 records `input_dim`, `target_dim`, `target_cols`,
  `feature_cols`, target scaler state, and feature scaler state.
- Checkpoint schema version `1` target-only checkpoints remain loadable.
- Resume validation rejects mismatched target columns, feature columns,
  dimensions, model config, and scaler config.

Backward compatibility:

- Configs without `feature_cols` must keep the current shapes and behavior.
- Old single-scaler configs continue to mean target scaler.
- Existing target-only model zoo and compare smoke tests stay green.

Future work:

- Basic feature-aware CSV training and compare support is complete.
- Future roadmap items are UI dashboard, production queue backend,
  auth / multi-user isolation, and advanced forecasting models.

The detailed design and phased migration plan live in
[exogenous_features_design.md](exogenous_features_design.md).

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

`data/profile.py` owns `DatasetProfile` and local CSV profiling. Profiling
reads CSV headers, row counts, target missing counts and dtypes, optional
timestamp range, duplicate timestamp count, inferred frequency, and whether a
requested `input_len + output_len` can build at least one window. It is
intentionally non-mutating: it does not apply missing policies, clean data,
write configs, or start training.

`make-config-from-catalog` is the only Phase 10 path that turns catalog
metadata into a runnable config. It requires a CSV catalog entry, writes a
normal `PlatformConfig` YAML, and leaves training as an explicit later command.
The trainer still builds datasets only from config files and never pulls
parameters from the catalog implicitly.

The dataset API exposes catalog detail and profile read paths. `GET
/datasets/{dataset_name}/profile` profiles only catalog-backed CSV entries and
does not accept user-provided paths, so API callers cannot turn the profile
endpoint into arbitrary local file reads.

## API Training Boundary

`api/services/training_service.py` owns API-specific training policy. It copies
the validated `PlatformConfig`, overwrites `experiment.output_dir` with the
safe API runs root, and then calls `Trainer`. This preserves CLI behavior while
preventing API clients from writing runs to arbitrary paths.

`api/services/compare_service.py` applies the same output-root policy to
`CompareConfig` and delegates to `CompareRunner`. The compare endpoint remains
synchronous for the demo API.

`api/jobs/base.py` defines `JobStoreProtocol`, the storage interface consumed by
`JobRunner` and the jobs route layer. It covers job creation, lookup, listing,
updates, status transitions, and cancellation.

`api/jobs/store.py` owns the default JSON job persistence under `runs/jobs`.
`JsonJobStore` validates job ids as safe path components, writes each
`job.json` and `request_config.json` atomically, lists jobs newest first, skips
corrupt job metadata during normal listing, and raises clear errors for
missing, unsafe, or directly-read corrupt job metadata. `JobStore` remains a
backward-compatible alias for `JsonJobStore`.

`api/jobs/sqlite_store.py` owns the SQLite job store prototype. It stores
`JobRecord` fields in a `jobs` table, writes the same request config snapshots
under `runs/jobs/<job_id>/request_config.json`, records lifecycle audit rows in
`job_events`, records local worker attempts in `job_attempts`, and uses
per-operation SQLite connections plus an `RLock` for simple multi-threaded
access. `claim_next_queued_job()` uses a SQLite transaction to mark the oldest
queued job running and create an attempt. It does not write a SQLite job's
`job.json` compatibility copy. `list_stale_running_jobs()` inspects running or
cancel-requested rows by latest attempt heartbeat or job `updated_at` without
mutating state. `mark_timed_out()` and
`mark_stale_running_jobs_timed_out()` explicitly mark stale work as
`timed_out`, and `retry_job()` records `job_retrying` / `job_requeued` events
before returning the job to `queued`.

`api/jobs/factory.py` builds either `JsonJobStore` or `SQLiteJobStore` from
`APISettings` and creates `JobRunner` with an injected store.

`api/jobs/runner.py` owns in-process asynchronous execution. It uses a
`ThreadPoolExecutor` with a conservative default of one worker, depends on
`JobStoreProtocol`, marks jobs queued/running/succeeded/failed, and records
result, artifact, leaderboard, and error fields. It does not duplicate training
or compare logic; it delegates to `training_service` and `compare_service`.

`api/jobs/worker.py` owns the Phase 8B local worker prototype. `JobWorker`
claims one queued SQLite job, reads the persisted request snapshot, validates it
as `PlatformConfig` or `CompareConfig`, delegates to the same safe execution
services, records minimal heartbeats at claim/success/failure boundaries, then
marks the job and attempt succeeded or failed. It is intentionally small;
daemon supervision, retry backoff, automatic timeout sweeps, and stale
heartbeat recovery are future hardening work.

`api/routes/jobs.py` exposes submit, list, get, result, logs, and cancel
endpoints. It maps unsafe job ids to HTTP 400, missing jobs to HTTP 404, and
not-ready or failed result lookups to HTTP 409. `APISettings.job_execution_mode`
controls whether submit endpoints run in-process or only queue SQLite jobs for
an external worker. SQLite-only events and attempts endpoints expose worker
observability without changing existing job record responses. SQLite-only
stale, timeout, and retry endpoints expose explicit local maintenance actions;
they do not start a scheduler or worker daemon.

`api/app.py` registers a FastAPI lifespan hook that shuts down the local
JobRunner executor and clears the lazy singleton during app shutdown. The next
jobs request can create a fresh runner. This is cleanup only; interrupted
running jobs are not recovered.

`api/services/experiment_store.py` owns read-side metadata access for API and
CLI callers. It validates `experiment_name` and `run_id` as safe path
components, accepts `latest`, resolves all candidate paths, and rejects any
resolved path outside the fixed runs root. Its public `resolve_run()` method
returns the physical run directory plus standard result and artifact paths for
callers that need an authorization boundary. It can list train, compare, and
incomplete runs; read train or compare `results.json`; read compare
`leaderboard.json`; and read train or compare `artifacts.json`. The store skips
`runs/jobs/<job_id>` because that tree is internal job metadata, not an
experiment run.

`api/services/artifact_service.py` owns safe file download lookup. It reuses
`ExperimentStore.resolve_run()` and `ExperimentStore.read_artifacts()` so
experiment and run id validation stay in one place, then requires
`artifact_name` to be a safe path component and an exact manifest entry name. It
never accepts a client path. The manifest path is resolved and checked against
the fixed runs root, then checked against the physical run directory returned by
`ExperimentStore`. Manifest `run_dir` and `compare_run_dir` values remain
metadata and cannot widen the boundary. This double check rejects both
outside-root paths and cross-run paths under the same root. The file must exist
before kind policy and size policy allow access. JSON, YAML, CSV, and log files
are downloadable by default. Checkpoints are denied unless policy explicitly
enables them, and files over 5 MiB are rejected by the default policy.

API routes build `ArtifactAccessPolicy` from `APISettings`, including
`artifact_max_bytes`, `artifact_allowed_kinds`, and
`allow_checkpoint_download`. CLI reads instantiate `ArtifactService` with the
default safe policy, so the CLI continues to reject checkpoint downloads.

## Artifact Manifests

`experiment/artifacts.py` owns `ArtifactEntry`, `ArtifactManifest`, train
manifest construction, compare manifest construction, and manifest writing. It
keeps manifest path rules close to artifact creation and verifies every
artifact path resolves inside the current run directory before a manifest is
written. Manifest directory fields are retained for compatibility and discovery,
but download authorization uses `ExperimentStore.resolve_run()`.

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

The model registry currently includes classical baselines (`naive`,
`moving_average`, `seasonal_naive`), simple trainable baselines (`linear`,
`mlp`), recurrent baselines (`rnn`, `gru`, `lstm`), and `tcn`. New model modules
register themselves on import, and `ts_platform.models.__init__` imports the
built-ins so CLI discovery and config-driven training see the same registry.
All registered forecasting models still inherit `BaseForecastModel` and expose
the same input/output shape boundary to `Trainer` and `CompareRunner`.

## Configuration-Driven Mechanism

All runnable experiment choices live in config files: dataset, split ratios,
scaler, model, optimizer, metrics, seed, and output location. This enables fair
comparisons because each run stores the exact config snapshot with its results.
