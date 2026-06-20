# Roadmap

## Completed Phases

### Phase 1: MVP Hardening

Goal: Make the initial training loop reliable enough for iterative work.

Delivered:

- Configuration-driven training.
- Safe experiment names.
- Reproducible run directories.
- Basic test and quality gates.

Notes: This phase established the local-first MVP shape.

### Phase 2: CSV Data

Goal: Add local CSV time series support.

Delivered:

- CSV dataset implementation.
- Time-based split support.
- Dataset catalog metadata.

Notes: CSV support is local-file based and remains a production security review
area.

### Phase 2.5: CSV/API Boundary Hardening

Goal: Harden CSV validation and API output boundaries.

Delivered:

- Strict CSV parameter validation.
- Split-local missing value handling.
- API output root override.

Notes: API callers cannot choose arbitrary output directories.

### Phase 3: Compare + Leaderboard

Goal: Compare multiple models in one parent run.

Delivered:

- `CompareRunner`.
- Per-model Trainer runs.
- `leaderboard.json` and `leaderboard.csv`.

Notes: Compare does not duplicate training logic.

### Phase 4: Results API + Artifact Manifest

Goal: Make results and run artifacts discoverable.

Delivered:

- `ExperimentStore`.
- Results and leaderboard lookup APIs.
- Train and compare `artifacts.json` manifests.

Notes: Run id compatibility became API-visible.

### Phase 5: Local Async Job Runner

Goal: Add asynchronous train and compare submissions for the demo API.

Delivered:

- Local `ThreadPoolExecutor` JobRunner.
- `JobStore`.
- `/jobs` API.

Notes: The runner is intentionally local and not durable.

### Phase 5.1: Job Runner Hardening

Goal: Improve local job reliability and error handling.

Delivered:

- FastAPI shutdown cleanup.
- Corrupted job metadata handling.
- Job result/log behavior tests.

Notes: The runner still does not recover running jobs after process crash.

### Phase 6: Safe Artifact Download

Goal: Serve registered artifacts without arbitrary path access.

Delivered:

- `ArtifactService`.
- Artifact download API.
- CLI `show-artifact`.

Notes: Checkpoint downloads are blocked by default.

### Phase 6.1: Artifact Download Hardening

Goal: Connect artifact download policy to API settings and tighten containment.

Delivered:

- API-configured max size and allowed kinds.
- Cross-run artifact rejection.
- Checkpoint enablement tests.

Notes: API policy remains stricter by default.

### Phase 6.2: Artifact Boundary Hardening

Goal: Stop trusting manifest directory metadata as a download boundary.

Delivered:

- Public `ExperimentStore.resolve_run`.
- Artifact boundary based on physical resolved run directory.
- Tampered manifest tests across service, API, and CLI.

Notes: Manifest `run_dir` and `compare_run_dir` are metadata only.

### Phase 7: Production Hardening Design

Goal: Document the production evolution path without adding infrastructure.

Delivered:

- Durable queue design.
- Deployment design.
- Security model.
- Roadmap.
- ADR for local JobRunner and production path.

Notes: This phase intentionally avoided Redis, Celery, Kubernetes, auth, UI,
and new model implementation.

### Phase 8: Durable Queue Prototype

Goal: Prototype a SQLite-backed queue while preserving the `/jobs` API.

Delivered:

- SQLite job metadata backend.
- External `worker-once` path.
- Job events, attempts, heartbeat, stale inspection, and finite `worker-loop`.
- Explicit retry and timeout policy operations.

Notes: The queue remains a local prototype. Automatic retry scheduling,
supervision, distributed workers, and Kubernetes integration remain out of
scope.

### Phase 9: Model Zoo Expansion Lite

Goal: Add classic, testable deep-learning forecasting baselines behind the
existing model registry.

Delivered:

- RNN, GRU, LSTM, and TCN forecasting baselines.
- Model zoo compare config.
- Shape, validation, registry, compare, and tiny training smoke tests.
- Model zoo docs.

Notes: Phase 9 keeps the existing `Trainer`, `CompareRunner`, CLI, and config
system boundaries. It does not add Transformer-style models, distributed
training, or new infrastructure.

### Phase 10: Dataset Catalog & Profiling Expansion

Goal: Improve dataset discovery and controlled local dataset use.

Delivered:

- CSV `DatasetProfile`.
- `profile-dataset` and `profile-catalog` CLI commands.
- `make-config-from-catalog` CLI command.
- Enhanced catalog metadata validation.
- Dataset detail/profile API endpoints.
- Dataset catalog and profiling docs.

Notes: Catalog metadata remains discovery/config-generation input only. The
trainer does not automatically infer config from catalog entries, the API does
not accept arbitrary profile paths, and this phase does not add remote dataset
downloads or exogenous feature columns.

### Phase 11: Exogenous Feature Design

Goal: Design controlled support for exogenous feature columns without breaking
the existing target-only CSV path.

Deliverables:

- `docs/exogenous_features_design.md`.
- ADR 0003 for the exogenous feature interface.
- Target/input tensor semantics.
- ForecastBatch migration plan.
- Target and feature scaler strategy.
- Model compatibility policy.
- Checkpoint/results impact.
- Phase 12 migration and testing plan.

Non-goals:

- Runtime `feature_cols` support.
- Changes to `CSVForecastDataset`, `BaseForecastModel`, `Trainer`,
  `Evaluator`, model `forward` logic, or checkpoint schema.
- Future-known covariate decoders.
- Remote feature stores or remote datasets.
- Automatic feature generation.

Acceptance criteria:

- The design keeps existing target-only configs working.
- `target_cols` and `feature_cols` semantics are documented.
- `input_dim` and `target_dim` migration is documented.
- Phase 12A through 12F are defined.
- Lightweight docs tests confirm the design docs are present.

Notes: Phase 11 is design-only. Non-empty `feature_cols` remain rejected until
the implementation phases begin.

### Phase 12A: Data Schema and ForecastBatch Migration

Goal: Introduce dimension compatibility infrastructure without enabling
exogenous feature columns.

Delivered:

- `ForecastDimensions` for `input_len`, `output_len`, `input_dim`, and
  `target_dim`.
- Optional `ForecastBatch` fields for future `target_x`, `feature_x`, and
  `metadata`.
- `ForecastingDataset.dimensions` plus target-only `input_dim` and
  `target_dim` on built-in datasets and scaled wrappers.
- `BaseForecastModel` compatibility for old `num_features` and new
  `input_dim`/`target_dim` constructor paths.
- `build_model` compatibility for old `num_features` and target-only
  `input_dim`/`target_dim` calls.
- Tests proving target-only dataset shapes, model attributes, and registry
  construction remain compatible.

Non-goals:

- Runtime `feature_cols` support.
- CSV feature-column validation or tensor concatenation.
- Target/feature scaler splitting.
- Feature-aware model forward logic.
- Trainer, evaluator, results, or checkpoint schema migration.

Notes: At the end of Phase 12A, `num_features` remained the active target-only
runtime path for `Trainer`, `CompareRunner`, jobs, artifacts, and checkpoint
schema version `1`.

### Phase 12B: CSVForecastDataset Feature Columns

Goal: Add CSV `feature_cols` support at the dataset and batch layer.

Delivered:

- Numeric `feature_cols` validation.
- Feature-aware `x = target history + feature history` batches.
- Target-only `y`.
- `target_x`, `feature_x`, and metadata for feature-aware samples.
- Split-local feature missing-value validation.

Notes: This phase did not enable feature-aware training.

### Phase 12C: Split Target/Feature Scaler Support

Goal: Scale target and feature slices separately without leaking feature state
into target metrics.

Delivered:

- `FeatureAwareScalerBundle`.
- `ScaledForecastingDataset` support for feature-aware samples.
- Separate target and feature scaler fit values.
- Reconstruction of scaled `x` from scaled target and feature slices.

Notes: Trainer remained blocked for feature-aware configs until later
integration.

### Phase 12D: Model Input/Target Dimension Migration

Goal: Let the model layer consume `input_dim` and forecast `target_dim`.

Delivered:

- `BaseForecastModel.validate_input()` and `target_slice()`.
- `build_model` support for `input_dim != target_dim`.
- Trainable models consume full `input_dim`.
- Statistical baselines ignore feature slices and forecast target history only.

Notes: This phase proved model forwards, not end-to-end training.

### Phase 12E: Trainer/Evaluator/Checkpoint Integration

Goal: Enable feature-aware CSV training while preserving target-only behavior.

Delivered:

- Trainer builds separate target and feature scalers from `data.scaler`.
- Feature-aware datasets flow through `FeatureAwareScalerBundle`.
- Models are constructed with `input_dim` and `target_dim`.
- Evaluator receives only the target scaler, so metrics remain target-only.
- Checkpoint schema version `2` stores dimensions, columns, and target/feature
  scaler states.
- Resume validates dimensions and target/feature column metadata.
- `results.json` records `data_metadata`.
- `csv_feature_forecast.yaml` example config and smoke coverage.

Notes: Compare/model-zoo smoke coverage is delivered in Phase 12F.

### Phase 12F: Feature-aware Compare and Model Zoo Smoke

Goal: Verify feature-aware training through compare and expose data metadata in
leaderboards.

Delivered:

- `compare_feature_forecast.yaml` for the feature-aware CSV fixture.
- Compare coverage for statistical baselines, trainable models, and the full
  model zoo shape.
- Leaderboard rows include `feature_aware`, dimensions, and target/feature
  column lists.
- CSV leaderboards serialize column lists as JSON strings while JSON
  leaderboards keep arrays.
- CLI smoke coverage for feature-aware compare and leaderboard inspection.

Notes: This phase keeps `Trainer`, `Evaluator`, and checkpoint behavior as the
single source of truth. Statistical baselines ignore feature slices; trainable
models consume the full `input_dim`.

## Recommended Next Phases

### Phase 12: Exogenous Features Implementation

Goal: Add controlled support for exogenous feature columns after the design is
accepted.

Staged plan:

- Phase 12A: Data schema and ForecastBatch migration. Completed as a
  compatibility-only layer.
- Phase 12B: `CSVForecastDataset` feature_cols support. Completed.
- Phase 12C: Scaler split support. Completed.
- Phase 12D: Model interface migration. Completed.
- Phase 12E: Trainer/Evaluator/checkpoint integration. Completed.
- Phase 12F: Compare/model zoo exogenous smoke tests. Completed.

Deliverables:

- CSV `feature_cols` parsing and validation.
- Target/input dimension tracking with `input_dim` and `target_dim`.
- `x` built from target history plus feature history.
- `y` kept target-only.
- Target scaler and feature scaler plumbing.
- Feature-aware model support for trainable models.
- Target-only behavior for statistical baselines.
- Checkpoint/result metadata for target and feature columns.
- Tests that prove split-local missing-value handling still cannot leak across
  train/validation/test boundaries.

Non-goals:

- Remote feature stores.
- Multi-tenant dataset permissions.
- Probabilistic forecasting.
- Automatic holiday/calendar feature generation.
- Future-known covariate decoding.

Acceptance criteria:

- Existing target-only configs keep working.
- Feature-aware configs train with supported models.
- Target-only baselines ignore feature slices and report target-only metrics.
- Checkpoint resume validates dimensions and column metadata.
- Compare/model zoo smoke tests cover exogenous configurations.

### Phase 13: Observability and Release

Goal: Make operations and releases easier to inspect.

Deliverables:

- Structured logs.
- Metrics design or lightweight implementation.
- Release checklist.
- Changelog discipline.

Non-goals:

- Full tracing platform.
- Hosted monitoring stack.

Acceptance criteria:

- Job lifecycle, compare outcomes, and model zoo failures are auditable.
- Releases have repeatable checks and documented changes.

### Phase 14: Optional UI / Dashboard

Goal: Provide a small dashboard only after API and storage boundaries are
stable.

Deliverables:

- Experiment list view.
- Job status view.
- Result and artifact links.

Non-goals:

- Multi-user SaaS.
- Complex workflow editor.

Acceptance criteria:

- UI uses existing APIs without requiring API redesign.
- Artifact access remains policy-controlled.
