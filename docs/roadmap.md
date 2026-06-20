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

### Phase 13: Release Hardening / Final Project Polish

Goal: Bring the MVP to a deliverable state without adding large new runtime
features.

Delivered:

- Expanded changelog coverage for the completed MVP.
- Final contributor quality gate and release checklist.
- Demo guide for local presentation workflows.
- Final report outline for coursework/project reporting.
- README demo entry points.
- Leaderboard format clarification for feature-aware metadata.
- Documentation tests for release materials.

Non-goals:

- New model families.
- Redis, Celery, Kubernetes, or Web UI work.
- Trainer or CompareRunner rewrites.

Notes: This phase documents the release process and presentation path. It does
not change the core training, compare, jobs, or artifact execution model.

## Recommended Next Phases

### Phase 14: CLI Modularization

Goal: Split the growing CLI into smaller command modules without changing the
user-facing command surface.

### Phase 15: UI Dashboard

Goal: Prototype a small dashboard that reads existing API surfaces for
experiments, jobs, results, leaderboards, and artifacts.

### Phase 16: Production Queue Backend

Goal: Replace the local queue prototype with a production-ready worker backend
when operational requirements justify it.

### Phase 17: Auth / Multi-user Isolation

Goal: Add authentication, authorization, ownership, and artifact isolation for
multi-user deployments.

### Phase 18: Advanced Forecasting Models

Goal: Explore richer forecasting models after the platform, data, safety, and
release processes are stable.
