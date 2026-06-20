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

## Recommended Next Phases

### Phase 10: Dataset Catalog Expansion

Goal: Improve dataset discovery and controlled local dataset use.

Deliverables:

- More catalog examples.
- Dataset validation docs.
- Optional dataset allowlist design.
- Additional CSV fixtures.

Non-goals:

- Remote dataset crawling.
- Multi-tenant dataset permissions.

Acceptance criteria:

- Users can discover and run documented local datasets safely.
- Invalid catalog entries fail clearly.

### Phase 11: Exogenous Features

Goal: Add controlled support for exogenous feature columns without breaking the
existing target-only CSV path.

Deliverables:

- CSV `feature_cols` parsing and validation.
- Dataset batch schema updates for target and feature tensors.
- Model compatibility policy for models that can or cannot consume exogenous
  features.
- Tests that prove split-local missing-value handling still cannot leak across
  train/validation/test boundaries.

Non-goals:

- Remote feature stores.
- Multi-tenant dataset permissions.
- Probabilistic forecasting.

Acceptance criteria:

- Existing target-only configs keep working.
- Feature-aware configs fail clearly when a selected model does not support
  exogenous inputs.

### Phase 12: Observability and Release

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

### Phase 13: Optional UI / Dashboard

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
