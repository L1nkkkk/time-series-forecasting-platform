# Development Process

## Branching

- Use short-lived feature branches from `main`.
- Prefix automation branches with `codex/` unless the repository owner requests
  another convention.

## Commit Messages

Use concise conventional-style messages:

- `feat: add synthetic dataset`
- `fix: validate split ratios`
- `test: cover scaler inverse transform`
- `docs: describe runner architecture`

## Pull Request Checklist

- Scope is small and described in the PR body.
- Public APIs have docstrings and type annotations.
- New behavior has tests.
- `python -m pytest` passes.
- `ruff check .` passes.
- `ruff format --check .` passes.
- `mypy src` passes or known gaps are documented.
- Config snapshots or generated run artifacts are not committed.

## Release Checklist

- `python -m pytest`
- `ruff check .`
- `ruff format --check .`
- `mypy src`
- CLI smoke commands
- Docs updated
- No runs artifacts committed
- No secrets
- PR reviewed

## Security Review Checklist

- No path traversal
- Safe path component validation
- `output_dir` not user-controlled in API
- Artifact download uses manifest and physical `run_dir` boundary
- Checkpoint download default disabled
- Job metadata path safe

## Job/Artifact Safety Checklist

- `job_id` safe
- `run_id` safe
- `artifact_name` safe
- Corrupted metadata behavior tested
- Artifact max size tested
- Forbidden kind tested
- Cross-run artifact path tested

## Review

Reviewers should focus on module boundaries, user-facing behavior, test
coverage, config validation, and reproducibility.

## Testing Strategy

- Unit tests cover config, registries, scalers, metrics, and model shapes.
- Smoke tests run a tiny synthetic training flow and verify output artifacts.
- Checkpoint tests cover schema validation, unknown schema rejection, model
  restore, and scaler restore.
- Resume tests train from a saved checkpoint and verify final epoch and result
  metadata.
- Evaluator tests verify original-scale and scaled-space metrics.
- Run directory tests verify unique runs and stale artifact cleanup.
- Experiment name safety tests reject path separators, parent references,
  absolute paths, and direct `ExperimentRecorder` run directory escape attempts.
- CLI tests call `ts_platform.cli.main.main([...])` with a temporary config.
- API tests cover health/list endpoints and synchronous `POST
  /experiments/train`.
- CSV dataset tests cover local file loading, time-based split boundaries,
  strict parameter validation, split-local missing value policies, split
  metadata, missing target columns, and train-only scaler fitting.
- Catalog tests cover local YAML dataset metadata loading, invalid schemas, and
  duplicate-name overwrite behavior.
- API experiment listing tests verify the endpoint uses a fixed runs root and
  returns run metadata from `results.json`.
- API training tests verify client-provided output directories are overwritten
  with the safe runs root.
- Baseline model tests cover moving-average and seasonal-naive shapes, values,
  parameter validation, and registry entries.
- Compare config tests cover model count, primary metric validation, alias
  safety, and extra-field rejection.
- Compare runner tests cover artifact writing, one Trainer run per model,
  shared config preservation, leaderboard ranking, CSV/JSON consistency,
  failure recording, and stop-on-failure behavior.
- CLI tests cover `train`, `compare`, `list-datasets`,
  `list-datasets --catalog`, `list-models`, `show-results`, and
  `show-leaderboard`, and `show-artifacts`.
- Phase 4 result-layer tests cover compare parent `results.json`, summary
  counts, all-failed compare persistence, `model_params` as an object in JSON
  and a string in CSV, `ExperimentStore` safe path handling, train/compare
  result reads, leaderboard reads, result API 400/404 behavior, synchronous
  compare API execution, and API output-root overrides.
- Artifact manifest tests cover train and compare `artifacts.json` writing,
  expected manifest entries, manifest path containment, `ExperimentStore`
  artifact reads and errors, artifact API 400/404 behavior, and CLI
  `show-artifacts`.
- Artifact download tests cover `ArtifactService` manifest-only lookup, unsafe
  artifact names, path escapes, kind policy, checkpoint denial, file size
  limits, media types, API file responses and error codes, ignored path query
  parameters, and CLI `show-artifact` stdout/output-file behavior.
- Artifact download hardening tests cover current run directory containment,
  cross-run artifact path rejection, APISettings max-size enforcement,
  APISettings allowed-kind enforcement, and explicit API checkpoint enablement.
- Artifact boundary hardening tests cover public `ExperimentStore.resolve_run`
  lookup for `latest`, recorded `run_id`, and `compare_run_id`; unsafe
  component and symlink escape rejection; and tampered manifest `run_dir` /
  `compare_run_dir` cases across `ArtifactService`, API download, and CLI
  `show-artifact`.
- Job tests cover `JobRecord` serialization, safe job ids, `JobStore`
  persistence and cancellation transitions, `JobRunner` success/failure/safe
  root behavior, non-blocking submission, Jobs API submit/status/result/cancel
  behavior, unsafe job id handling, app shutdown cleanup, corrupt job metadata
  list/get behavior, and read-only CLI `list-jobs` / `show-job`.

## CI Strategy

GitHub Actions installs dependencies, runs Ruff, checks formatting, runs mypy,
and executes pytest.

## Release Strategy

Use semantic versioning once the MVP stabilizes. Each release should include
changes, migration notes, and known limitations.

## Publish Workflow

Do not push a feature branch while tests or quality gates are failing. Before
publishing, inspect `git status` and `git diff --stat`, commit the intended
changes with a concise conventional message, push the feature branch, and open a
pull request when GitHub tooling is available. For Phase 5.1 job hardening
work, push `codex/phase5-job-hardening` and create or update a PR targeting
`main`.
For Phase 6 safe artifact download work, push
`codex/phase6-artifact-download` and create or update a PR targeting `main`.
For Phase 6.1 artifact hardening work, push
`codex/phase6-artifact-hardening` and create or update a PR targeting `main`.
For Phase 6.2 artifact boundary hardening work, push
`codex/phase6-artifact-boundary` and create or update a PR targeting `main`.
For Phase 7 production hardening design work, push
`codex/phase7-production-design` and create or update a PR targeting `main`.

## Compatibility Notes

Do not casually change `run_id` formatting. It is visible in `results.json`,
`artifacts.json`, API responses, CLI output, and lookup routes. If the format
changes, version the API behavior and document a migration path.

Job ids intentionally mirror the same timestamp plus six-hex suffix shape as
run ids. They are API-visible and stored in `runs/jobs/<job_id>`, so changes
need the same compatibility care.

## Local Job Runner Limitations

The Phase 5 runner is for local demos and tests. It uses an in-process
`ThreadPoolExecutor`, so jobs stop when the API process stops, and running
Python threads are not force-killed on cancel. API cancellation marks queued
jobs `cancelled` and running jobs `cancel_requested`. FastAPI shutdown closes
the local executor and clears the lazy runner singleton, but interrupted running
jobs are not recovered. Corrupt job metadata is skipped by list operations and
reported as an error for direct reads so it can be cleaned manually. A future
durable worker phase should move execution to a process or queue boundary with
explicit retry, resume, and cancellation semantics.
