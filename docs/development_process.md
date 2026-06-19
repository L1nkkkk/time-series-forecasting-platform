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
  `list-datasets --catalog`, and `list-models`.

## CI Strategy

GitHub Actions installs dependencies, runs Ruff, checks formatting, runs mypy,
and executes pytest.

## Release Strategy

Use semantic versioning once the MVP stabilizes. Each release should include
changes, migration notes, and known limitations.
