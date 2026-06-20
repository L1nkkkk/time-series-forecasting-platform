# Contributing

## Setup

Install the project in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

On Windows, use the Python launcher or the project Python installation if
`python` points at the Windows Store shim.

## Run Tests

Run the standard quality gate before publishing a branch:

```bash
python -m pytest
ruff check .
ruff format --check .
mypy src
python -m ts_platform.cli.main train --config configs/examples/simple_forecast.yaml
python -m ts_platform.cli.main train --config configs/examples/csv_forecast.yaml
python -m ts_platform.cli.main list-datasets
python -m ts_platform.cli.main list-datasets --catalog configs/datasets/local_csv.yaml
python -m ts_platform.cli.main list-models
python -m ts_platform.cli.main compare --config configs/examples/compare_forecast.yaml
python -m ts_platform.cli.main show-results --experiment compare_forecast --run latest
python -m ts_platform.cli.main show-leaderboard --experiment compare_forecast --run latest
python -m ts_platform.cli.main show-artifacts --experiment compare_forecast --run latest
python -m ts_platform.cli.main show-artifact --experiment compare_forecast --run latest --artifact leaderboard_json
python -m ts_platform.cli.main list-jobs
```

## Branch Naming

Use short-lived branches from `main`. Automation branches should use the
`codex/` prefix unless the repository owner requests a different convention.

## Coding Style

- Keep changes scoped to the requested behavior.
- Prefer existing module boundaries and helper APIs.
- Add tests for new behavior.
- Keep generated run artifacts out of commits.
- Do not add external services or dependencies without an explicit design
  decision.

## PR Checklist

- Summary explains the intent and scope.
- Tests and quality gates are reported.
- Documentation is updated when behavior or architecture changes.
- Security-sensitive path handling is covered by tests.
- No secrets, generated run outputs, or local environment files are committed.
