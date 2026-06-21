# Contributing

## Setup

Install the project in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

On Windows, use the Python launcher or the project Python installation if
`python` points at the Windows Store shim.

## Run Tests

Run the final quality gate before publishing a branch:

```bash
python -m pytest
ruff check .
ruff format --check .
mypy src
python -m ts_platform.cli.main train --config configs/examples/simple_forecast.yaml
python -m ts_platform.cli.main train --config configs/examples/csv_forecast.yaml
python -m ts_platform.cli.main train --config configs/examples/csv_feature_forecast.yaml
python -m ts_platform.cli.main list-datasets
python -m ts_platform.cli.main list-datasets --catalog configs/datasets/local_csv.yaml
python -m ts_platform.cli.main profile-dataset --path tests/fixtures/tiny_series.csv --target-cols value --timestamp-col timestamp --input-len 8 --output-len 2
python -m ts_platform.cli.main profile-catalog --catalog configs/datasets/local_csv.yaml --input-len 8 --output-len 2
python -m ts_platform.cli.main make-config-from-catalog --catalog configs/datasets/local_csv.yaml --dataset tiny_csv --output /tmp/tiny_csv_generated.yaml --input-len 8 --output-len 2 --model linear --epochs 1
python -m ts_platform.cli.main list-models
python -m ts_platform.cli.main compare --config configs/examples/compare_forecast.yaml
python -m ts_platform.cli.main compare --config configs/examples/compare_model_zoo.yaml
python -m ts_platform.cli.main compare --config configs/examples/compare_feature_forecast.yaml
python -m ts_platform.cli.main show-results --experiment compare_feature_forecast --run latest
python -m ts_platform.cli.main show-leaderboard --experiment compare_feature_forecast --run latest
python -m ts_platform.cli.main show-artifacts --experiment compare_feature_forecast --run latest
python -m ts_platform.cli.main show-artifact --experiment compare_feature_forecast --run latest --artifact leaderboard_json
python -m ts_platform.cli.main list-jobs
```

Do not commit generated `runs/` artifacts, `runs/jobs.sqlite3`, checkpoints, or
temporary generated configs. Every PR must report the tests and smoke commands
that were actually run.

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

## CLI Commands

CLI commands are registered from `src/ts_platform/cli/commands/`.
`src/ts_platform/cli/main.py` should remain a thin entrypoint that builds the
parser and dispatches to the selected handler.

When adding or changing a CLI command, keep the parser registration and handler
in the relevant command module, add command coverage in `tests/test_cli.py`,
and update user-facing docs when command behavior changes.

## PR Checklist

- Summary explains the intent and scope.
- Tests and quality gates are reported.
- Documentation is updated when behavior or architecture changes.
- Security-sensitive path handling is covered by tests.
- No secrets, generated run outputs, or local environment files are committed.
