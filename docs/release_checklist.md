# Release Checklist

## Code Quality

- `python -m pytest`
- `ruff check .`
- `ruff format --check .`
- `mypy src`

## CLI Smoke

Run the full local smoke gate before publishing a release branch:

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
python -m ts_platform.cli.main predict --model-export runs/csv_feature_forecast/latest/model_export.pt --values-json '[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]'
python -m ts_platform.cli.main list-jobs
```

## API Smoke

Optionally start the local demo API:

```bash
uvicorn ts_platform.api.app:create_app --factory
```

Suggested checks:

- `GET /health`
- `GET /datasets`
- `GET /models`
- `GET /experiments`
- `POST /configs/train/run`
- `POST /datasets/profile-csv`
- `POST /datasets/catalog/config`
- `POST /experiments/{experiment}/{run}/predict`
- `POST /jobs/train`
- `GET /jobs`
- Dashboard completed-run Markdown report export, Results Run Lookup artifact
  download with runs-root override, prediction values-file loading, Jobs CLI
  settings, and CLI-parity panels
- Optional hardening smoke: set `TS_PLATFORM_API_KEY`, verify `/models` returns
  401 without a key and 200 with `x-api-key`; verify oversized requests return
  413; enable `TS_PLATFORM_AUDIT_LOG_PATH` and confirm JSONL events are written.

API smoke is intentionally manual for this release checklist.

## Artifact Safety

- `artifacts.json` exists for train and compare runs.
- `show-artifacts` works.
- `show-artifact leaderboard_json` works.
- Checkpoint download remains blocked by default.

## Job Safety

- `list-jobs` works.
- SQLite job store remains optional.
- No `runs/jobs.sqlite3` is committed.

## Documentation

- README updated.
- CHANGELOG updated.
- docs/roadmap updated.
- docs/demo_guide updated.
- docs/report_export updated when dashboard report fields change.

## Repository Hygiene

- No `runs/` artifacts.
- No checkpoints.
- No secrets.
- No local env files.
- No temporary generated YAML under repo root.
