# Demo Guide

## Setup

```bash
pip install -e ".[dev]"
```

## 1. List Datasets and Models

```bash
python -m ts_platform.cli.main list-datasets
python -m ts_platform.cli.main list-models
```

## 2. Profile a CSV Dataset

```bash
python -m ts_platform.cli.main profile-dataset --path tests/fixtures/tiny_series.csv --target-cols value --timestamp-col timestamp --input-len 8 --output-len 2
```

Key output fields:

- `row_count`: number of rows found in the CSV.
- `target_missing_counts`: missing values per requested target column.
- `inferred_frequency`: pandas-inferred timestamp frequency when available.
- `can_build_windows`: whether the requested input/output lengths can produce
  at least one sample window.
- `warnings`: non-fatal data or metadata issues.

## 3. Generate Config From Catalog

```bash
python -m ts_platform.cli.main make-config-from-catalog --catalog configs/datasets/local_csv.yaml --dataset tiny_csv --output /tmp/tiny_csv_generated.yaml --input-len 8 --output-len 2 --model linear --epochs 1
```

## 4. Run Target-only Training

```bash
python -m ts_platform.cli.main train --config configs/examples/simple_forecast.yaml
python -m ts_platform.cli.main train --config configs/examples/csv_forecast.yaml
```

## 5. Run Feature-aware Training

```bash
python -m ts_platform.cli.main train --config configs/examples/csv_feature_forecast.yaml
```

The feature-aware CSV example uses:

- `target_cols = [value]`
- `feature_cols = [temperature, holiday]`
- target-only metrics

## 6. Run Model Zoo Compare

```bash
python -m ts_platform.cli.main compare --config configs/examples/compare_model_zoo.yaml
```

## 7. Run Feature-aware Compare

```bash
python -m ts_platform.cli.main compare --config configs/examples/compare_feature_forecast.yaml
```

Feature-aware leaderboard metadata includes:

- `feature_aware`
- `input_dim`
- `target_dim`
- `feature_dim`
- `target_cols`
- `feature_cols`

## 8. Inspect Results and Artifacts

```bash
python -m ts_platform.cli.main show-results --experiment compare_feature_forecast --run latest
python -m ts_platform.cli.main show-leaderboard --experiment compare_feature_forecast --run latest
python -m ts_platform.cli.main show-artifacts --experiment compare_feature_forecast --run latest
python -m ts_platform.cli.main show-artifact --experiment compare_feature_forecast --run latest --artifact leaderboard_json
```

## 9. Jobs / Worker Demo Optional

```bash
python -m ts_platform.cli.main list-jobs
python -m ts_platform.cli.main worker-once --sqlite-db runs/jobs.sqlite3 --jobs-root runs/jobs --runs-root runs
python -m ts_platform.cli.main worker-loop --sqlite-db runs/jobs.sqlite3 --jobs-root runs/jobs --runs-root runs --max-jobs 1
```

This is a local prototype, not a production queue.

## Talking Points

- Software engineering process.
- Layered architecture.
- Safety boundaries.
- Feature-aware training.
- Extensibility.
