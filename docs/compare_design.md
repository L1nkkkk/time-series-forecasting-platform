# Compare Experiment Design

## Why Compare Is Needed

Phase 3 needs repeatable multi-model evaluation on the same dataset, split,
scaler, metrics, and seed. A compare workflow should prevent ad hoc runs from
using slightly different configs and should produce machine-readable summaries
for model selection.

## Config Draft

```yaml
experiment:
  name: csv_compare
  output_dir: runs
  seed: 42

data:
  name: csv
  input_len: 8
  output_len: 2
  params:
    path: tests/fixtures/tiny_series.csv
    timestamp_col: timestamp
    target_cols: [value]

models:
  - name: naive
  - name: linear
  - name: mlp
    params:
      hidden_sizes: [64, 32]

training:
  epochs: 2

evaluation:
  metrics: [mae, mse, rmse, wape]
```

The compare config should share one data/training/evaluation section and allow
per-model parameter overrides.

## Output Directory Structure

```text
runs/<compare_name>/<compare_run_id>/
  compare_config.yaml
  leaderboard.csv
  leaderboard.json
  models/
    naive/<run_id>/
    linear/<run_id>/
    mlp/<run_id>/
```

Each model run should still contain the existing config snapshot, checkpoint,
environment metadata, logs, and `results.json`.

## Leaderboard Schema

`leaderboard.csv` and `leaderboard.json` should contain one row per model run:

- `rank`
- `model_name`
- `model_params`
- `run_id`
- `run_dir`
- `primary_metric`
- `primary_metric_value`
- one column per reported test metric
- `created_at`

The primary metric should default to the first configured evaluation metric and
sort ascending for error metrics.

## Reusing Trainer

Compare should construct one normal `PlatformConfig` per model entry and call
the existing `Trainer`. The compare runner should only coordinate config
expansion, output directory layout, result collection, and leaderboard writing.
It should not duplicate training, evaluation, checkpoint, or scaler logic.

## Phase 3 Test List

- Compare config validates at least two model entries.
- Per-model params override shared defaults.
- Each model run writes normal Trainer artifacts.
- Shared data config is preserved across model runs.
- Leaderboard ranks by the primary metric.
- Leaderboard CSV and JSON contain the same rows.
- Failed model runs are recorded without hiding successful runs.
- CLI compare command rejects unsupported dataset/model names clearly.
