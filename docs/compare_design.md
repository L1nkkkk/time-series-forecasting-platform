# Compare Experiments

## Why Compare Exists

Compare provides repeatable multi-model evaluation on the same dataset, split,
scaler, metrics, and seed. It prevents ad hoc runs from using slightly different
configs and writes machine-readable leaderboards for model selection.

## Config

```yaml
experiment:
  name: compare_forecast
  output_dir: runs
  seed: 42
  overwrite: true

data:
  name: csv
  input_len: 8
  output_len: 2
  batch_size: 8
  params:
    path: tests/fixtures/tiny_series.csv
    timestamp_col: timestamp
    target_cols: [value]
    missing_policy: error
    sort_by_time: true

models:
  - name: naive
  - name: moving_average
    params:
      window_size: 4
  - name: seasonal_naive
    params:
      season_length: 7
  - name: linear
  - name: mlp
    params:
      hidden_sizes: [16]
      dropout: 0.0

training:
  epochs: 1

evaluation:
  metrics: [mae, mse, rmse, wape]
  include_scaled_metrics: false

primary_metric: mae
continue_on_error: true
```

`models` must contain at least two entries. `primary_metric` must exist in
`evaluation.metrics`; when omitted it defaults to the first evaluation metric.
Aliases are optional and must be safe path components.

`configs/examples/compare_model_zoo.yaml` uses the same compare schema as a
lightweight model zoo benchmark. It keeps the dataset synthetic and tiny
(`length: 48`, `input_len: 4`, `output_len: 2`, `epochs: 1`) while running the
classical baselines, linear/MLP baselines, recurrent baselines, and TCN through
the same `Trainer` path. This makes it useful as a quick registry, config, and
leaderboard smoke test rather than a model-quality benchmark.

## Output Directory Structure

```text
runs/<compare_name>/<compare_run_id-or-latest>/
  artifacts.json
  compare_config_snapshot.yaml
  environment.json
  results.json
  leaderboard.csv
  leaderboard.json
  models/
    001_naive/latest/
    002_moving_average/latest/
    003_seasonal_naive/latest/
    004_linear/latest/
```

Each model run should still contain the existing config snapshot, checkpoint,
environment metadata, logs, and `results.json`.

The compare parent `results.json` contains the compare-level summary:

- `run_type: compare`
- `experiment_name`
- `compare_run_id`
- `compare_run_dir`
- `created_at`
- `leaderboard_json_path`
- `leaderboard_csv_path`
- `primary_metric`
- `success_count`
- `failed_count`
- `rows`

`compare_run_id` and `created_at` come from the parent
`ExperimentRecorder`. `rows` is the same array written to `leaderboard.json`.
The file is written even when every model fails and `continue_on_error: true`.

The compare parent `artifacts.json` indexes the parent compare outputs:
`results.json`, `leaderboard.json`, `leaderboard.csv`,
`compare_config_snapshot.yaml`, and `environment.json`. Model subdirectories
are normal train runs and write their own train manifests.

## Leaderboard

`leaderboard.csv` and `leaderboard.json` contain one row per model run:

- `rank`
- `status`
- `model_name`
- `model_alias`
- `model_params`
- `run_id`
- `run_dir`
- `checkpoint_path`
- `primary_metric`
- `primary_metric_value`
- `created_at`
- `error`
- one column per reported test metric

Successful rows are sorted ascending by `primary_metric`. Failed rows have
`rank: null`, no metric values, and an error message.

`leaderboard.json` keeps `model_params` as a JSON object. `leaderboard.csv`
serializes only that column as a JSON string so the CSV remains a flat table.

## Reusing Trainer

`CompareRunner` constructs one normal `PlatformConfig` per model entry and calls
the existing `Trainer`. It only coordinates config expansion, output directory
layout, result collection, and leaderboard writing. It does not duplicate
training, evaluation, checkpoint, or scaler logic.

Because each model entry is only a registry name plus params, compare can also
run model zoo benchmarks. `compare_model_zoo.yaml` includes `naive`,
`moving_average`, `seasonal_naive`, `linear`, `mlp`, `rnn`, `gru`, `lstm`, and
`tcn` with small hidden sizes so it remains fast on CPU.

## API Compare Endpoint

`POST /experiments/compare` accepts `CompareConfig`, forces
`experiment.output_dir` to the API safe runs root, and then delegates to
`CompareRunner`. It is synchronous for the demo API and intentionally does not
introduce an async queue or distributed training layer.

## Failure Strategy

When `continue_on_error: true`, a model failure is recorded as a failed
leaderboard row and later models still run. If every model fails, compare still
writes an all-failed leaderboard so the user can inspect every error in one
place.

When `continue_on_error: false`, the first failing model raises a
`RuntimeError` and compare stops.

## Test Coverage

- Compare config validates at least two model entries.
- `primary_metric` must be in `evaluation.metrics`.
- Model aliases must be safe path components.
- Each model run writes normal Trainer artifacts.
- Shared data config is preserved across model runs.
- Leaderboard ranks by the primary metric.
- Leaderboard CSV and JSON contain the same rows.
- Failed model runs are recorded without hiding successful runs.
- CLI compare command rejects unsupported dataset/model names clearly.
