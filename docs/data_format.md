# Data Format

## CSV Forecasting Data

CSV datasets are local files configured with `data.name: csv`.

Required CSV target data:

```csv
timestamp,value
2024-01-01,1.0
2024-01-02,2.0
2024-01-03,3.0
```

Configuration example:

```yaml
data:
  name: csv
  input_len: 8
  output_len: 2
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  params:
    path: tests/fixtures/tiny_series.csv
    timestamp_col: timestamp
    target_cols: [value]
    missing_policy: error
    sort_by_time: true
```

## Fields

- `path`: CSV file path. Missing files raise `FileNotFoundError`.
- `timestamp_col`: optional timestamp column. When provided, values are parsed
  as datetimes. Duplicate timestamps are rejected.
- `target_cols`: non-empty list of target columns. Columns must exist and be
  numeric or safely convertible to numeric values.
- `missing_policy`: one of `error`, `drop`, `forward_fill`, or `zero_fill`.
- `sort_by_time`: when true, sort by `timestamp_col` before splitting.

## Missing Values

- `error`: fail if any target value is missing.
- `drop`: remove rows with missing target values.
- `forward_fill`: forward-fill targets; if missing values remain at the start,
  fail clearly.
- `zero_fill`: replace missing target values with `0`.

## Split Strategy

CSV datasets use time-based splitting. Raw rows are split into train,
validation, and test periods before sliding windows are generated. Windows never
cross split boundaries, and scaler fitting uses only training split target
values.

`val_ratio: 0` is allowed. The validation dataset is empty and the trainer skips
validation while still computing test metrics.

## Current Limits

Exogenous `feature_cols` are not supported in Phase 2. Passing feature columns
raises an explicit error. This keeps model inputs shaped as
`[input_len, num_targets]` and targets shaped as `[output_len, num_targets]`.
