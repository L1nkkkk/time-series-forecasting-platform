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
  as datetimes. Duplicate timestamps are rejected. Non-string values are
  rejected before the CSV is read.
- `target_cols`: non-empty list of target columns. Columns must exist and be
  numeric or safely convertible to numeric values. A single string is rejected
  because it is ambiguous.
- `missing_policy`: one of `error`, `drop`, `forward_fill`, or `zero_fill`.
  Invalid values are rejected during parameter parsing, even if the CSV contains
  no missing values.
- `sort_by_time`: strict boolean. When true, sort by `timestamp_col` before
  splitting.

`feature_cols` are rejected with `exogenous feature_cols are not supported yet`
when non-empty.

## Missing Values

- `error`: fail if any target value is missing in the selected split.
- `drop`: remove rows with missing target values in the selected split.
- `forward_fill`: forward-fill targets; if missing values remain at the start,
  fail clearly.
- `zero_fill`: replace missing target values with `0` in the selected split.

Missing policies are split-local. The dataset first loads and validates the
full CSV, computes raw train/validation/test row boundaries, selects the current
split, and only then applies the missing policy. This prevents `forward_fill`
from crossing train/validation/test boundaries and prevents `drop` from
changing another split's rows.

After the policy runs, each non-empty split must still contain at least
`input_len + output_len` rows so it can create one sliding window. Errors report
the split name, remaining row count, and minimum required rows.

## Split Strategy

CSV datasets use time-based splitting. Raw rows are split into train,
validation, and test periods before sliding windows are generated. Windows never
cross split boundaries, and scaler fitting uses only training split target
values.

`val_ratio: 0` is allowed. The validation dataset is empty and the trainer skips
validation while still computing test metrics.

`CSVForecastDataset.split_metadata()` returns:

- `mode`
- `split_start`
- `split_end`
- `row_count`
- `window_count`
- `start_timestamp`
- `end_timestamp`

When `timestamp_col` is omitted, timestamp metadata is `null`.

## Dataset Profiling

`profile-dataset` reads a local CSV and reports metadata without cleaning data,
writing files, or training a model:

```bash
py -m ts_platform.cli.main profile-dataset --path tests/fixtures/tiny_series.csv --target-cols value --timestamp-col timestamp --input-len 8 --output-len 2
```

The profile payload contains:

- `name`: optional dataset name.
- `dataset_type`: currently `csv`.
- `path`: local CSV path.
- `exists`: whether the file exists.
- `row_count` and `column_count`.
- `columns`: CSV column names.
- `timestamp_col`, `start_timestamp`, and `end_timestamp`.
- `target_cols`.
- `target_missing_counts`: missing-value count per available target column.
- `target_dtypes`: pandas dtype per available target column.
- `duplicate_timestamp_count`.
- `inferred_frequency`: pandas-inferred frequency when available.
- `min_required_rows`: `input_len + output_len` when both values are provided.
- `can_build_windows`: whether the raw row count can produce at least one
  window for the requested lengths.
- `warnings`: non-fatal data or metadata issues.

Warnings include missing files, missing target columns, missing timestamp
columns, missing target values, duplicate timestamps, frequency inference
failure, and insufficient rows for windows. Profiling does not apply
`missing_policy`, does not sort rows, and does not validate numeric
convertibility as strictly as training does; it is an inspection step before
building or running a config.

## Catalog Metadata

Dataset catalogs are YAML metadata documents. They are for discovery,
profiling, and config generation; they do not download data and the trainer
does not automatically infer config from them.

Recommended local CSV fields:

- `name`: non-empty catalog name.
- `dataset_type`: `csv`.
- `domain`: broad domain label.
- `description`: human-readable description.
- `path`: local CSV path. Required for CSV entries.
- `timestamp_col`: optional timestamp column name.
- `target_cols`: optional list of target column names. It must be a list when
  present; a plain string is rejected.
- `frequency`: optional documented frequency.
- `license`: optional license label.

Catalog entries with the same normalized name overwrite previous metadata when
registered. This keeps local catalog files easy to override while making the
behavior explicit.

## Current Limits

Exogenous `feature_cols` are not currently supported. Passing feature columns
raises an explicit error. This keeps model inputs shaped as
`[input_len, num_targets]` and targets shaped as `[output_len, num_targets]`.

Profiling currently supports local CSV files only. Remote URLs and parquet
files are not supported.
