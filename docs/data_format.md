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
- `feature_cols`: optional list of numeric input-only feature columns. When
  provided, every column must exist, must be numeric or safely convertible to
  numeric values, and must not overlap with `target_cols`.

## ForecastBatch and Dimensions

Datasets return `ForecastBatch` dictionaries. The current required runtime
fields are:

- `x`: history tensor shaped `[input_len, input_dim]`.
- `y`: target tensor shaped `[output_len, target_dim]`.

For target-only datasets, `input_dim == target_dim == num_features` and the
batch remains exactly:

```python
{"x": x, "y": y}
```

CSV datasets with `feature_cols` use:

- `target_dim = len(target_cols)`
- `feature_dim = len(feature_cols)`
- `input_dim = target_dim + feature_dim`
- `num_features = target_dim`

For feature-aware CSV samples, `x` is target history followed by feature
history, while `y` remains future targets only:

```text
x = concat(target_cols history, feature_cols history)
y = target_cols future
```

The concat order is always all `target_cols` first, then all `feature_cols`.
Feature-aware CSV samples also include:

- `target_x`: target-history slice.
- `feature_x`: exogenous feature-history slice.
- `metadata`: target columns, feature columns, and dimension metadata.

`ScaledForecastingDataset` supports target-only datasets with the old single
scaler path and can scale feature-aware samples with a
`FeatureAwareScalerBundle`. In feature-aware mode, the target scaler transforms
`target_x` and `y`, the feature scaler transforms `feature_x`, and the returned
`x` is reconstructed as `concat(scaled_target_x, scaled_feature_x)`.

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

Feature columns are not filled or dropped by `missing_policy`. If any selected
split contains missing values in `feature_cols`, dataset construction fails
with an error containing `feature columns contain missing values`.

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
- `target_cols`: optional list of target column names for discovery-only
  metadata. It must be a list when present; a plain string is rejected.
  `profile-catalog` and `make-config-from-catalog` require it for CSV entries;
  missing targets produce a profile warning or a config-generation error.
- `frequency`: optional documented frequency.
- `license`: optional license label.

Catalog entries with the same normalized name overwrite previous metadata when
registered. This keeps local catalog files easy to override while making the
behavior explicit.

## Future Exogenous Feature Columns

Phase 12B CSV support distinguishes forecast targets from input-only
exogenous features at the dataset and batch layer:

- `target_cols`: columns predicted by the model.
- `feature_cols`: columns used only as model inputs.
- `y`: future values for `target_cols` only.
- `x`: target history concatenated with feature history.

Metrics, inverse transforms, and original-scale result reporting remain
target-only in the planned end state. Feature columns do not enter `y`, target
scaler inverse transforms, or target metrics.

Models can now be constructed and called directly with feature-aware tensors
for shape tests, but full feature-aware training is intentionally still
blocked. Trainer rejects feature-aware CSV configs with `feature-aware training
is not implemented until Phase 12E` until Trainer, evaluator, and checkpoint
integration are migrated.

See [exogenous_features_design.md](exogenous_features_design.md) for the full
interface, scaler, checkpoint, and migration plan.

## Current Limits

Feature-aware CSV datasets can be constructed, inspected, and scaled directly
with split target/feature scalers. Model forwards can consume the resulting
`input_dim` and return `target_dim`, but full training with `feature_cols`
remains blocked until Trainer, evaluator, and checkpoint integration are
migrated.

Profiling currently supports local CSV files only. Remote URLs and parquet
files are not supported.
