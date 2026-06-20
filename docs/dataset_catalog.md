# Dataset Catalog

Dataset catalogs describe local datasets for discovery, profiling, and config
generation. They are metadata only: the platform does not download remote data,
does not auto-train from a catalog entry, and does not let API users submit
arbitrary paths for profiling.

## YAML Schema

Catalog files contain a top-level `datasets` list:

```yaml
datasets:
  - name: tiny_csv
    dataset_type: csv
    domain: demo
    description: Tiny local CSV fixture for tests and examples.
    path: tests/fixtures/tiny_series.csv
    timestamp_col: timestamp
    target_cols: [value]
    frequency: D
    license: test-fixture
```

Required fields for all entries:

- `name`
- `dataset_type`
- `domain`
- `description`

Recommended fields for local CSV entries:

- `path`: local CSV path. Required when `dataset_type: csv`.
- `timestamp_col`: optional timestamp column.
- `target_cols`: optional list of target columns for discovery-only metadata.
  If present, it must be a list of strings. CSV profiling and
  `make-config-from-catalog` require it; missing `target_cols` produces a
  profile warning or a config-generation error.
- `frequency`: optional documented frequency.
- `license`: optional license label.

`source` and `citation` are optional free-text metadata fields. When `source`
is omitted, the loader uses `path` for CSV entries. A catalog is metadata only:
loading it does not train a model, clean data, or download remote datasets.

## Validation

`load_dataset_catalog()` validates the metadata shape before returning entries.
It rejects empty names, missing `dataset_type`, CSV entries without `path`,
non-string `timestamp_col`, and `target_cols` values that are not lists of
strings.

Registering metadata with the same normalized name overwrites the previous
entry. This is intentional and lets local catalogs override built-in discovery
metadata.

## Profile Catalog

Profile CSV entries in a catalog:

```bash
py -m ts_platform.cli.main profile-catalog --catalog configs/datasets/local_csv.yaml --input-len 8 --output-len 2
```

CSV entries with both `path` and `target_cols` produce full `DatasetProfile`
payloads. Unsupported dataset types are represented by a skipped profile row
with an `unsupported dataset_type` warning. Profiling reads data but does not
clean it, write files, train, or download anything.

## Make Config From Catalog

Generate a training config from one CSV catalog entry:

```bash
py -m ts_platform.cli.main make-config-from-catalog --catalog configs/datasets/local_csv.yaml --dataset tiny_csv --output /tmp/tiny_csv_generated.yaml --input-len 8 --output-len 2 --model linear --epochs 1
```

The generated config uses:

- `experiment.name: train_<dataset_name>_<model>`
- `experiment.output_dir: runs`
- `experiment.overwrite: true`
- `data.name: csv`
- `data.batch_size: 8` unless `--batch-size` is provided
- `data.scaler.name: standard`
- `data.params.path`, `timestamp_col`, and `target_cols` from the catalog
- `data.params.missing_policy: error`
- `data.params.sort_by_time: true`
- `training.device: cpu`
- `evaluation.metrics: [mae, mse, rmse, wape]`
- `evaluation.include_scaled_metrics: false`

The command writes YAML and prints a JSON summary. It does not run training.
The output must still pass the normal config loader before training.

## API Detail And Profile

`GET /datasets/{dataset_name}` returns registered catalog metadata.

`GET /datasets/{dataset_name}/profile` profiles only CSV datasets already
present in the catalog. It accepts optional `input_len` and `output_len` query
parameters and rejects unsupported query parameters such as `path`. Missing
dataset names return 404. Non-CSV datasets return 400. Missing local CSV files
return 200 with `exists: false` and a warning.

## Current Limitations

- Local CSV only.
- No parquet support.
- No remote dataset download.
- No exogenous `feature_cols`.
- No automatic training from catalog metadata.
- No user-submitted arbitrary path profiling through the API.
