# Dataset Catalog

Dataset catalogs describe local and public-source datasets for discovery,
profiling, config generation, and selected dataset preparation. Most public
entries remain metadata, but selected benchmark rows include enough asset
metadata for the platform to download or convert them into local trainable CSV
files.

`configs/datasets/public_time_series.yaml` is a curated public-source catalog
covering energy, finance, traffic, weather, environment, mobility, retail,
forecasting competitions, medical/ICU time series, and industrial predictive
maintenance datasets. Remote entries keep source attribution but remain
metadata-only unless they define `download_url` and `archive_format`.

The dashboard exposes the merged catalog with keyword search and domain
filtering. Filtered views only change the visible table; the custom experiment
dataset template selector still uses the complete catalog.

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

Recommended fields for prepared public assets:

- `download_url`: direct downloadable source URL.
- `archive_format`: one of `raw_csv`, `csv`, `raw_txt`, `raw_matrix`, or
  `zip_csv`.
- `local_path`: local prepared file name under `data/external/<name>/<version>/`.
- `version`: dataset asset version label, default `v1`.
- `checksum`: optional `sha256:<digest>` integrity check.

`source` and `citation` are optional free-text metadata fields. When `source`
is omitted, the loader uses `path` for CSV entries. A catalog is metadata only:
loading it does not train a model or download remote datasets. Downloads happen
only through explicit prepare commands or prepare API calls.

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

## Prepare Public Dataset Assets

Prepare a supported public dataset:

```bash
py -m ts_platform.cli.main prepare-dataset --dataset etth1
```

The prepare command writes:

- local CSV data under `data/external/<dataset>/<version>/`;
- `data/cache/datasets/manifest.json`;
- `data/cache/datasets/prepared_catalog.yaml`;
- a default train config under `data/cache/datasets/configs/`.

Inspect or clear the cache:

```bash
py -m ts_platform.cli.main show-dataset-cache
py -m ts_platform.cli.main clear-dataset-cache --dataset etth1
```

## API Detail And Profile

`GET /datasets` returns built-in catalog metadata merged with persisted user
dataset metadata from `data/user_datasets.json`. Prepared public entries expose
their local CSV path and can be used by the custom experiment form.

`POST /datasets/user` persists one user-supplied CSV dataset metadata entry.
The UI uses this when the user saves a local CSV dataset after choosing a file
or filling in a path manually.

`DELETE /datasets/user` clears persisted user dataset metadata.

`DELETE /datasets/user/{dataset_name}` removes one persisted user dataset
metadata entry.

`POST /datasets/{dataset_name}/prepare` prepares one supported public dataset
asset.

`GET /datasets/{dataset_name}/asset` returns one dataset asset status record.

`GET /datasets/cache` returns all prepared dataset asset records.

`DELETE /datasets/cache/{dataset_name}` removes one prepared asset and cache
manifest entry.

`GET /datasets/{dataset_name}` returns registered catalog metadata. User
metadata overrides a built-in row with the same normalized name.

`GET /datasets/{dataset_name}/profile` profiles only CSV datasets already
present in the catalog. It accepts optional `input_len` and `output_len` query
parameters and rejects unsupported query parameters such as `path`. Missing
dataset names return 404. Non-CSV datasets return 400. Missing local CSV files
return 200 with `exists: false` and a warning.

The local dashboard exposes this profile endpoint as a dataset table action for
local CSV entries. It uses the current custom experiment input/output window
settings so users can quickly see whether a saved CSV has enough rows and
required columns before launching training.

## Current Limitations

- Training still uses local CSV paths; public entries become trainable only
  after explicit preparation.
- No parquet support.
- Remote download is supported only for curated rows with direct URLs and known
  archive formats.
- Catalog entries are discovery/config-generation metadata and do not
  automatically launch feature-aware training.
- No automatic training from catalog metadata.
- No user-submitted arbitrary path profiling through the API.
