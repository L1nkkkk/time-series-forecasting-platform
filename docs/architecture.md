# Architecture

## Design Lineage

This project references BasicTS at the idea level: separate dataset, scaler,
metric, runner, model, and config responsibilities, then bind them through a
configuration-driven runner. The implementation in this repository is original.

BasicTS is Apache-2.0 licensed. Because this MVP does not copy BasicTS source
code, no third-party notice is required beyond documenting the design reference.

## Module Responsibilities

- `config`: Load YAML or JSON and validate all user input with Pydantic.
- `data`: Define forecasting datasets, dataset registries, catalog metadata,
  catalog loading, split helpers, and transforms.
- `scaler`: Normalize and inverse-normalize tensors, and serialize scaler state
  for checkpoints.
- `models`: Define forecasting models and model registry.
- `metrics`: Calculate losses and evaluation metrics.
- `runner`: Orchestrate training, validation, testing, checkpoint save/restore,
  resume, and evaluator calls.
- `experiment`: Create run directories, write logs, save configs, collect
  reproducibility metadata, and record results.
- `cli`: Parse command-line input and delegate to the runner.
- `api`: Expose platform endpoints without embedding training business logic.

## Component Flow

```text
YAML/JSON config
  -> Config loader and schema
  -> Dataset registry + catalog
  -> Split datasets by window policy for synthetic data or time policy for CSV
  -> Scaler fit on train split or restore from checkpoint
  -> Model registry
  -> Model/optimizer restore when resume_from is configured
  -> Trainer loop
  -> Evaluator inverse-transforms predictions for original-scale metrics
  -> Versioned checkpoint + config snapshot + results
```

## Training Flow

1. Load and validate config.
2. Set random seed.
3. Build dataset from registry.
4. Split into train, validation, and test datasets.
5. For CSV datasets, validate local data, handle missing values, sort by time
   when configured, split raw rows into train/validation/test periods, and then
   generate sliding windows inside each split.
6. If `resume_from` is set, load checkpoint, validate compatibility, and
   restore scaler/model/optimizer state. Otherwise fit scaler on the training
   split.
7. Wrap splits with transforms and build deterministic DataLoaders.
8. Build or restore model using sequence lengths and feature count.
9. Train from `checkpoint epoch + 1` through the target final epoch.
10. Evaluate validation metrics after every epoch when validation data exists.
11. Evaluate test metrics and record final results.
12. Save a schema-versioned checkpoint containing config, model, optimizer,
    scaler, metrics, and environment metadata.

## Data Flow

Datasets yield dictionaries with:

- `x`: history tensor shaped `[input_len, num_features]`
- `y`: target tensor shaped `[output_len, num_features]`

DataLoader batches become:

- `x`: `[batch, input_len, num_features]`
- `y`: `[batch, output_len, num_features]`

Models must return predictions shaped like `y`.

Evaluation receives scaled model outputs and scaled targets. It computes
original-scale metrics by inverse-transforming both predictions and targets
before calling the metrics registry. When configured, scaled-space metrics are
also recorded under a separate `scaled` key.

## Checkpoint and Resume Boundaries

- `runner/checkpoint.py` owns checkpoint schema validation, save/load, model
  restore, scaler restore, and optimizer state restore.
- `scaler` implementations own `state_dict` and `load_state_dict`.
- `runner/evaluator.py` owns original-scale versus scaled-space metric
  calculation.
- `Trainer` coordinates these pieces but does not encode checkpoint schema
  details.

If `val_ratio` is `0`, validation is skipped and `validation_metrics` is
`null`. Test evaluation is still required.

## CSV Data Flow

`CSVForecastDataset` loads a local CSV, validates target columns, optionally
parses and sorts a timestamp column, applies the configured missing value
policy, and uses raw-row time-based splits. Sliding windows are generated only
inside the selected split, so train/validation/test targets do not overlap.

The scaler fit path remains unchanged from the trainer perspective:
`train_dataset.scaler_fit_values()` returns only training-period target values.
This keeps validation and test periods out of scaler state.

Local catalog files are metadata documents loaded through
`data/catalog_loader.py`. They can register entries in `DATASET_CATALOG` for
discovery and API listing, but they do not replace explicit training configs.

## Registry Mechanism

Registries map string names to implementation classes or callables. This keeps
configuration files stable and prevents the trainer from importing every custom
implementation directly.

## Configuration-Driven Mechanism

All runnable experiment choices live in config files: dataset, split ratios,
scaler, model, optimizer, metrics, seed, and output location. This enables fair
comparisons because each run stores the exact config snapshot with its results.
