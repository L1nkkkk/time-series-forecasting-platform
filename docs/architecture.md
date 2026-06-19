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
  split helpers, and transforms.
- `scaler`: Normalize and inverse-normalize tensors.
- `models`: Define forecasting models and model registry.
- `metrics`: Calculate losses and evaluation metrics.
- `runner`: Orchestrate training, validation, testing, checkpoints, and
  evaluator calls.
- `experiment`: Create run directories, write logs, save configs, collect
  reproducibility metadata, and record results.
- `cli`: Parse command-line input and delegate to the runner.
- `api`: Expose platform endpoints without embedding training business logic.

## Component Flow

```text
YAML/JSON config
  -> Config loader and schema
  -> Dataset registry + catalog
  -> Split datasets
  -> Scaler fit on train split
  -> Model registry
  -> Trainer loop
  -> Evaluator + metrics registry
  -> Checkpoint + config snapshot + results
```

## Training Flow

1. Load and validate config.
2. Set random seed.
3. Build dataset from registry.
4. Split into train, validation, and test datasets.
5. Fit scaler on the training split and wrap all splits with transforms.
6. Build model from registry using sequence lengths and feature count.
7. Train for configured epochs.
8. Evaluate validation metrics after every epoch.
9. Save checkpoints according to policy.
10. Evaluate test metrics and record final results.

## Data Flow

Datasets yield dictionaries with:

- `x`: history tensor shaped `[input_len, num_features]`
- `y`: target tensor shaped `[output_len, num_features]`

DataLoader batches become:

- `x`: `[batch, input_len, num_features]`
- `y`: `[batch, output_len, num_features]`

Models must return predictions shaped like `y`.

## Registry Mechanism

Registries map string names to implementation classes or callables. This keeps
configuration files stable and prevents the trainer from importing every custom
implementation directly.

## Configuration-Driven Mechanism

All runnable experiment choices live in config files: dataset, split ratios,
scaler, model, optimizer, metrics, seed, and output location. This enables fair
comparisons because each run stores the exact config snapshot with its results.
