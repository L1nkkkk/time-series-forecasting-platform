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
  -> Split datasets
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
5. If `resume_from` is set, load checkpoint, validate compatibility, and
   restore scaler/model/optimizer state. Otherwise fit scaler on the training
   split.
6. Wrap splits with transforms and build deterministic DataLoaders.
7. Build or restore model using sequence lengths and feature count.
8. Train from `checkpoint epoch + 1` through the target final epoch.
9. Evaluate validation metrics after every epoch when validation data exists.
10. Evaluate test metrics and record final results.
11. Save a schema-versioned checkpoint containing config, model, optimizer,
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

## Registry Mechanism

Registries map string names to implementation classes or callables. This keeps
configuration files stable and prevents the trainer from importing every custom
implementation directly.

## Configuration-Driven Mechanism

All runnable experiment choices live in config files: dataset, split ratios,
scaler, model, optimizer, metrics, seed, and output location. This enables fair
comparisons because each run stores the exact config snapshot with its results.
