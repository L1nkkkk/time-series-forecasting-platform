# Time Series Forecasting Platform

This repository contains a configuration-driven MVP for a comprehensive time
series forecasting platform. It follows the same engineering ideas highlighted
by BasicTS: datasets, scalers, models, metrics, runners, and configs are
separate modules with small public interfaces and registry-based extension
points.

The current MVP focuses on a runnable local training loop:

- Synthetic forecasting dataset.
- Naive last-value, linear, and MLP forecasting models.
- Standard and min-max scalers.
- MAE, MSE, RMSE, MAPE, and WAPE metrics.
- Config snapshots, checkpoints, metrics output, and environment metadata.
- CLI and a synchronous FastAPI demo API.
- Original-scale evaluation metrics with optional scaled-space metrics.
- Versioned checkpoints that can restore model, scaler, and optimizer state.

No BasicTS code is copied into this project.

## Installation

Use Python 3.10 or newer.

```bash
py -m pip install -e ".[dev]"
```

On Unix-like shells, replace `py` with `python`.

## Quick Start

Run the example training config:

```bash
py -m ts_platform.cli.main train --config configs/examples/simple_forecast.yaml
```

The run writes logs, a checkpoint, a config snapshot, environment metadata, and
`results.json` under `runs/simple_forecast/latest/` because the example config
sets `overwrite: true`.

## Metrics

Validation and test metrics are reported on the original data scale by default.
When `evaluation.include_scaled_metrics: true` is set, `results.json` also
stores scaled-space metrics:

```json
{
  "test_metrics": {
    "original": {"mae": 0.0},
    "scaled": {"mae": 0.0}
  }
}
```

## Configuration

Training is driven by YAML or JSON. The example config declares:

- `experiment`: run name, output directory, seed, overwrite behavior.
- `data`: dataset name, split ratios, sequence lengths, scaler, and synthetic
  dataset parameters.
- `model`: registered model name and model-specific parameters.
- `training`: epochs, learning rate, optimizer, device, and checkpoint policy.
- `evaluation`: metric names and whether to include scaled-space metrics.

See [configs/examples/simple_forecast.yaml](configs/examples/simple_forecast.yaml).

## Checkpoints and Resume

Checkpoints use schema version `1` and include:

- validated config snapshot
- model name, params, input/output lengths, feature count, and state dict
- optimizer name and state dict
- scaler name, params, and state dict
- metrics and environment metadata

Resume training by setting `training.resume_from` to a checkpoint path. The
configured `training.epochs` is the target final epoch, not the number of extra
epochs. For example, resuming an epoch-1 checkpoint with `epochs: 2` trains only
epoch 2. If the checkpoint epoch is already at or beyond the target, training is
skipped and evaluation still runs.

## Run Directory Strategy

- `overwrite: false` creates a unique run directory:
  `runs/<experiment_name>/<timestamp>_<short_id>/`.
- `overwrite: true` writes to `runs/<experiment_name>/latest/` and clears stale
  artifacts before running.

Every `results.json` includes `run_id`, `created_at`, `run_dir`, and
`experiment_name`.

## Validation Split

`val_ratio: 0` is allowed. In that case validation is skipped,
`validation_metrics` is `null`, and test metrics are still computed.

## Add a Dataset

1. Implement a `torch.utils.data.Dataset` compatible with
   `ForecastingDataset`.
2. Return batches with `x` shaped `[input_len, num_features]` and `y` shaped
   `[output_len, num_features]`.
3. Register the dataset with `DATASET_REGISTRY.register("name", DatasetClass)`.
4. Add catalog metadata through `DATASET_CATALOG.register(...)`.

See [examples/custom_dataset.py](examples/custom_dataset.py).

## Add a Model

1. Subclass `BaseForecastModel`.
2. Implement `forward(x)` for `x` shaped `[batch, input_len, num_features]`.
3. Return predictions shaped `[batch, output_len, num_features]`.
4. Register the model with `MODEL_REGISTRY.register("name", ModelClass)`.

## Run Tests

```bash
py -m pytest
ruff check .
ruff format --check .
mypy src
py -m ts_platform.cli.main train --config configs/examples/simple_forecast.yaml
```

## API Demo

```bash
uvicorn ts_platform.api.app:create_app --factory --reload
```

Available endpoints:

- `GET /health`
- `GET /datasets`
- `GET /models`
- `GET /experiments`
- `POST /experiments/train`
