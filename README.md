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
`results.json` under `runs/`.

## Configuration

Training is driven by YAML or JSON. The example config declares:

- `experiment`: run name, output directory, seed, overwrite behavior.
- `data`: dataset name, split ratios, sequence lengths, scaler, and synthetic
  dataset parameters.
- `model`: registered model name and model-specific parameters.
- `training`: epochs, learning rate, optimizer, device, and checkpoint policy.
- `evaluation`: metric names.

See [configs/examples/simple_forecast.yaml](configs/examples/simple_forecast.yaml).

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
