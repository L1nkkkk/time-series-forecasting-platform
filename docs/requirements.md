# Requirements

## Functional Requirements

- Provide a central interface for time series datasets.
- Support dataset metadata, loading, splitting, preprocessing, and caching
  extension points.
- Support user-defined datasets without changing the trainer.
- Provide a model library with a unified forecasting interface.
- Support configuration-driven training through YAML and JSON.
- Train, validate, test, checkpoint, resume in later iterations, log, and export
  experiment results.
- Compute MAE, MSE, RMSE, MAPE, and WAPE.
- Persist config snapshots and runtime environment metadata for reproducibility.
- Provide CLI access for local experiments.
- Provide a first synchronous FastAPI API for platform integration.

## Non-Functional Requirements

- Keep data, scaler, model, metrics, runner, experiment, CLI, and API boundaries
  separate.
- Use type annotations and docstrings on public classes and functions.
- Use registries for datasets, models, scalers, and metrics.
- Avoid hard-coded secrets, tokens, personal paths, and destructive overwrites.
- Return clear errors for invalid configs, missing files, and shape mismatches.
- Keep the MVP small enough to validate with unit tests and smoke tests.

## MVP Scope

- Synthetic dataset with train/validation/test splits.
- Standard and min-max scalers.
- Naive last-value, linear, and MLP forecasting models.
- Regression metrics: MAE, MSE, RMSE, MAPE, and WAPE.
- Config loader and schema validation.
- Trainer with local checkpointing, metrics output, config snapshot, and
  environment metadata.
- CLI command: `train --config`.
- Synchronous FastAPI demo endpoints.
- Pytest, Ruff, mypy, and GitHub Actions CI configuration.

## Deferred Scope

- Large public dataset downloaders and full dataset catalog automation.
- Distributed training and GPU scheduling.
- Hyperparameter search and model comparison dashboards.
- Async API training jobs with durable task queues.
- Experiment tracking integrations such as MLflow or Weights & Biases.
- Production authentication and authorization.
