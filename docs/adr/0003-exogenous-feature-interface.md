# ADR 0003: Exogenous Feature Interface

## Status

Accepted

## Context

The current platform assumes every model sees and predicts the same feature
dimension:

```text
x: [batch, input_len, num_features]
y: [batch, output_len, num_features]
```

That shape is sufficient for target-only forecasting. It does not support
exogenous variables such as weather, holidays, prices, promotions, or sensor
context that should be model inputs without becoming forecast targets.

`CSVForecastDataset`, `BaseForecastModel`, `Trainer`, `Evaluator`, and
checkpoint schema currently share the `num_features` boundary. `feature_cols`
is accepted only as a CSV parameter shape and then rejected when non-empty.

## Decision

The platform will migrate toward separate input and target dimensions:

- `target_cols`: columns to predict.
- `feature_cols`: exogenous input-only columns.
- `input_dim = len(target_cols) + len(feature_cols)`.
- `target_dim = len(target_cols)`.
- `x` contains target history plus feature history.
- `y` contains future target values only.
- Metrics are target-only.
- Feature-aware trainable models may consume feature columns.
- Target-only baselines ignore feature columns and use the target slice.

Future `BaseForecastModel` instances should store `input_len`, `output_len`,
`input_dim`, and `target_dim`. During migration, `num_features` remains a
compatibility alias for `target_dim` in target-only configs.

The scaler boundary will split target scaling from feature scaling. The target
scaler owns target inverse transforms and original-scale metrics. The feature
scaler only transforms feature inputs and is never used for metrics.

## Consequences

Positive:

- Supports more realistic business and sensor forecasting datasets.
- Keeps metric semantics clear because only targets are evaluated.
- Preserves old target-only configs through a staged migration.
- Lets statistical baselines remain useful as target-only references.

Negative:

- Requires a model interface migration from `num_features` to
  `input_dim`/`target_dim`.
- Adds scaler complexity.
- Requires checkpoint schema expansion and compatibility checks.
- Expands test coverage across data, scaler, model, runner, and compare paths.

## Migration Plan

Phase 12A: Data schema and ForecastBatch migration

- Introduce target/input dimension terminology.
- Add optional batch metadata structure.
- Preserve target-only behavior.

Phase 12B: `CSVForecastDataset` feature_cols support

- Validate target and feature columns separately.
- Build `x` from target history plus feature history.
- Keep `y` target-only.

Phase 12C: Scaler split support

- Add target scaler and feature scaler plumbing.
- Fit both only from training-period values.
- Inverse-transform only target tensors.

Phase 12D: Model interface migration

- Move trainable models to `input_dim` inputs and `target_dim` outputs.
- Keep target-only statistical baselines on the target slice.

Phase 12E: Trainer/Evaluator/checkpoint integration

- Thread dimensions, columns, and scaler states through training.
- Expand checkpoint and resume compatibility validation.
- Record result metadata for target and feature columns.

Phase 12F: Compare/model zoo exogenous smoke tests

- Add feature-aware compare and train smoke tests.
- Keep existing target-only model zoo checks green.
