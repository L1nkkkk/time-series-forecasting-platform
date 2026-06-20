# Exogenous Features Design

## Motivation

Many real forecasting problems need inputs beyond the target history. Weather,
holiday flags, prices, promotions, sensor context, load indicators, and other
auxiliary variables can explain future target movement even when they are not
themselves forecast targets.

The current trainable platform supports CSV `target_cols` only. That is enough
for target-only baselines and early model comparison, but full training still
cannot represent datasets where the model should learn from exogenous columns
without predicting those columns. Phase 12B adds the CSV dataset/batch layer for
those columns, and Phase 12C adds dataset-level split target/feature scaling
while Phase 12D migrates the model forward interface. Trainer, evaluator, and
checkpoint migration remain later phases.

## Current State

CSV configs declare target columns through `data.params.target_cols`.
`CSVForecastDataset` reads those columns, validates them as numeric targets,
and target-only configs yield samples shaped as:

```text
x: Tensor[input_len, num_features]
y: Tensor[output_len, num_features]
```

For target-only configs, `num_features == len(target_cols)`. Model constructors
receive `num_features`, and models are expected to return predictions with the
same last dimension as `y`.

Phase 12B enables `feature_cols` in `CSVForecastDataset` only. The dataset
validates numeric feature columns, builds `x` from target history plus feature
history, keeps `y` target-only, and exposes `target_x`, `feature_x`, and column
metadata for feature-aware samples. `num_features` remains a compatibility
alias for `target_dim`.

Phase 12A started the implementation by adding compatibility dimensions without
enabling exogenous runtime behavior. Target-only datasets expose
`input_dim == target_dim == num_features`, `ForecastBatch` reserves optional
`target_x`, `feature_x`, and `metadata` fields, and model construction can
resolve either old `num_features` arguments or target-only
`input_dim`/`target_dim` arguments.

Phase 12C enables `ScaledForecastingDataset` to scale feature-aware samples
with `FeatureAwareScalerBundle`. The target scaler transforms `target_x` and
`y`, the feature scaler transforms `feature_x`, and `x` is reconstructed from
the scaled slices.

Phase 12D enables model construction and forward passes for `input_dim !=
target_dim`. Trainable models consume the full `input_dim`, while statistical
baselines use `target_slice()` and ignore feature columns. Trainer still fails
clearly with `feature-aware training is not implemented until Phase 12E`, so
full Trainer, evaluator, and checkpoint integration remain later phases.

## Proposed Semantics

Future CSV support should separate forecast targets from input-only features:

- `target_cols`: columns that the platform predicts.
- `feature_cols`: exogenous columns used only as model inputs.
- `x`: target history concatenated with feature history.
- `y`: future target values only.
- Metrics: computed only over `target_cols`.

The important boundary is that `feature_cols` are never part of `y`. They are
not inverse-transformed with the target scaler and they do not contribute to
original-scale metrics.

This design assumes historical exogenous values are available for the input
window. It does not add future-known covariate decoding or missing future
feature value handling.

## Tensor Shapes

Current target-only shape:

```text
x: [input_len, num_features]
y: [output_len, num_features]
batched x: [batch, input_len, num_features]
batched y: [batch, output_len, num_features]
```

Future target-plus-feature shape:

```text
target_dim = len(target_cols)
feature_dim = len(feature_cols)
input_dim = target_dim + feature_dim

x: [input_len, input_dim]
y: [output_len, target_dim]
batched x: [batch, input_len, input_dim]
batched y: [batch, output_len, target_dim]
```

When `feature_cols` is empty, `input_dim == target_dim`, and the old
target-only behavior is preserved. During migration, `num_features` should
remain a compatibility alias for `target_dim` when no exogenous columns are
configured.

## ForecastBatch Migration

Current `ForecastBatch`:

```python
{
    "x": Tensor[input_len, num_features],
    "y": Tensor[output_len, num_features],
}
```

Proposed eventual structure:

```python
{
    "x": Tensor[input_len, input_dim],
    "y": Tensor[output_len, target_dim],
    "target_x": Optional[Tensor[input_len, target_dim]],
    "feature_x": Optional[Tensor[input_len, feature_dim]],
    "metadata": Optional[dict],
}
```

Phase 12 can expose only `x` and `y` first. In that minimal migration,
`x` already contains concatenated target and feature history. `target_x`,
`feature_x`, and `metadata` can remain internal or optional until a downstream
consumer needs them.

Compatibility requirements:

- Existing target-only datasets keep yielding the same effective shapes.
- Existing trainer loops can continue reading `batch["x"]` and `batch["y"]`.
- Tests should prove old configs produce unchanged results and checkpoints.

## Scaler Strategy

Exogenous support needs separate scaling decisions for targets and features.

Target scaler:

- Fits only on training-period `target_cols`.
- Transforms target portions of `x` and all of `y`.
- Inverse-transforms model predictions and target labels.
- Defines original-scale metrics.

Feature scaler:

- Fits only on training-period `feature_cols`.
- Transforms only the feature portion of `x`.
- Is never used for `y` inverse transforms.
- Does not participate in metrics.

Proposed future config shape:

```yaml
data:
  scaler:
    target:
      name: standard
      params: {}
    features:
      name: standard
      params: {}
```

Backward-compatible config shape:

```yaml
data:
  scaler:
    name: standard
    params: {}
```

The old shape should continue to mean "target scaler". The implementation
phase can decide whether the default feature scaler is `identity` or a clone of
the target scaler. The design constraint is that target and feature scaler
state must be stored separately once features are supported.

## Model Compatibility

Models should be split into two compatibility groups.

Feature-aware trainable models:

- `linear`
- `mlp`
- `rnn`
- `gru`
- `lstm`
- `tcn`

These models accept `input_dim` and `target_dim`. Their input layers consume
`input_dim`, while their projection heads output `output_len * target_dim`.

Target-only statistical baselines:

- `naive`
- `moving_average`
- `seasonal_naive`

These baselines should default to target history only. When exogenous columns
are configured, they can ignore the feature slice and compute from
`target_x`. This keeps them useful as target-only comparison baselines.

The Phase 12D model constructor boundary is:

```python
BaseForecastModel(
    input_len=input_len,
    output_len=output_len,
    input_dim=input_dim,
    target_dim=target_dim,
)
```

During migration, `num_features` remains a compatibility alias for
`target_dim` in target-only configurations.

## Config Design

Future CSV config example:

```yaml
data:
  name: csv
  input_len: 24
  output_len: 6
  params:
    path: data/local/load.csv
    timestamp_col: timestamp
    target_cols: [load]
    feature_cols: [temperature, holiday, price]
    missing_policy: error
    sort_by_time: true
  scaler:
    target:
      name: standard
      params: {}
    features:
      name: standard
      params: {}
```

Backward compatibility rules:

- Configs without `feature_cols` keep current behavior.
- CSV datasets can validate and batch `feature_cols`, but Trainer keeps
  feature-aware configs blocked until the Trainer/evaluator/checkpoint
  integration phase.
- Old `data.scaler.name` configs remain valid.
- New nested scaler configs should be introduced with schema migration tests.

Catalog metadata can include `feature_cols` later for discovery, but catalog
loading must remain metadata-only. Catalog entries must not cause automatic
training or remote downloads.

## Checkpoint and Results Impact

Future checkpoints must record enough shape and column metadata to make resume
safe:

- `input_dim`
- `target_dim`
- `target_cols`
- `feature_cols`
- target scaler name, params, and state
- feature scaler name, params, and state

Resume validation must reject checkpoint/config mismatches in target columns,
feature columns, dimensions, model config, and scaler config.

Future `results.json` should record:

- `target_cols`
- `feature_cols`
- metric target columns
- whether feature columns were used

Metrics remain target-only. Leaderboards continue to rank by target metrics.

## Migration Plan

Phase 12A: Data schema and ForecastBatch migration

- Introduce `input_dim`, `target_dim`, and optional batch metadata types.
- Keep target-only behavior unchanged.
- Add tests for old `num_features` compatibility.
- Do not enable `feature_cols`, scaler splitting, feature-aware model forwards,
  or checkpoint schema changes.

Phase 12B: `CSVForecastDataset` feature_cols support

- Validate feature columns separately from target columns.
- Build `x` from target history plus feature history.
- Keep `y` target-only.
- Preserve split-local missing-value boundaries.
- Return optional target/feature history slices and metadata for feature-aware
  CSV samples.
- Block `ScaledForecastingDataset` and Trainer for feature-aware CSV configs
  until split target/feature scaling exists.

Phase 12C: Scaler split support

- Add target scaler and feature scaler plumbing.
- Fit both scalers from training-period values only.
- Inverse-transform only target predictions and labels.
- Support feature-aware batch scaling through `FeatureAwareScalerBundle`.
- Keep Trainer blocked for feature-aware configs until the model/evaluator and
  checkpoint paths understand `input_dim != target_dim`.

Phase 12D: Model interface migration

- Add `input_dim` and `target_dim` to the model boundary.
- Migrate trainable models to consume `input_dim` and output `target_dim`.
- Make statistical baselines target-slice only.
- Keep Trainer blocked until Phase 12E despite model-level feature-aware
  forwards working in tests.

Phase 12E: Trainer/Evaluator/checkpoint integration

- Thread dimensions, column metadata, and split scaler states through training.
- Update checkpoint save, restore, and compatibility validation.
- Record result metadata for target and feature columns.

Phase 12F: Compare/model zoo exogenous smoke tests

- Add feature-aware smoke configs.
- Verify compare works with feature-aware and target-only model classes.
- Keep target-only compare configs green.

## Testing Plan

Data tests:

- CSV config with `feature_cols` validates once the implementation phase starts.
- Missing target and missing feature columns produce distinct errors.
- `x` has `input_dim`; `y` has `target_dim`.
- Split-local missing handling still cannot cross train/validation/test bounds.

Scaler tests:

- Target scaler fits only target train values.
- Feature scaler fits only feature train values.
- Inverse transform is applied only to target predictions and labels.
- Old single-scaler configs remain valid.

Model shape tests:

- Trainable models accept `input_dim` and return `target_dim`.
- Statistical baselines ignore feature slices and return target-only output.
- Target-only configs preserve the current `num_features` behavior.

Training smoke tests:

- A tiny CSV with target and feature columns is rejected by Trainer until the
  model/evaluator/checkpoint phases are complete.
- Original-scale metrics are target-only.
- Checkpoint resume validates dimensions and column metadata.

Compare smoke tests:

- Feature-aware compare config succeeds for trainable models after
  Trainer/Evaluator/checkpoint integration.
- Target-only baselines remain available and ignore features.
- Leaderboard metrics remain target-only.

Backward compatibility tests:

- Existing target-only CSV configs still train.
- Existing synthetic configs still train.
- Existing checkpoint schema is either restored through a clear compatibility
  path or rejected with an actionable version error after a schema bump.

## Non-goals

- No probabilistic forecasting.
- No future-known covariate decoder.
- No missing future feature values support.
- No feature generation.
- No automatic holiday or calendar features.
- No remote datasets.
- No remote feature store.
- No change to Phase 11 runtime behavior.
