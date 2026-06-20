# Model Zoo

The model zoo contains lightweight forecasting baselines that all share the
same `BaseForecastModel` contract:

- Input: `[batch, input_len, input_dim]`
- Output: `[batch, output_len, target_dim]`
- Forecast style: direct multi-step forecast in a single forward pass

Phase 12D migrates every built-in model to `input_dim` and `target_dim`.
Target-only configs still use the old `num_features` constructor path, where
`input_dim == target_dim == num_features`.

All models are available through `MODEL_REGISTRY`, so they can be selected from
training configs, compare configs, the CLI, `Trainer`, and `CompareRunner`.

## Models

| Registry name | Class | Main parameters | Typical use |
| --- | --- | --- | --- |
| `naive` | `NaiveLastValueModel` | none | Last-value sanity baseline. |
| `moving_average` | `MovingAverageForecastModel` | `window_size` | Smooth local-level baseline. |
| `seasonal_naive` | `SeasonalNaiveForecastModel` | `season_length` | Simple repeating-season baseline. |
| `linear` | `LinearForecastModel` | none | Small trainable direct projection. |
| `mlp` | `MLPForecastModel` | `hidden_sizes`, `dropout` | Nonlinear trainable baseline. |
| `rnn` | `RNNForecastModel` | `hidden_size`, `num_layers`, `dropout`, `bidirectional` | Vanilla recurrent baseline. |
| `gru` | `GRUForecastModel` | `hidden_size`, `num_layers`, `dropout`, `bidirectional` | Gated recurrent baseline. |
| `lstm` | `LSTMForecastModel` | `hidden_size`, `num_layers`, `dropout`, `bidirectional` | LSTM recurrent baseline. |
| `tcn` | `TCNForecastModel` | `hidden_channels`, `num_layers`, `kernel_size`, `dropout` | Lightweight temporal-conv baseline. |

## Recurrent Models

`rnn`, `gru`, and `lstm` use PyTorch recurrent encoders with
`batch_first=True`. The encoder reads the full input history, including any
feature columns in `input_dim`; the final hidden state from the final recurrent
layer is selected, and a linear layer projects that hidden representation to
`output_len * target_dim`.

Common parameters:

- `hidden_size`: positive integer, default `32`.
- `num_layers`: positive integer, default `1`.
- `dropout`: float where `0 <= dropout < 1`, default `0.0`. It is only passed
  to PyTorch when `num_layers > 1`.
- `bidirectional`: boolean, default `false`. When enabled, the final forward
  and backward hidden states are concatenated before projection.

These models do not use an autoregressive decoder; the forecast horizon is
produced by one direct projection.

## TCN Model

`tcn` converts input tensors from `[batch, input_len, input_dim]` to
`[batch, input_dim, input_len]`, then applies a stack of Conv1d blocks. Each
block uses:

- `Conv1d`
- `ReLU`
- `Dropout` or identity when dropout is `0`

Dilation grows as `2 ** layer_index`. Each convolution uses
`padding = dilation * (kernel_size - 1)` and crops the right side back to the
original length, which gives a simple causal-ish baseline without future-step
leakage. The final hidden time step is projected to the full output horizon.

Parameters:

- `hidden_channels`: positive integer, default `32`.
- `num_layers`: positive integer, default `3`.
- `kernel_size`: positive integer, default `3`.
- `dropout`: float where `0 <= dropout < 1`, default `0.0`.

## Compare Example

Run all built-in baselines on a tiny synthetic dataset:

```bash
py -m ts_platform.cli.main compare --config configs/examples/compare_model_zoo.yaml
```

The example config uses one epoch, small hidden sizes, `primary_metric: mae`,
`continue_on_error: true`, and `include_scaled_metrics: false` so it can act as
a quick CPU smoke test for model registration and compare output.

## Future Exogenous Feature Support

The planned exogenous feature interface separates model input width from target
output width:

- `input_dim = len(target_cols) + len(feature_cols)`
- `target_dim = len(target_cols)`

Trainable models support feature-aware forward paths:

- `linear`
- `mlp`
- `rnn`
- `gru`
- `lstm`
- `tcn`

These models consume the full `input_dim` history and project to
`output_len * target_dim`.

`BaseForecastModel.validate_input()` checks that model inputs are shaped
`[batch, input_len, input_dim]`, and `target_slice()` returns the target-history
prefix for baselines that should ignore feature columns.

Statistical baselines remain target-only by default:

- `naive`
- `moving_average`
- `seasonal_naive`

When features are present, these baselines ignore the feature slice and
forecast only from target history. Full feature-aware training remains blocked
until Trainer, evaluator, and checkpoint integration lands. See
[exogenous_features_design.md](exogenous_features_design.md) for the detailed
design.

## Current Limitations

- Probabilistic forecasting is not supported.
- Full feature-aware training is not supported yet; model forwards can consume
  `input_dim != target_dim`, but Trainer, evaluator, and checkpoint integration
  remain deferred.
- Recurrent models use direct projection, not an autoregressive decoder.
- TCN is a lightweight baseline, not a complex SOTA implementation.
- The model zoo is designed for local CPU smoke tests and simple comparisons;
  it does not add distributed training or scheduling.

## Add a Model

1. Create a model class that subclasses `BaseForecastModel`.
2. Validate model-specific parameters in `__init__`.
3. Implement `forward(x)` for `x` shaped `[batch, input_len, input_dim]`.
4. Return predictions shaped `[batch, output_len, target_dim]`.
5. Register the model in its module with `MODEL_REGISTRY.register(...)`.
6. Import the module from `ts_platform.models.__init__` so CLI discovery loads
   it.
7. Add shape tests, parameter validation tests, and a tiny trainer or compare
   smoke test.
