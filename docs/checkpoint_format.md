# Checkpoint Format

Checkpoints are saved with `torch.save` and use `schema_version: 1`.

## Schema

```json
{
  "schema_version": 1,
  "epoch": 1,
  "config": {},
  "model": {
    "name": "linear",
    "params": {},
    "input_len": 24,
    "output_len": 6,
    "num_features": 2,
    "state_dict": {}
  },
  "optimizer": {
    "name": "adam",
    "state_dict": {}
  },
  "scaler": {
    "name": "standard",
    "params": {},
    "state": {}
  },
  "metrics": {},
  "environment": {}
}
```

## Fields

- `schema_version`: checkpoint schema version. Unknown versions are rejected
  with a clear error.
- `epoch`: completed epoch represented by the checkpoint.
- `config`: validated config snapshot serialized with Pydantic.
- `model`: registered model identity, construction parameters, dimensions, and
  PyTorch state dict.
- `optimizer`: optimizer name and state dict, or `null` state when the model has
  no trainable parameters.
- `scaler`: registered scaler identity, parameters, and scaler state dict.
- `metrics`: latest validation/test metrics associated with the checkpoint.
- `environment`: Python version, platform, package versions, and git commit
  when available.

## Restore Flow

1. Load checkpoint with `load_checkpoint`.
2. Reject unsupported `schema_version`.
3. Validate current config against checkpoint model/scaler/optimizer settings.
4. Restore scaler through `restore_scaler_from_checkpoint`.
5. Restore model through `restore_model_from_checkpoint`.
6. Build optimizer from current config and load checkpoint optimizer state.
7. Continue from `checkpoint epoch + 1` until `training.epochs`.

`training.epochs` is the target final epoch. If the checkpoint epoch is already
at or beyond the target, the trainer skips additional training and runs
evaluation.
