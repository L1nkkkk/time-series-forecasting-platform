# ADR 0001: Configuration-Driven Modular Architecture

## Status

Accepted.

## Context

The platform needs to support many datasets, scalers, models, metrics, and
training settings while remaining testable and reproducible. Researchers should
be able to compare experiments by changing configuration files rather than
rewriting orchestration code.

## Decision

Adopt a configuration-driven architecture with registries and a runner:

- Pydantic schemas validate YAML and JSON configs.
- Registries resolve datasets, scalers, models, and metrics by name.
- The runner owns orchestration of training, validation, testing, checkpointing,
  and result recording.
- Data, scaler, model, metric, runner, experiment, CLI, and API modules remain
  separate.

## Consequences

Positive:

- New datasets and models can be added without changing the trainer.
- Experiments are easier to reproduce from saved config snapshots.
- Unit tests can target small modules independently.
- API and CLI share the same runner path.

Tradeoffs:

- Config schemas need ongoing maintenance as new features appear.
- Registries add a small amount of indirection.
- The runner must keep clear extension points to avoid becoming too large.
