# Changelog

## Unreleased

### Added

- Configuration-driven training.
- CSV dataset support.
- Dataset catalog loading.
- Dataset profiling CLI/API.
- Config generation from catalog.
- Model zoo baselines:
  - naive
  - moving_average
  - seasonal_naive
  - linear
  - mlp
  - rnn
  - gru
  - lstm
  - tcn
- Compare runner and leaderboard.
- Results API and CLI readers.
- Artifact manifests.
- Safe artifact download API and CLI.
- Local jobs API.
- SQLite JobStore prototype.
- Worker-once and worker-loop.
- Job attempts/events.
- Retry and timeout operations.
- Feature-aware CSV batches.
- Split target/feature scaler support.
- Feature-aware model interface.
- Feature-aware training.
- Checkpoint schema v2.
- Feature-aware compare and leaderboard metadata.

### Changed

- Checkpoint schema now supports v1 and v2.
- Leaderboard rows include data metadata.
- BaseForecastModel supports input_dim / target_dim.
- CSV data path remains target-only for metrics.

### Security / Safety

- API train output_dir is constrained to safe runs root.
- Artifact download is manifest-bound and run-dir-bound.
- Checkpoint download remains disabled by default.
- Dataset API profile does not accept arbitrary path query.
- Job IDs / run IDs / artifact names are safe path components.
