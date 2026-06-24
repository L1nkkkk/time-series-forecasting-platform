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
- Prepared public dataset assets with download, local CSV materialization,
  cache manifest, generated catalog, and generated train config support.
- Dataset asset CLI/API commands for prepare, cache status, and cache clearing.
- DLinear, NLinear, and PatchTST forecasting models.
- Best checkpoint tracking, early stopping, gradient clipping, and step/cosine
  learning-rate scheduler controls.
- Ideal target demo comparing linear, DLinear, NLinear, and PatchTST on ETTh1.
- Course requirement mapping documentation for the time-series deep learning
  rapid development platform topic.

### Changed

- Checkpoint schema now supports v1 and v2.
- Leaderboard rows include data metadata.
- BaseForecastModel supports input_dim / target_dim.
- CSV data path remains target-only for metrics.
- `compare_model_zoo` now includes the modern linear-family and PatchTST
  baselines.

### Security / Safety

- API train output_dir is constrained to safe runs root.
- Artifact download is manifest-bound and run-dir-bound.
- Checkpoint download remains disabled by default.
- Dataset API profile does not accept arbitrary path query.
- Job IDs / run IDs / artifact names are safe path components.
