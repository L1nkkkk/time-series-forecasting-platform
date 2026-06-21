# Dashboard Demo

## Start

```bash
uvicorn ts_platform.api.app:create_app --factory
```

Open:

http://127.0.0.1:8000/ui

## What It Shows

- health
- datasets
- models
- experiments
- train demos
- compare demos
- leaderboard
- artifacts
- jobs

## Demo Buttons

- `simple_forecast`: synthetic target-only training.
- `csv_forecast`: local CSV target-only training.
- `csv_feature_forecast`: local CSV feature-aware training.
- `compare_feature_forecast`: feature-aware compare with leaderboard metadata.

## Safety

- Demo endpoints only use whitelist configs.
- No arbitrary path execution.
- `output_dir` is still constrained to the runs root.
- Checkpoint download remains blocked by default.
