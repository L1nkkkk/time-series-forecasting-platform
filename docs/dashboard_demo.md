# Dashboard Demo

## Start

```bash
uvicorn ts_platform.api.app:create_app --factory
```

Open:

http://127.0.0.1:8000/ui

## Recommended Demo Flow

1. Start the API:

   ```bash
   uvicorn ts_platform.api.app:create_app --factory
   ```

2. Open:

   http://127.0.0.1:8000/ui

3. Click Refresh in Overview.

4. Run `csv_feature_forecast` to show feature-aware training.

5. Run `compare_feature_forecast` to show model comparison.

6. Inspect leaderboard columns:

   - `feature_aware`
   - `input_dim`
   - `target_dim`
   - `feature_dim`
   - `target_cols`
   - `feature_cols`

7. Load artifacts and leaderboard from the Artifacts / Leaderboard Preview
   panel.

8. Optionally show Jobs as the local async prototype.

## Timing Notes

- `simple_forecast` is fastest.
- `csv_feature_forecast` is good for a feature-aware single-run demo.
- `compare_feature_forecast` is the most complete demo but can take longer.
- For live presentations, run `compare_feature_forecast` before the
  presentation if time is limited, then use Load Leaderboard.

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
