# Dashboard Demo

## Start

```bash
uvicorn ts_platform.api.app:create_app --factory
```

Open:

http://127.0.0.1:8000/ui

The dashboard opens in Chinese by default. Use the language button in the top
bar to switch between Chinese and English. Top navigation splits the dashboard
into Overview, Datasets, Results, Custom Experiment, and Jobs pages so demos do
not require scrolling through one long page. The Datasets page keeps the catalog
behind a filtered dropdown selector, and the Jobs page can submit whitelisted
demo configs as asynchronous local jobs.

## Recommended Demo Flow

1. Start the API:

   ```bash
   uvicorn ts_platform.api.app:create_app --factory
   ```

2. Open:

   http://127.0.0.1:8000/ui

3. Use the top navigation to move between Overview, Datasets, Results, Custom
   Experiment, and Jobs.

4. Click Refresh in Overview.

5. Open Datasets, filter if needed, choose one dataset from the dropdown, and
   inspect its source/detail card.

6. Run `csv_feature_forecast` to show feature-aware training.

7. Run `compare_feature_forecast` to show model comparison.

8. In Jobs, submit a train or compare demo config as an async local job and
   refresh the job list.

9. Inspect leaderboard columns:

   - `feature_aware`
   - `input_dim`
   - `target_dim`
   - `feature_dim`
   - `target_cols`
   - `feature_cols`

10. Inspect the W&B-inspired training monitor with per-metric panels, smoothing,
   latest/best/delta summaries, and point tooltips.

11. Inspect artifacts from the Artifacts tab. JSON, YAML, CSV, and log artifacts
   can be previewed or downloaded through the manifest-backed artifact API.

12. Click Export Report on a completed run to download a Markdown summary for
   the demo or final write-up.

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
- paged dashboard navigation
- dataset dropdown selection
- experiments
- train demos
- compare demos
- training monitor
- Markdown report export
- leaderboard
- artifacts
- async demo jobs

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
