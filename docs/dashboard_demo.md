# Dashboard Demo

Chinese version: [dashboard_demo.zh-CN.md](dashboard_demo.zh-CN.md)

## Start

```bash
uvicorn ts_platform.api.app:create_app --factory
```

Open:

http://127.0.0.1:8000/ui

The dashboard opens in Chinese by default. Use the language button in the top
bar to switch between Chinese and English. Top navigation splits the dashboard
into Overview, Datasets, Results, Custom Experiment, Monitor, and Jobs pages so
demos do not require scrolling through one long page. The Datasets page keeps
the catalog behind a filtered dropdown selector, and the Jobs page can submit
whitelisted demo configs as asynchronous local jobs. The Monitor page
automatically tracks queued and running jobs, including the paced
`ideal_training_30min_demo` training job. The Jobs page also exposes local
CLI-parity tools for config-path execution, model-export prediction,
CSV/catalog profiling, catalog-to-config generation, SQLite job maintenance,
and bounded local worker runs. The Results page exposes direct run lookup for
results, leaderboards, artifact manifests, and artifact downloads, including a
local runs-root override for CLI-style inspection. The Jobs page exposes local
job backend, jobs-root, SQLite DB, and runs-root settings for job and worker
commands.

## Recommended Demo Flow

1. Start the API:

   ```bash
   uvicorn ts_platform.api.app:create_app --factory
   ```

2. Open:

   http://127.0.0.1:8000/ui

3. Use the top navigation to move between Overview, Datasets, Results, Custom
   Experiment, Monitor, and Jobs.

4. Click Refresh in Overview.

5. Open Datasets, filter if needed, choose one dataset from the dropdown, and
   inspect its source/detail card.

6. Run `csv_feature_forecast` to show feature-aware training.

7. Run `compare_feature_forecast` to show model comparison.

8. In Jobs, submit `ideal_training_30min_demo` as an async local job. The UI
   switches to Monitor and refreshes progress automatically for about 30 minutes.

9. Submit `ideal_target_demo` to prepare ETTh1 and run the compare leaderboard
   flow.

10. Use Config File Runner to run or submit a local YAML/JSON train or compare
   config path.

11. Use Dataset CLI Tools to profile a CSV path, profile a catalog, or generate
    a config from a catalog entry.

12. Select a completed training run and use Model Export Prediction with pasted
    JSON or a selected values JSON file to verify inference handoff.

13. Inspect leaderboard columns:

   - `feature_aware`
   - `input_dim`
   - `target_dim`
   - `feature_dim`
   - `target_cols`
   - `feature_cols`

14. Inspect the W&B-inspired training monitor with per-metric panels, smoothing,
   latest/best/delta summaries, and point tooltips.

15. Use Run Lookup on the Results page to load `results.json`, leaderboard
   rows, `artifacts.json`, or download one named artifact for an experiment/run.
   Set runs root when inspecting a non-default local run directory.

16. Inspect artifacts from the Artifacts tab. JSON, YAML, CSV, and log artifacts
   can be previewed or downloaded through the manifest-backed artifact API.
   Model export artifacts can be downloaded for inference handoff.

17. Click Export Report on a completed run to download a Markdown summary for
   the demo or final write-up.

## Timing Notes

- `simple_forecast` is fastest.
- `csv_feature_forecast` is good for a feature-aware single-run demo.
- `ideal_training_30min_demo` is paced to roughly 30 minutes and writes live
  progress/ETA fields for the Monitor page.
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
- live job monitor
- Markdown report export
- direct run lookup
- leaderboard
- artifacts
- async demo jobs
- config-path train/compare execution
- model-export prediction
- CSV/catalog profiling and catalog-to-config generation
- SQLite retry/stale/worker controls

## Demo Buttons

- `simple_forecast`: synthetic target-only training.
- `csv_forecast`: local CSV target-only training.
- `csv_feature_forecast`: local CSV feature-aware training.
- `ideal_training_30min_demo`: paced ETTh1 DLinear training for live monitoring.
- `ideal_target_demo`: ETTh1 compare job for the ideal target leaderboard.
- `compare_feature_forecast`: feature-aware compare with leaderboard metadata.

## Safety

- Demo endpoints only use whitelist configs.
- Config-path, catalog-path, CSV-path, and model-export path tools are local
  trusted-path utilities intended for the desktop/demo environment.
- Run Lookup and job CLI settings can point at local runs roots, jobs roots, and
  SQLite DB files for desktop/demo parity.
- `output_dir` is still constrained to the runs root.
- Checkpoint download remains blocked by default.
