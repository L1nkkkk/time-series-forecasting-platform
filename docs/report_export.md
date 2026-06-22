# Dashboard Report Export

The local dashboard can export the selected experiment run as a Markdown report.
This is intended for coursework demos, project handoff notes, and quick offline
inspection.

## How To Export

1. Start the API:

   ```bash
   uvicorn ts_platform.api.app:create_app --factory
   ```

2. Open the dashboard:

   http://127.0.0.1:8000/ui

3. Select a completed train or compare run in the run library.

4. Click **Export Report** in the run detail header.

The browser downloads a `.md` file named from the experiment and run id.

## Report Contents

Train run reports include:

- run summary metadata
- original-scale test metrics
- training monitor latest/best/delta summaries
- forecast sample inventory
- data metadata
- artifact manifest summary

Compare run reports include:

- run summary metadata
- primary metric and success/failure counts
- top leaderboard rows
- artifact manifest summary

## Scope And Safety

Report export is generated in the browser from already-loaded API responses.
It does not read arbitrary files, request artifact paths, or download
checkpoints. Full JSON, CSV, YAML, and log artifacts should still be inspected
through the manifest-backed artifact preview/download controls.
