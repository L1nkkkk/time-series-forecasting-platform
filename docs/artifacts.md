# Artifact Manifests

Every completed train and compare run writes `artifacts.json` in the run
directory. The manifest is a small index of files produced by that run so APIs,
CLI commands, and downstream tools can discover outputs without guessing file
names.

The manifest is read-only metadata. The API returns the manifest only; it does
not provide arbitrary file download. File reads use a separate named artifact
download API that still starts from the manifest.

## Train Manifest

Train runs use this shape:

```json
{
  "run_type": "train",
  "experiment_name": "csv_forecast",
  "run_id": "20260619T120000Z_a1b2c3",
  "run_dir": "runs/csv_forecast/latest",
  "artifacts": [
    {
      "name": "results",
      "kind": "json",
      "path": "runs/csv_forecast/latest/results.json",
      "description": "Training result payload"
    },
    {
      "name": "checkpoint",
      "kind": "checkpoint",
      "path": "runs/csv_forecast/latest/checkpoint.pt",
      "description": "Final model checkpoint"
    },
    {
      "name": "config_snapshot",
      "kind": "yaml",
      "path": "runs/csv_forecast/latest/config_snapshot.yaml",
      "description": "Validated config snapshot"
    },
    {
      "name": "environment",
      "kind": "json",
      "path": "runs/csv_forecast/latest/environment.json",
      "description": "Runtime environment metadata"
    },
    {
      "name": "train_log",
      "kind": "log",
      "path": "runs/csv_forecast/latest/train.log",
      "description": "Training log"
    }
  ]
}
```

Optional missing files are skipped instead of failing the run.

## Compare Manifest

Compare parent runs use this shape:

```json
{
  "run_type": "compare",
  "experiment_name": "compare_forecast",
  "compare_run_id": "20260619T120000Z_a1b2c3",
  "compare_run_dir": "runs/compare_forecast/latest",
  "artifacts": [
    {
      "name": "results",
      "kind": "json",
      "path": "runs/compare_forecast/latest/results.json",
      "description": "Compare result payload"
    },
    {
      "name": "leaderboard_json",
      "kind": "json",
      "path": "runs/compare_forecast/latest/leaderboard.json",
      "description": "Leaderboard rows as JSON"
    },
    {
      "name": "leaderboard_csv",
      "kind": "csv",
      "path": "runs/compare_forecast/latest/leaderboard.csv",
      "description": "Leaderboard rows as CSV"
    },
    {
      "name": "compare_config_snapshot",
      "kind": "yaml",
      "path": "runs/compare_forecast/latest/compare_config_snapshot.yaml",
      "description": "Validated compare config snapshot"
    },
    {
      "name": "environment",
      "kind": "json",
      "path": "runs/compare_forecast/latest/environment.json",
      "description": "Runtime environment metadata"
    }
  ]
}
```

Compare manifests point to the parent compare outputs. Per-model train runs
under `models/<model_alias>/latest/` also write their own train manifests.

## Safety

Manifest builders verify each artifact path resolves inside the current run
directory before writing. The manifest declares artifact names, kinds, paths,
and compatibility metadata such as `run_dir` or `compare_run_dir`; those
directory fields are not used as the download authorization boundary.

`ExperimentStore` validates `experiment_name` and `run_id`, supports `latest`
and recorded `run_id` / `compare_run_id` lookup, and resolves the physical run
directory under the fixed runs root.

`ArtifactService` adds download-time checks on top of the manifest:

- `artifact_name` must be a safe path component.
- The requested name must match one `artifacts.json` entry exactly.
- Clients never pass artifact paths.
- The manifest path must resolve inside the fixed runs root.
- The manifest path must also resolve inside the physical run directory returned
  by `ExperimentStore.resolve_run()`.
- Tampered manifest `run_dir` or `compare_run_dir` metadata cannot widen the
  allowed download scope.
- Cross-run manifest paths are rejected even when they stay under the same
  runs root.
- The file must exist and be a regular file.
- Allowed kinds are `json`, `yaml`, `csv`, and `log` by default.
- Checkpoint downloads are blocked by default because checkpoints can be large
  binary model state and may include sensitive training metadata.
- Downloadable files are limited to 5 MiB by default.

The API builds its artifact access policy from `APISettings`:

- `artifact_max_bytes`: maximum downloadable artifact size.
- `artifact_allowed_kinds`: allowed non-checkpoint artifact kinds.
- `allow_checkpoint_download`: explicit checkpoint download switch.

The CLI uses the default `ArtifactService` policy and does not expose a
checkpoint download switch.

The default media types are:

- `json`: `application/json`
- `yaml`: `text/yaml`
- `csv`: `text/csv`
- `log`: `text/plain`
- `checkpoint`: `application/octet-stream`, only when an explicit policy
  enables checkpoint downloads.

## Querying

CLI:

```bash
py -m ts_platform.cli.main show-artifacts --experiment compare_forecast --run latest
py -m ts_platform.cli.main show-artifact --experiment compare_forecast --run latest --artifact leaderboard_json
```

API:

```text
GET /experiments/{experiment_name}/{run_id}/artifacts
GET /experiments/{experiment_name}/{run_id}/artifacts/{artifact_name}
```

`show-artifacts` and `GET /artifacts` return manifest metadata. `show-artifact`
and `GET /artifacts/{artifact_name}` read a single registered artifact file.
Unknown artifact names return not found errors, unsafe names are rejected, kind
policy failures return forbidden errors, and oversized files are rejected before
the response body is sent.
