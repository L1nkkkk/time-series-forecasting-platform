# Leaderboard Format

Compare runs write `leaderboard.json` and `leaderboard.csv` in the compare run
directory. Both files contain one row per configured model.

## JSON Schema

`leaderboard.json` is a JSON array. Each row contains:

- `rank`: integer for successful rows, `null` for failed rows.
- `status`: `success` or `failed`.
- `model_name`: registry model name.
- `model_alias`: safe run alias such as `001_naive`.
- `model_params`: JSON string containing model parameters.
- `run_id`: Trainer run id for successful rows, `null` for failed rows.
- `run_dir`: Trainer run directory for successful rows, `null` for failed rows.
- `checkpoint_path`: final checkpoint path for successful rows, `null` for
  failed rows.
- `primary_metric`: metric used for ranking.
- `primary_metric_value`: numeric value for successful rows, `null` for failed
  rows.
- `created_at`: Trainer run timestamp for successful rows, `null` for failed
  rows.
- `error`: failure message for failed rows, `null` for successful rows.
- `test_<metric>`: one column per configured evaluation metric, populated from
  `test_metrics.original`.

## CSV Columns

`leaderboard.csv` uses the same fields in this order:

```text
rank,status,model_name,model_alias,model_params,run_id,run_dir,
checkpoint_path,primary_metric,primary_metric_value,created_at,error,
test_<metric>...
```

CSV empty cells correspond to JSON `null` values.

## Rank Rules

Successful rows are sorted ascending by `primary_metric_value`. Rank starts at
`1`. Phase 3 treats configured evaluation metrics as error metrics, where lower
is better.

## Failed Rows

When `continue_on_error: true`, failed models are appended after successful
rows. They have:

- `status: failed`
- `rank: null`
- no metric values
- no run artifact paths
- `error` containing the failure message

If all models fail, compare still writes an all-failed leaderboard. When
`continue_on_error: false`, the first failure aborts the compare run.
