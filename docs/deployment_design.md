# Deployment Design

## Local Development

Local development uses the editable package install and the same CLI/API code
paths as tests:

```bash
pip install -e ".[dev]"
uvicorn ts_platform.api.app:create_app --factory --reload
```

- `runs/` is the local output root.
- `runs/jobs/` stores local job metadata.
- Tests use `tmp_path` for run roots, job roots, configs, and artifacts.
- No Redis, database, object storage, or Kubernetes cluster is required.

## Demo Deployment

The recommended demo deployment is intentionally small:

- One API process.
- Local runs directory.
- Local jobs directory.
- No external services.

This is appropriate for:

- Coursework demo.
- Local research.
- Small experiments.
- Smoke testing.

The demo deployment should keep the current local `ThreadPoolExecutor`
JobRunner because it is easy to run and does not introduce operational setup.

## Production-like Deployment

A production-like deployment separates submission, execution, and storage:

- API service.
- Worker service.
- Queue backend.
- Shared artifact storage.
- Metrics/logging.
- Optional model/dataset registry service.

```text
Client
  -> API service
  -> Queue backend
  -> Worker service
  -> Shared runs/artifact storage
  -> Result APIs
```

The API service should validate requests, write job records, and expose results.
The worker service should execute train or compare jobs and write result
payloads, logs, checkpoints, and artifact manifests. The queue backend should
own durable scheduling state. Shared storage should make artifacts visible to
both workers and API readers.

## Storage

Current storage concepts:

- Runs root.
- Jobs root.
- Checkpoints.
- Config snapshots.
- `artifacts.json`.
- Result payloads.
- Leaderboard files.
- Logs.

Production-like deployments should decide whether these live on a shared
filesystem or in object storage. The API should avoid serving arbitrary paths;
it should keep using `ExperimentStore` and `ArtifactService` style boundary
checks even when storage moves behind an abstraction.

## Configuration

Current configuration:

- `APISettings` dataclass.

Future configuration:

- `pydantic-settings`.
- Environment variables.
- Deployment-specific config.
- Secrets manager.

This phase does not implement environment settings or secrets handling. It only
documents the target configuration direction.

## Failure Modes

| Failure mode | Handling strategy |
| --- | --- |
| API restart | Current local jobs keep JSON metadata but running threads are lost; future durable queues keep job state and let workers resume or retry. |
| Worker crash | Current local runner has no separate worker; future workers update heartbeat and allow retry after timeout. |
| Corrupted job metadata | Current listing skips corrupt `job.json` and direct reads return errors; SQLite should validate rows and emit repairable errors. |
| Missing artifacts | `ExperimentStore` and `ArtifactService` return clear not-found errors; production storage should preserve the same behavior. |
| Partial run | Incomplete runs are listed as incomplete; workers should write result metadata only after required artifacts are durable. |
| Checkpoint download disabled | Keep checkpoint download disabled by default unless an explicit policy enables it. |
| Queue backend unavailable | API should return a clear submission failure and avoid accepting jobs it cannot persist. |

## Deployment Roadmap

- Local demo: one FastAPI process, local `runs/`, local `runs/jobs/`, no
  external services.
- Single-host production-like: SQLite queue, local or mounted storage, explicit
  backup policy.
- Separate worker: API submits jobs, worker consumes queue, heartbeat and retry
  semantics are recorded.
- Redis/RQ or Celery: introduce only when multi-worker or richer scheduling
  requirements justify the dependency.
- Kubernetes Jobs: use for long-running training, resource isolation, and GPU
  scheduling after the API and queue abstractions are stable.
