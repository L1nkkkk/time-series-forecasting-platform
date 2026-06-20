# Durable Queue Design

## Current Local Job Runner

The current job runner is a demo and research-grade local executor. It is
intentionally simple, testable, and free of external infrastructure:

- It uses `ThreadPoolExecutor`.
- It stores JSON job metadata under `runs/jobs/<job_id>/`.
- Each job directory contains `request_config.json`.
- Each job directory contains `job.json`.
- It is suitable for local demos, coursework, small research experiments, and
  smoke testing.
- It is not durable across process crashes.
- It is not multi-process.
- It is not multi-worker.
- It has no retry scheduler.
- It has no heartbeat.
- It has no true cancellation for a running Python thread.

The implementation should remain available for the MVP because it keeps local
development approachable and makes the `/jobs` API easy to exercise in tests.

## Requirements for Production Jobs

A production job system needs stronger execution and audit guarantees:

- Durable job metadata.
- Retry.
- Attempt tracking.
- Heartbeat.
- Timeout.
- Cancellation.
- Structured event log.
- Worker process.
- Concurrency limits.
- Resource limits.
- Result and artifact linkage.
- Failure audit.
- Idempotent result handling.
- Safe config snapshots.

## Candidate Backends

| Backend | Pros | Cons | Best For | Recommendation |
| --- | --- | --- | --- | --- |
| Current ThreadPoolExecutor | No external dependencies; easy tests; works locally; small implementation | Not durable; no restart recovery; no real kill; no multi-worker scheduling | Demo API, coursework, smoke tests, local research | Keep for the MVP and local development |
| SQLite-backed queue | Durable metadata on one host; simple deployment; inspectable state; no Redis dependency | Single-host focus; limited concurrent write scaling; still needs worker discipline | Next durable prototype, production-like single-host deployments | Prioritize as Phase 8A |
| RQ + Redis | Simple worker model; mature Redis queue; lower complexity than Celery | Requires Redis; less feature-rich than Celery; operational dependency | Multi-worker deployments where simple scheduling is enough | Consider after SQLite proves the service boundary |
| Celery + Redis/RabbitMQ | Powerful routing, retries, scheduling, and monitoring ecosystem | More moving parts; harder local setup; higher operational burden | Larger deployments with complex task orchestration | Defer until requirements justify complexity |
| Kubernetes Jobs | Strong process isolation; resource requests; GPU scheduling; crash handling by platform | Requires Kubernetes; slower feedback loop; more deployment design | Long-running training jobs with resource isolation needs | Treat as a longer-term resource isolation option |

Conclusion:

- Keep the current `ThreadPoolExecutor` for the MVP.
- Use a SQLite-backed queue as the next production-hardening step.
- Consider Redis/RQ or Celery after the service boundary and durable metadata
  model are proven.
- Use Kubernetes Jobs later when resource isolation, long-running training, or
  GPU scheduling become primary requirements.

## Recommended Migration Path

### Phase 8A: SQLite-backed queue prototype

Goal: Replace JSON-only local job metadata with a durable SQLite queue while
preserving the existing `/jobs` API surface.

Deliverables:

- SQLite schema for jobs, attempts, and events.
- Queue repository abstraction behind the jobs service layer.
- Migration path from current `JobRecord` shape to SQLite rows.
- API responses compatible with current `/jobs` endpoints.
- Tests for restart-safe metadata reads and corrupted-row behavior.

Non-goals:

- Redis, Celery, or Kubernetes.
- Multi-host workers.
- Authentication or authorization.
- Complex scheduling policies.

Acceptance criteria:

- Existing `/jobs` endpoints keep their response shape.
- Job metadata survives API process restart.
- Corrupt or partial queue records are handled with clear errors.
- Local test setup does not require external services.

### Phase 8B: Separate worker process

Goal: Move job execution out of the API process while keeping queue state
durable and API-compatible.

Deliverables:

- Worker CLI or process entry point.
- Queue polling or reservation protocol.
- Worker id and heartbeat fields.
- Retry on worker crash.
- Job timeout handling.
- Structured event log writes.

Non-goals:

- Distributed scheduling across machines.
- GPU isolation.
- Kubernetes deployment.
- User-facing API redesign.

Acceptance criteria:

- API can submit jobs without holding a Python training thread.
- Worker can resume queued jobs after restart.
- Lost heartbeat or timeout moves jobs to a clear terminal or retrying state.
- Event logs explain job lifecycle transitions.

### Phase 8C: Redis/RQ or Celery backend

Goal: Introduce an external queue backend only after durable single-host
semantics are proven.

Deliverables:

- Backend interface that can target SQLite or Redis-based queues.
- Operational notes for Redis/RQ and Celery.
- Worker concurrency configuration.
- Compatibility tests proving `/jobs` response stability.

Non-goals:

- Kubernetes-native orchestration.
- Full multi-tenant isolation.
- User permission model.

Acceptance criteria:

- Redis/RQ or Celery backend can run the same train and compare jobs.
- Job results and artifacts remain linked through stable paths or storage keys.
- Existing API clients do not need to change.

### Phase 8D: Kubernetes Jobs

Goal: Support resource-isolated training jobs for longer-running and GPU-backed
workloads.

Deliverables:

- Kubernetes job template design.
- Artifact storage strategy.
- Worker or controller integration plan.
- Resource request and limit guidelines.
- Failure and retry semantics mapped to `/jobs` states.

Non-goals:

- Near-term MVP implementation.
- Replacing local development workflow.
- Building a full platform scheduler.

Acceptance criteria:

- A queued platform job can map to a Kubernetes Job spec.
- Artifacts are written to shared storage or object storage.
- API status reflects Kubernetes lifecycle states in a backward-compatible way.

## Data Model Draft

`jobs`:

- `job_id`
- `job_type`
- `status`
- `created_at`
- `updated_at`
- `started_at`
- `finished_at`
- `experiment_name`
- `run_id`
- `compare_run_id`
- `result_path`
- `artifacts_path`
- `leaderboard_json_path`
- `error`
- `config_snapshot_path`

`job_attempts`:

- `attempt_id`
- `job_id`
- `status`
- `worker_id`
- `started_at`
- `finished_at`
- `error`
- `heartbeat_at`

`job_events`:

- `event_id`
- `job_id`
- `event_type`
- `created_at`
- `message`
- `payload_json`

## State Machine

States:

- `queued`
- `running`
- `succeeded`
- `failed`
- `cancel_requested`
- `cancelled`
- `retrying`
- `timed_out`

Transitions:

- `queued` -> `running` when a worker reserves a job.
- `queued` -> `cancelled` when cancellation is requested before execution.
- `running` -> `succeeded` when result and artifact links are written.
- `running` -> `failed` when execution fails and no retry remains.
- `running` -> `retrying` when execution fails and retry policy allows another
  attempt.
- `retrying` -> `queued` when the next attempt is ready.
- `running` -> `cancel_requested` when the API records a cooperative
  cancellation request.
- `cancel_requested` -> `cancelled` when the worker observes the request before
  completing useful work.
- `cancel_requested` -> `succeeded` when the underlying training call finishes
  before cancellation can be applied.
- `running` -> `timed_out` when heartbeat or timeout rules expire.
- `timed_out` -> `retrying` when retry policy allows another attempt.

Terminal states are `succeeded`, `failed`, and `cancelled`. `timed_out` may be
terminal or may move to `retrying`, depending on retry policy.

## API Compatibility

Future queue backend changes must not break the existing `/jobs` API surface:

- `POST /jobs/train`
- `POST /jobs/compare`
- `GET /jobs`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/result`
- `GET /jobs/{job_id}/logs`
- `POST /jobs/{job_id}/cancel`

New fields should be added in a backward-compatible way. Existing fields should
keep their names and meaning unless a versioned API migration is documented.

## Non-goals

This phase does not implement a durable queue. It only documents the production
job design, target state machine, data model, and migration path.
