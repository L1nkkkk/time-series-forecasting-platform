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
| SQLite-backed queue | Durable metadata on one host; simple deployment; inspectable state; no Redis dependency | Single-host focus; limited concurrent write scaling; still needs worker discipline | Durable local prototype, production-like single-host deployments | Use for Phase 8A-8D |
| RQ + Redis | Simple worker model; mature Redis queue; lower complexity than Celery | Requires Redis; less feature-rich than Celery; operational dependency | Multi-worker deployments where simple scheduling is enough | Consider after SQLite proves the service boundary |
| Celery + Redis/RabbitMQ | Powerful routing, retries, scheduling, and monitoring ecosystem | More moving parts; harder local setup; higher operational burden | Larger deployments with complex task orchestration | Defer until requirements justify complexity |
| Kubernetes Jobs | Strong process isolation; resource requests; GPU scheduling; crash handling by platform | Requires Kubernetes; slower feedback loop; more deployment design | Long-running training jobs with resource isolation needs | Treat as a longer-term resource isolation option |

Conclusion:

- Keep the current `ThreadPoolExecutor` for the MVP.
- Continue hardening the SQLite-backed queue through worker observability,
  finite local loops, retry policy, and timeout policy prototypes.
- Consider Redis/RQ or Celery after the service boundary, durable metadata
  model, and local recovery semantics are proven.
- Use Kubernetes Jobs later when resource isolation, long-running training, or
  GPU scheduling become primary requirements.

## Recommended Migration Path

### Phase 8A: SQLite-backed queue prototype

Goal: Replace JSON-only local job metadata with a durable SQLite queue while
preserving the existing `/jobs` API surface.

Status: implemented as a prototype behind the current in-process
`ThreadPoolExecutor` runner.

Deliverables:

- SQLite schema for jobs and events.
- `JobStoreProtocol` abstraction behind the jobs service layer.
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
- `request_config.json` snapshots remain under `runs/jobs/<job_id>/`.

Implementation notes:

- The default backend remains JSON for compatibility.
- SQLite is selected with `APISettings.job_backend = "sqlite"`.
- SQLite stores job metadata in `runs/jobs.sqlite3`.
- `SQLiteJobStore` writes `job_events` rows for the minimum Phase 8A audit
  trail.
- Phase 8A does not write a compatibility `job.json` copy for SQLite jobs; the
  `jobs` table is authoritative.
- The current runner is still in-process. Durable metadata is not the same as
  full worker crash recovery.

### Phase 8B: Separate worker process

Goal: Move job execution out of the API process while keeping queue state
durable and API-compatible.

Status: implemented as a local prototype with `worker-once`, SQLite job
claiming, and job attempts.

Deliverables:

- Worker CLI entry point: `worker-once`.
- Queue reservation protocol: `claim_next_queued_job`.
- `job_attempts` table with worker id, timestamps, heartbeat, status, and
  error fields.
- Structured event log writes.
- API `job_execution_mode = "external_worker"` for queue-only submit.

Non-goals:

- Distributed scheduling across machines.
- GPU isolation.
- Kubernetes deployment.
- User-facing API redesign.
- Infinite daemon loop.
- Retry, timeout, or stale heartbeat recovery.

Acceptance criteria:

- API can submit jobs without holding a Python training thread.
- `worker-once` can claim and execute one queued train or compare job.
- Attempts and events explain job lifecycle transitions.
- Event logs explain job lifecycle transitions.

Implementation notes:

- `external_worker` mode requires the SQLite backend; JSON remains
  `in_process` only.
- `worker-once` exits after one job or returns `{"status": "idle"}` when there
  is no queued job.
- SQLite claiming uses a transaction with `BEGIN IMMEDIATE` to avoid the
  simplest double-claim race on one host.
- Schema migration remains future work; tables are created with
  `CREATE TABLE IF NOT EXISTS`.

### Phase 8C: Worker observability and loop prototype

Goal: Make the local SQLite worker prototype easier to inspect and operate
without introducing a true daemon or external queue infrastructure.

Status: implemented as a local prototype with SQLite-only events/attempts
APIs, read-only CLI inspection, minimal heartbeat recording, stale running job
inspection, and finite `worker-loop`.

Deliverables:

- `GET /jobs/{job_id}/events` for SQLite event rows.
- `GET /jobs/{job_id}/attempts` for SQLite attempt rows.
- `show-job-events` and `show-job-attempts` CLI commands.
- Minimal `JobWorker` heartbeat at claim, success, and failure boundaries.
- `worker-loop` with `--max-jobs`, `--max-idle-cycles`, and
  `--sleep-seconds`.
- Read-only stale running job inspection through
  `SQLiteJobStore.list_stale_running_jobs()`.

Non-goals:

- Redis, Celery, RabbitMQ, or Kubernetes.
- Infinite daemon behavior.
- Automatic stale recovery.
- Retry scheduling.
- Timeout enforcement.

Acceptance criteria:

- Existing `/jobs` endpoints keep their response shape.
- JSON backend rejects events/attempts observability endpoints clearly.
- Worker loop always has finite bounds.
- Heartbeat failures in the failure path do not hide the original job error.
- Stale inspection returns candidates only and never mutates job state.

Implementation notes:

- The observability API is intentionally SQLite-only because JSON jobs do not
  have attempts or event tables.
- Heartbeat is not periodic during long training calls; it marks key worker
  lifecycle boundaries only.
- `worker-loop` is a convenience wrapper around `JobWorker.run_once()`, not a
  production supervisor.

### Phase 8D: Retry and timeout policy prototype

Goal: Define and test local retry, timeout, and stale heartbeat behavior before
introducing an external queue backend.

Status: implemented as an explicit local prototype for SQLite jobs.

Deliverables:

- `RetryPolicy` with `max_attempts` and `stale_after_seconds`.
- `timed_out` and `retrying` job states.
- Explicit stale job timeout marking through API, CLI, and store methods.
- Explicit retry for `failed`, `timed_out`, and `cancelled` jobs.
- Conflict behavior for non-retryable status and max-attempt exhaustion.
- Tests for retry exhaustion, timeout classification, API, and CLI behavior.

Non-goals:

- Automatic retry scheduler.
- Retry backoff.
- Worker supervision.
- Periodic heartbeat threads.
- Multi-host scheduling.
- Kubernetes-native orchestration.
- User permission model.
- Full production daemon supervision.

Acceptance criteria:

- A stale or timed-out job can be classified without guessing.
- Retryable and terminal failures have clear state transitions.
- Existing API clients still receive compatible `JobRecord` payloads.
- `worker-once` and `worker-loop` can claim explicitly retried jobs.
- No background process mutates jobs without an explicit API, CLI, or store
  call.

### Phase 8E: Redis/RQ or Celery backend

Goal: Introduce an external queue backend only after durable single-host
semantics and local retry rules are proven.

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

### Phase 8F: Kubernetes Jobs

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
- `leaderboard_json_path`
- `artifacts_path`
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
- `failed` -> `retrying` -> `queued` when an explicit retry is accepted.
- `timed_out` -> `retrying` -> `queued` when an explicit retry is accepted.
- `cancelled` -> `retrying` -> `queued` when an explicit retry is accepted.
- `running` -> `cancel_requested` when the API records a cooperative
  cancellation request.
- `cancel_requested` -> `cancelled` when the worker observes the request before
  completing useful work.
- `cancel_requested` -> `succeeded` when the underlying training call finishes
  before cancellation can be applied.
- `running` -> `timed_out` when heartbeat or timeout rules expire.

Terminal states are `succeeded`, `failed`, `cancelled`, and `timed_out`.
`failed`, `cancelled`, and `timed_out` may move to `retrying` only through an
explicit retry operation.

## API Compatibility

Future queue backend changes must not break the existing `/jobs` API surface:

- `POST /jobs/train`
- `POST /jobs/compare`
- `GET /jobs`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/events`
- `GET /jobs/{job_id}/attempts`
- `GET /jobs/{job_id}/result`
- `GET /jobs/{job_id}/logs`
- `POST /jobs/{job_id}/cancel`

New fields should be added in a backward-compatible way. Existing fields should
keep their names and meaning unless a versioned API migration is documented.

## Non-goals

This design does not require Redis, Celery, RabbitMQ, Kubernetes, a web
frontend, user management, distributed training, or new model work. Phase 8A
implements SQLite metadata and events. Phase 8B adds local worker claiming and
attempts. Phase 8C adds SQLite observability endpoints, CLI inspection,
minimal heartbeat recording, finite `worker-loop`, and read-only stale
inspection. Phase 8D adds explicit timeout and retry policy operations.
Automatic retry scheduling, retry backoff, worker supervision, periodic
heartbeat, automatic stale heartbeat handling, and crash recovery hardening
remain future work.
