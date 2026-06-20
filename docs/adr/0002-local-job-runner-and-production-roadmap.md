# ADR 0002: Local Job Runner and Production Roadmap

## Status

Accepted

## Context

The project first introduced asynchronous API jobs with a local
`ThreadPoolExecutor` runner because it has no external dependency, is easy to
test, works well for local demos, is fast to implement, and aligns with the MVP
scope.

The project did not go directly to Redis, Celery, or Kubernetes because those
options add operational burden, introduce more moving parts, are not necessary
for the current milestone, and make the project harder for coursework and demo
users to run locally.

## Decision

- Keep the local JobRunner for the MVP.
- Preserve the `/jobs` API surface.
- Document the durable queue roadmap.
- Make future job backends replaceable behind the service layer.
- Do not implement durable queue infrastructure in this phase.

## Consequences

Positive:

- Simple.
- Testable.
- No infrastructure.
- Works locally.

Negative:

- Not durable.
- No restart recovery.
- No real kill for running Python threads.
- No distributed workers.

## Migration Path

- SQLite queue.
- Separate worker.
- Redis/RQ or Celery.
- Kubernetes Jobs.
