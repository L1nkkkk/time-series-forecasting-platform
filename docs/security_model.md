# Security Model

## Current Security Boundaries

The current platform uses filesystem and request validation boundaries designed
for a research/demo MVP:

- Safe path component validation.
- `experiment_name` uses safe path component validation.
- `run_id` uses safe path component validation.
- `job_id` uses safe path component validation.
- `artifact_name` uses safe path component validation.
- API overwrites `experiment.output_dir`.
- `ExperimentStore` uses a fixed runs root.
- `ArtifactService` uses manifest-based lookup.
- `ArtifactService` enforces the physical `run_dir` boundary.
- Checkpoint download is disabled by default.
- Artifact max size is enforced.
- CSV params validation.
- CSV datasets can reference local files, so production deployments need a
  dataset allowlist.
- Split-local missing value handling.
- No arbitrary artifact path input.
- No direct filesystem root query in API.
- The API currently has no authentication or authorization.
- The current platform is not a multi-tenant SaaS.
- Checkpoint loading should only use trusted sources.

Path-like API parameters are treated as identifiers, not paths. This includes
`experiment_name`, `run_id`, `job_id`, and `artifact_name`.

## Explicit Non-goals

The current platform does not support:

- Authentication.
- Authorization.
- Multi-user isolation.
- User ownership.
- Quotas.
- Secrets management.
- Sandboxing arbitrary user code.
- Production-grade dataset allowlist.
- Arbitrary remote dataset download policy.
- Model checkpoint trust verification.

## Risk Areas

- User config can trigger compute work.
- CSV config references local files.
- Checkpoint loading must be trusted-source only.
- Job runner is local process only.
- Artifacts are protected by path boundaries but not user auth.
- No rate limiting.
- No audit log yet.

## Future Security Roadmap

- Authn/authz.
- Per-user workspace.
- Run ownership.
- Artifact ACL.
- Dataset allowlist.
- Checkpoint trust policy.
- Rate limiting.
- Audit logging.
- Secret scanning.
- Request size limits.
- Worker sandboxing.

## Current Platform Classification

This platform is currently a research/demo platform, not a multi-tenant SaaS.
