# Final Report Outline

## 1. Project Background

Describe time series forecasting application scenarios and the platform goal:
a local, configuration-driven MVP for repeatable training, comparison,
inspection, and safe artifact access.

## 2. Requirements

- Dataset catalog.
- Model zoo.
- Training framework.
- Compare.
- Result management.
- Jobs.
- Safety.
- Extensibility.

## 3. Architecture

- Config.
- Data.
- Model registry.
- Trainer.
- Compare runner.
- Experiment store.
- Artifact service.
- Jobs.
- API/CLI.

## 4. Software Engineering Process

Explain the Phase 1 through Phase 13 iteration path, including hardening,
feature increments, documentation, and quality gates.

## 5. Key Technical Designs

- Split-local CSV missing handling.
- Original-scale metrics.
- Artifacts manifest.
- Safe artifact download.
- Model export and prediction.
- SQLite jobs.
- Feature-aware training.
- Checkpoint v2.
- Local dashboard and CLI-parity API tools.

## 6. Testing and Quality Assurance

- Unit tests.
- Smoke tests.
- CLI tests.
- API tests.
- Safety tests.
- Quality gates.

## 7. Results and Demo

- Model zoo compare.
- Feature-aware compare.
- Leaderboard.

## 8. Limitations

- Dashboard is local/demo-grade, not a production web application.
- No multi-user auth or per-user authorization; only optional coarse API-key
  protection.
- No multi-tenant SaaS.
- No Redis/Celery production queue.
- No probabilistic forecasting.
- No future-known covariate decoder.

## 9. Future Work

- Production dashboard.
- Production queue.
- Multi-user auth and authorization.
- Deployment.
- More datasets.
- More models.
