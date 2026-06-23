"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ts_platform import __version__
from ts_platform.api.middleware import install_hardening_middleware
from ts_platform.api.routes import datasets, demo, experiments, jobs, models, predict, tools
from ts_platform.api.settings import APISettings
from ts_platform.data.catalog_loader import register_dataset_catalog
from ts_platform.runner.devices import device_status

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).with_name("static")


def create_app(settings: APISettings | None = None) -> FastAPI:
    """Create the FastAPI application."""

    resolved_settings = settings or APISettings.from_env()
    _load_local_catalogs(resolved_settings)

    app = FastAPI(title="TS Platform", version=__version__, lifespan=_lifespan)
    install_hardening_middleware(app, resolved_settings)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, **device_status()}

    @app.get("/ui", include_in_schema=False)
    @app.get("/ui/", include_in_schema=False)
    def dashboard_ui() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/ui/static", StaticFiles(directory=STATIC_DIR), name="ui-static")

    app.include_router(datasets.router)
    app.include_router(demo.router)
    app.include_router(models.router)
    app.include_router(experiments.router)
    app.include_router(jobs.router)
    app.include_router(predict.router)
    app.include_router(tools.router)
    return app


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Clean up app-level resources on shutdown."""

    try:
        yield
    finally:
        jobs.shutdown_job_runner()


def _load_local_catalogs(settings: APISettings) -> None:
    """Load optional local dataset catalog files without blocking app startup."""

    for catalog_path in sorted(Path().glob(settings.dataset_catalog_glob)):
        try:
            register_dataset_catalog(catalog_path)
        except (OSError, ValueError) as exc:
            logger.warning("Skipping dataset catalog %s: %s", catalog_path, exc)


app = create_app()
