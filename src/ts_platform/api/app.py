"""FastAPI application factory."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI

from ts_platform import __version__
from ts_platform.api.routes import datasets, experiments, models
from ts_platform.api.settings import APISettings
from ts_platform.data.catalog_loader import register_dataset_catalog

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create the FastAPI application."""

    settings = APISettings()
    _load_local_catalogs(settings)

    app = FastAPI(title="TS Platform", version=__version__)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    app.include_router(datasets.router)
    app.include_router(models.router)
    app.include_router(experiments.router)
    return app


def _load_local_catalogs(settings: APISettings) -> None:
    """Load optional local dataset catalog files without blocking app startup."""

    for catalog_path in sorted(Path().glob(settings.dataset_catalog_glob)):
        try:
            register_dataset_catalog(catalog_path)
        except (OSError, ValueError) as exc:
            logger.warning("Skipping dataset catalog %s: %s", catalog_path, exc)


app = create_app()
