"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from ts_platform import __version__
from ts_platform.api.routes import datasets, experiments, models


def create_app() -> FastAPI:
    """Create the FastAPI application."""

    app = FastAPI(title="TS Platform", version=__version__)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    app.include_router(datasets.router)
    app.include_router(models.router)
    app.include_router(experiments.router)
    return app


app = create_app()
