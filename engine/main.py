"""FastAPI application entrypoint.

Run locally:
    uvicorn engine.main:app --reload

Then open http://localhost:8000/docs for the interactive API.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from engine import __version__
from engine.api import eeg, health, sessions
from engine.config import get_settings

logging.basicConfig(level=logging.INFO)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="EEG Horror Engine",
        version=__version__,
        description=(
            "Adaptive horror-experience engine driven by live EEG affect "
            "signals. Pre-generates sound / character / map assets, then "
            "reshapes the experience in real time from measured fear & stress."
        ),
    )

    # Game clients (Unity/Unreal/web) connect cross-origin; open for the
    # prototype. Lock this down before any public deployment.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    prefix = settings.api_prefix
    app.include_router(health.router, prefix=prefix)
    app.include_router(sessions.router, prefix=prefix)
    app.include_router(eeg.router, prefix=prefix)

    @app.get("/")
    async def root() -> dict:
        return {
            "name": "eeg-horror-engine",
            "version": __version__,
            "docs": "/docs",
            "api_prefix": prefix,
        }

    return app


app = create_app()
