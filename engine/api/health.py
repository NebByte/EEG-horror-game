"""Health / readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from engine import __version__
from engine.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "asset_provider": settings.asset_provider,
        "environment": settings.environment,
    }
