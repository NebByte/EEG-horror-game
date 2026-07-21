"""Session lifecycle + asset pre-generation endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException

from engine.experience.state import session_store
from engine.generative.pipeline import AssetPipeline
from engine.schemas import (
    AssetBank,
    Directive,
    Session,
    SessionCreate,
    SessionStatus,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sessions", tags=["sessions"])


async def _generate_bank(session_id: str) -> None:
    """Background task: build the asset bank and attach it to the session."""
    rec = session_store.get(session_id)
    if rec is None:
        return
    try:
        session_store.set_status(session_id, SessionStatus.generating)
        pipeline = AssetPipeline()
        bank = await pipeline.build_bank(rec.session.seed)
        session_store.set_bank(session_id, bank)
        logger.info("Asset bank ready for session %s", session_id)
    except Exception:  # noqa: BLE001
        logger.exception("Asset generation failed for session %s", session_id)
        session_store.set_status(session_id, SessionStatus.error)


@router.post("", response_model=Session, status_code=201)
async def create_session(body: SessionCreate) -> Session:
    """Create a session from a seed profile."""
    return session_store.create(body.seed)


@router.post("/{session_id}/generate", response_model=Session)
async def generate_assets(session_id: str, background: BackgroundTasks) -> Session:
    """Kick off (async) pre-generation of the asset bank for a session."""
    rec = session_store.get(session_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="session not found")
    background.add_task(_generate_bank, session_id)
    session_store.set_status(session_id, SessionStatus.generating)
    return rec.session


@router.get("/{session_id}", response_model=Session)
async def get_session(session_id: str) -> Session:
    rec = session_store.get(session_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="session not found")
    return rec.session


@router.get("/{session_id}/assets", response_model=AssetBank)
async def get_assets(session_id: str) -> AssetBank:
    rec = session_store.get(session_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="session not found")
    if rec.bank is None:
        raise HTTPException(status_code=409, detail="assets not generated yet")
    return rec.bank


@router.get("/{session_id}/directive", response_model=Directive)
async def get_directive(session_id: str) -> Directive:
    """Return the most recent directive (empty default before any EEG data)."""
    rec = session_store.get(session_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="session not found")
    return rec.session.directive or Directive(reason="no affect data yet")
