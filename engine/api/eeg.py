"""EEG ingestion endpoints: REST batch push + WebSocket live stream."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from engine.eeg.affect import infer_affect
from engine.experience.state import session_store
from engine.schemas import AffectState, Directive, EEGChunk

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sessions", tags=["eeg"])


class EEGResponse(Directive):
    """Directive plus the affect it was derived from."""

    affect: AffectState


@router.post("/{session_id}/eeg", response_model=EEGResponse)
async def push_eeg(session_id: str, chunk: EEGChunk) -> EEGResponse:
    """Push a batch of EEG samples; get back affect + the next directive."""
    if session_store.get(session_id) is None:
        raise HTTPException(status_code=404, detail="session not found")

    affect = infer_affect(chunk)
    directive = session_store.update_affect(session_id, affect)
    return EEGResponse(**directive.model_dump(), affect=affect)


@router.websocket("/{session_id}/stream")
async def stream_eeg(websocket: WebSocket, session_id: str) -> None:
    """Bidirectional live loop.

    The client streams :class:`EEGChunk` JSON frames in; for each frame the
    engine streams back an :class:`EEGResponse` (affect + directive). This is the
    hot path the game runs during play.
    """
    await websocket.accept()

    if session_store.get(session_id) is None:
        await websocket.close(code=4404, reason="session not found")
        return

    try:
        while True:
            raw = await websocket.receive_json()
            try:
                chunk = EEGChunk.model_validate(raw)
            except Exception as exc:  # noqa: BLE001
                await websocket.send_json({"error": f"invalid chunk: {exc}"})
                continue

            affect = infer_affect(chunk)
            directive = session_store.update_affect(session_id, affect)
            await websocket.send_json(
                EEGResponse(**directive.model_dump(), affect=affect).model_dump()
            )
    except WebSocketDisconnect:
        logger.info("EEG stream closed for session %s", session_id)
