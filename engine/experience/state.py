"""In-memory session store.

Prototype-grade: holds sessions, their asset banks, and per-session
orchestrators in process memory. Swap for Redis/Firestore before running more
than one worker (see ROADMAP).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from engine.experience.orchestrator import Orchestrator
from engine.schemas import (
    AffectState,
    AssetBank,
    Directive,
    SeedProfile,
    Session,
    SessionStatus,
)


@dataclass
class SessionRecord:
    """Everything the engine tracks for one play session."""

    session: Session
    orchestrator: Orchestrator
    bank: AssetBank | None = None
    history: list[AffectState] = field(default_factory=list)


class SessionStore:
    """Thread-unsafe, single-process session registry (fine for the prototype)."""

    def __init__(self) -> None:
        self._records: dict[str, SessionRecord] = {}

    def create(self, seed: SeedProfile) -> Session:
        sid = uuid.uuid4().hex[:12]
        session = Session(id=sid, status=SessionStatus.created, seed=seed)
        self._records[sid] = SessionRecord(
            session=session, orchestrator=Orchestrator(seed)
        )
        return session

    def get(self, sid: str) -> SessionRecord | None:
        return self._records.get(sid)

    def require(self, sid: str) -> SessionRecord:
        rec = self._records.get(sid)
        if rec is None:
            raise KeyError(sid)
        return rec

    def set_status(self, sid: str, status: SessionStatus) -> None:
        self.require(sid).session.status = status

    def set_bank(self, sid: str, bank: AssetBank) -> None:
        rec = self.require(sid)
        rec.bank = bank
        rec.session.status = SessionStatus.ready

    def update_affect(self, sid: str, affect: AffectState) -> Directive:
        """Store the latest affect and advance the orchestrator one tick."""
        rec = self.require(sid)
        rec.session.affect = affect
        rec.history.append(affect)
        # Keep history bounded for the prototype.
        if len(rec.history) > 512:
            rec.history = rec.history[-512:]

        directive = rec.orchestrator.step(affect, rec.bank)
        rec.session.directive = directive
        rec.session.status = SessionStatus.live
        return directive

    def list_ids(self) -> list[str]:
        return list(self._records.keys())


# Module-level singleton used by the API layer.
session_store = SessionStore()
