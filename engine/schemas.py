"""Pydantic data contracts shared across the API, EEG, and generative layers."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# EEG ingestion
# --------------------------------------------------------------------------- #
class EEGSample(BaseModel):
    """A single multi-channel EEG sample."""

    t: float = Field(..., description="Sample timestamp (unix seconds).")
    channels: dict[str, float] = Field(
        ..., description="Microvolt reading per channel label, e.g. {'AF3': 12.4}."
    )


class EEGChunk(BaseModel):
    """A batch of samples pushed from the headset for one session."""

    sample_rate_hz: float = Field(..., gt=0)
    samples: list[EEGSample] = Field(..., min_length=1)


# --------------------------------------------------------------------------- #
# Affect / emotion state
# --------------------------------------------------------------------------- #
class BandPowers(BaseModel):
    """Relative band powers averaged across channels (each 0..1-ish)."""

    delta: float = 0.0
    theta: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0


class AffectState(BaseModel):
    """Normalised affect estimate derived from EEG. All fields in [0, 1]."""

    fear: float = 0.0
    stress: float = 0.0
    arousal: float = 0.0
    engagement: float = 0.0
    relaxation: float = 0.0
    # Valence is signed: -1 (negative) .. +1 (positive).
    valence: float = 0.0
    confidence: float = Field(
        0.0, description="How much signal (vs. noise) backs this estimate."
    )
    bands: BandPowers = Field(default_factory=BandPowers)


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #
class SeedProfile(BaseModel):
    """Player-supplied seed used to pre-generate the asset bank."""

    theme: str = Field("abandoned hospital", description="High-level horror setting.")
    fears: list[str] = Field(
        default_factory=lambda: ["darkness", "isolation"],
        description="Self-reported fears used to bias generation.",
    )
    intensity: float = Field(
        0.5, ge=0.0, le=1.0, description="Baseline intensity preference."
    )
    player_name: Optional[str] = None


class SessionCreate(BaseModel):
    seed: SeedProfile = Field(default_factory=SeedProfile)


class SessionStatus(str, Enum):
    created = "created"
    generating = "generating"
    ready = "ready"
    live = "live"
    error = "error"


class Session(BaseModel):
    id: str
    status: SessionStatus
    seed: SeedProfile
    affect: AffectState = Field(default_factory=AffectState)
    directive: "Directive | None" = None


# --------------------------------------------------------------------------- #
# Generated assets
# --------------------------------------------------------------------------- #
class AssetKind(str, Enum):
    sound = "sound"
    character = "character"
    map = "map"


class Asset(BaseModel):
    id: str
    kind: AssetKind
    name: str
    prompt: str = Field(..., description="Prompt used to generate the asset.")
    # For the prototype, media is referenced by URI (gs://, file://, http://)
    # rather than embedded. Structured payload holds model output metadata.
    uri: Optional[str] = None
    payload: dict = Field(default_factory=dict)
    # Affect tags let the orchestrator pick assets that match the target mood.
    tags: list[str] = Field(default_factory=list)


class AssetBank(BaseModel):
    """The full set of assets pre-generated for a session."""

    sounds: list[Asset] = Field(default_factory=list)
    characters: list[Asset] = Field(default_factory=list)
    maps: list[Asset] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Live directives (engine -> game client)
# --------------------------------------------------------------------------- #
class Directive(BaseModel):
    """What the engine tells the game to do right now, given current affect."""

    intensity: float = Field(0.5, ge=0.0, le=1.0)
    # Named actions the client understands, e.g. "spawn_character".
    spawn_character_id: Optional[str] = None
    ambient_sound_id: Optional[str] = None
    stinger_sound_id: Optional[str] = None
    map_mutation: Optional[str] = None
    # Human-readable reason (great for debugging / telemetry overlays).
    reason: str = ""
    # Safety flag: engine is deliberately easing off.
    safety_backoff: bool = False


Session.model_rebuild()
