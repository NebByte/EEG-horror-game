"""Offline mock provider.

Deterministic, dependency-free asset generation so the whole system runs and is
testable without any cloud credentials. Output mirrors the shape a real provider
returns (prompt + structured payload + tags), just with placeholder media URIs.
"""

from __future__ import annotations

import hashlib

from engine.generative.base import AssetProvider
from engine.schemas import Asset, AssetKind, SeedProfile

_MOODS = ("dread", "panic", "unease", "relief")


def _stable_id(*parts: str) -> str:
    return hashlib.sha1("::".join(parts).encode()).hexdigest()[:12]


class MockProvider(AssetProvider):
    """Returns placeholder assets instantly."""

    name = "mock"

    async def generate_soundscape(self, seed: SeedProfile, mood: str) -> Asset:
        prompt = (
            f"Ambient horror soundscape for a {seed.theme}, mood={mood}, "
            f"evoking {', '.join(seed.fears)}."
        )
        aid = _stable_id("sound", seed.theme, mood)
        return Asset(
            id=aid,
            kind=AssetKind.sound,
            name=f"{mood}-ambience",
            prompt=prompt,
            uri=f"file://mock/sounds/{aid}.wav",
            payload={
                "bpm": 40 if mood in ("dread", "unease") else 96,
                "layers": ["drone", "whispers", "distant-metal"],
                "loopable": True,
            },
            tags=[mood, "ambient"],
        )

    async def generate_character(self, seed: SeedProfile, mood: str) -> Asset:
        prompt = (
            f"A horror character for a {seed.theme}. It embodies the player's "
            f"fear of {', '.join(seed.fears)}. Mood: {mood}."
        )
        aid = _stable_id("char", seed.theme, mood)
        return Asset(
            id=aid,
            kind=AssetKind.character,
            name=f"{mood}-stalker",
            prompt=prompt,
            uri=f"file://mock/characters/{aid}.png",
            payload={
                "description": f"A gaunt figure haunting the {seed.theme}.",
                "behavior": "stalk" if mood in ("dread", "panic") else "lurk",
                "aggression": 0.9 if mood == "panic" else 0.4,
            },
            tags=[mood, "enemy"],
        )

    async def generate_map(self, seed: SeedProfile, mood: str) -> Asset:
        prompt = f"A level layout for a {seed.theme}, tuned for {mood}."
        aid = _stable_id("map", seed.theme, mood)
        return Asset(
            id=aid,
            kind=AssetKind.map,
            name=f"{mood}-wing",
            prompt=prompt,
            uri=f"file://mock/maps/{aid}.json",
            payload={
                "rooms": 8 if mood == "dread" else 5,
                "lighting": "near-dark" if mood in ("dread", "panic") else "dim",
                "hazards": ["locked-doors", "flickering-lights"],
            },
            tags=[mood, "level"],
        )
