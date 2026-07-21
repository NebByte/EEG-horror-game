"""Provider interface for generative asset backends.

Any backend (mock, Google Vertex, a future local model server) implements this
protocol. The pipeline depends only on this interface, so swapping providers is
a config change, not a code change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from engine.schemas import Asset, SeedProfile


class AssetProvider(ABC):
    """Abstract generative backend.

    Methods are ``async`` because real providers make network calls; the mock
    implementation simply returns immediately.
    """

    name: str = "base"

    @abstractmethod
    async def generate_soundscape(self, seed: SeedProfile, mood: str) -> Asset:
        """Generate one ambient/sound asset for a given mood tag."""

    @abstractmethod
    async def generate_character(self, seed: SeedProfile, mood: str) -> Asset:
        """Generate one character (description + optional concept art)."""

    @abstractmethod
    async def generate_map(self, seed: SeedProfile, mood: str) -> Asset:
        """Generate one map/level layout."""

    async def close(self) -> None:  # pragma: no cover - optional cleanup hook
        """Release any provider resources."""
        return None
