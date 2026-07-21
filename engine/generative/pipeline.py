"""Asset pre-generation pipeline.

Phase 1 of the experience: given a seed profile, fan out generation across the
four mood buckets to build a complete :class:`AssetBank` the orchestrator can
draw from at runtime.
"""

from __future__ import annotations

import asyncio
import logging

from engine.config import Settings, get_settings
from engine.generative.base import AssetProvider
from engine.generative.mock import MockProvider
from engine.schemas import AssetBank, SeedProfile

logger = logging.getLogger(__name__)

# Mood buckets we pre-generate assets for. The orchestrator selects among these
# at runtime based on live affect.
MOODS = ("unease", "dread", "panic", "relief")


def get_provider(settings: Settings | None = None) -> AssetProvider:
    """Instantiate the configured provider, falling back to mock on error."""
    settings = settings or get_settings()

    if settings.asset_provider == "vertex":
        try:
            from engine.generative.vertex import VertexProvider

            return VertexProvider(
                project=settings.gcp_project,
                location=settings.gcp_location,
                text_model=settings.vertex_text_model,
                image_model=settings.vertex_image_model,
                gcs_bucket=settings.gcs_bucket,
            )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.warning(
                "Vertex provider unavailable (%s); falling back to mock.", exc
            )
            return MockProvider()

    return MockProvider()


class AssetPipeline:
    """Builds an :class:`AssetBank` from a seed profile."""

    def __init__(self, provider: AssetProvider | None = None) -> None:
        self.provider = provider or get_provider()

    async def build_bank(self, seed: SeedProfile) -> AssetBank:
        """Generate one sound, character, and map per mood, concurrently."""
        logger.info(
            "Generating asset bank with provider=%s theme=%r",
            self.provider.name,
            seed.theme,
        )

        sound_tasks = [self.provider.generate_soundscape(seed, m) for m in MOODS]
        char_tasks = [self.provider.generate_character(seed, m) for m in MOODS]
        map_tasks = [self.provider.generate_map(seed, m) for m in MOODS]

        sounds, characters, maps = await asyncio.gather(
            asyncio.gather(*sound_tasks),
            asyncio.gather(*char_tasks),
            asyncio.gather(*map_tasks),
        )

        return AssetBank(sounds=sounds, characters=characters, maps=maps)
