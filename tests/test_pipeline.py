"""Mock pipeline and orchestrator produce a coherent asset bank + directives."""

from __future__ import annotations

import pytest

from engine.experience.orchestrator import Orchestrator
from engine.generative.pipeline import MOODS, AssetPipeline
from engine.generative.mock import MockProvider
from engine.schemas import AffectState, SeedProfile


@pytest.mark.asyncio
async def test_build_bank_covers_all_moods():
    bank = await AssetPipeline(MockProvider()).build_bank(SeedProfile())
    assert len(bank.sounds) == len(MOODS)
    assert len(bank.characters) == len(MOODS)
    assert len(bank.maps) == len(MOODS)
    # Every mood should be represented in character tags.
    tagged = {tag for c in bank.characters for tag in c.tags}
    for mood in MOODS:
        assert mood in tagged


@pytest.mark.asyncio
async def test_orchestrator_spawns_under_fear_and_backs_off_under_stress():
    seed = SeedProfile()
    bank = await AssetPipeline(MockProvider()).build_bank(seed)
    orch = Orchestrator(seed)

    scared = AffectState(fear=0.8, stress=0.5, arousal=0.8, engagement=0.7)
    d = orch.step(scared, bank)
    assert d.spawn_character_id is not None

    # Sustained high stress should eventually trip the safety back-off.
    over = AffectState(fear=0.9, stress=1.0, arousal=0.95, engagement=0.9)
    directive = None
    for _ in range(10):
        directive = orch.step(over, bank)
    assert directive is not None and directive.safety_backoff
