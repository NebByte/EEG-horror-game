"""Affect model behaves monotonically as simulated arousal rises."""

from __future__ import annotations

from engine.eeg.affect import infer_affect
from engine.eeg.simulator import EEGSimulator


def _affect_at(arousal: float):
    sim = EEGSimulator(sample_rate_hz=256.0)
    # Average a few windows to reduce noise from the simulator's RNG.
    fears, stresses = [], []
    for _ in range(5):
        a = infer_affect(sim.chunk(seconds=2.0, arousal=arousal))
        fears.append(a.fear)
        stresses.append(a.stress)
    return sum(fears) / len(fears), sum(stresses) / len(stresses)


def test_fear_increases_with_arousal():
    low_fear, low_stress = _affect_at(0.1)
    high_fear, high_stress = _affect_at(0.95)

    assert high_fear > low_fear
    assert high_stress > low_stress


def test_affect_bounds():
    sim = EEGSimulator(sample_rate_hz=256.0)
    a = infer_affect(sim.chunk(seconds=2.0, arousal=0.5))
    for field in ("fear", "stress", "arousal", "engagement", "relaxation"):
        assert 0.0 <= getattr(a, field) <= 1.0
    assert -1.0 <= a.valence <= 1.0
