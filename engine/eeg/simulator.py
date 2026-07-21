"""Synthetic EEG source for developing and demoing without a headset.

Generates band-limited noise whose spectral balance is skewed toward a target
affect, so you can drive the whole pipeline (and the live loop) end-to-end
before any hardware is connected.
"""

from __future__ import annotations

import time

import numpy as np

from engine.schemas import EEGChunk, EEGSample


class EEGSimulator:
    """Produce synthetic :class:`EEGChunk`s biased toward a target affect."""

    def __init__(
        self,
        channels: list[str] | None = None,
        sample_rate_hz: float = 256.0,
    ) -> None:
        self.channels = channels or ["AF3", "AF4", "F3", "F4", "T7", "T8", "O1", "O2"]
        self.sample_rate_hz = sample_rate_hz
        self._rng = np.random.default_rng()

    def chunk(self, seconds: float = 2.0, arousal: float = 0.5) -> EEGChunk:
        """Return a synthetic chunk.

        ``arousal`` in [0, 1] shifts spectral energy from alpha (calm) toward
        beta/gamma (activated), which the affect model reads as higher
        arousal/stress/fear.
        """
        n = max(4, int(self.sample_rate_hz * seconds))
        t = np.arange(n) / self.sample_rate_hz
        now = time.time()

        # Amplitude weights per band shift with arousal.
        alpha_amp = 1.5 * (1.0 - arousal) + 0.2
        beta_amp = 1.5 * arousal + 0.2
        gamma_amp = 0.8 * arousal

        samples: list[EEGSample] = []
        # Precompute per-channel signals for efficiency.
        signals: dict[str, np.ndarray] = {}
        for ch in self.channels:
            sig = (
                alpha_amp * np.sin(2 * np.pi * 10 * t + self._rng.random())
                + beta_amp * np.sin(2 * np.pi * 20 * t + self._rng.random())
                + gamma_amp * np.sin(2 * np.pi * 38 * t + self._rng.random())
                + 0.3 * self._rng.standard_normal(n)
            )
            signals[ch] = sig

        for i in range(n):
            samples.append(
                EEGSample(
                    t=now + i / self.sample_rate_hz,
                    channels={ch: float(signals[ch][i]) for ch in self.channels},
                )
            )

        return EEGChunk(sample_rate_hz=self.sample_rate_hz, samples=samples)
