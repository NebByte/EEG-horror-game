"""Map EEG band powers to an affect estimate (fear / stress / arousal / ...).

The mappings below are heuristic and grounded in commonly cited EEG affect
correlates. They are deliberately transparent and tunable — the goal for the
prototype is a plausible, monotonic signal the experience layer can react to,
not clinical accuracy. Replace/augment with a trained model (see ROADMAP) later.

Correlates used:
    * Arousal      ~ beta / (alpha + theta)      (mental activation / alertness)
    * Stress       ~ beta / alpha                 (elevated beta over alpha)
    * Relaxation   ~ alpha power                  (eyes-closed / calm)
    * Engagement   ~ beta / (alpha + theta)       (task engagement index)
    * Valence      ~ frontal alpha asymmetry      (approach vs. withdrawal)
    * Fear         ~ high arousal + negative valence + gamma
"""

from __future__ import annotations

import math

from engine.eeg.bands import compute_band_powers, frontal_alpha_asymmetry
from engine.schemas import AffectState, EEGChunk


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _squash(x: float, k: float = 1.0) -> float:
    """Logistic squash of an unbounded ratio into (0, 1)."""
    return 1.0 / (1.0 + math.exp(-k * x))


def infer_affect(chunk: EEGChunk) -> AffectState:
    """Infer an :class:`AffectState` from a window of EEG samples."""
    bands = compute_band_powers(chunk)
    eps = 1e-6

    beta = bands.beta
    alpha = bands.alpha
    theta = bands.theta
    gamma = bands.gamma

    # Engagement / arousal index (Pope et al.): beta / (alpha + theta).
    engagement_ratio = beta / (alpha + theta + eps)
    arousal = _squash(engagement_ratio - 0.6, k=3.0)
    engagement = _squash(engagement_ratio - 0.5, k=2.5)

    # Stress: beta dominance over alpha.
    stress = _squash((beta / (alpha + eps)) - 1.0, k=2.0)

    # Relaxation: relative alpha power.
    relaxation = _clip01(alpha * 2.0)

    # Valence from frontal alpha asymmetry, squashed into [-1, 1].
    faa = frontal_alpha_asymmetry(chunk)
    valence = math.tanh(faa)

    # Fear: high arousal AND negative valence, boosted by gamma bursts.
    neg_valence = _clip01((-valence + 1.0) / 2.0)  # 0 (positive) .. 1 (negative)
    fear = _clip01(0.55 * arousal + 0.30 * neg_valence + 0.15 * _clip01(gamma * 4.0))

    # Confidence: total in-band power vs. a floor (very rough SNR proxy).
    in_band = beta + alpha + theta + gamma + bands.delta
    confidence = _clip01(in_band)

    return AffectState(
        fear=round(fear, 4),
        stress=round(stress, 4),
        arousal=round(arousal, 4),
        engagement=round(engagement, 4),
        relaxation=round(relaxation, 4),
        valence=round(valence, 4),
        confidence=round(confidence, 4),
        bands=bands,
    )
