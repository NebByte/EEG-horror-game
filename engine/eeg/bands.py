"""Band-power extraction from raw EEG windows.

We use a straightforward periodogram (FFT magnitude squared) and integrate power
over the classic EEG bands. This is intentionally simple and dependency-light so
the prototype runs anywhere; a production build would add proper windowing
(Welch), artifact rejection, and per-channel montage handling.
"""

from __future__ import annotations

import numpy as np

from engine.schemas import BandPowers, EEGChunk

# Standard EEG frequency bands (Hz).
BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _chunk_to_matrix(chunk: EEGChunk) -> tuple[list[str], np.ndarray]:
    """Convert a chunk into (channel_labels, samples x channels matrix)."""
    labels = list(chunk.samples[0].channels.keys())
    rows = [[s.channels.get(label, 0.0) for label in labels] for s in chunk.samples]
    return labels, np.asarray(rows, dtype=float)


def _bandpowers_1d(signal: np.ndarray, fs: float) -> dict[str, float]:
    """Return relative band powers for one channel's time series."""
    n = signal.shape[0]
    if n < 4:
        return {band: 0.0 for band in BANDS}

    # Detrend (remove DC / slow drift) then FFT periodogram.
    signal = signal - float(np.mean(signal))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = np.abs(np.fft.rfft(signal)) ** 2

    total = float(np.sum(psd[(freqs >= 0.5) & (freqs <= 45.0)])) or 1.0
    out: dict[str, float] = {}
    for band, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        out[band] = float(np.sum(psd[mask]) / total)
    return out


def compute_band_powers(chunk: EEGChunk) -> BandPowers:
    """Average relative band powers across all channels in the chunk."""
    _, matrix = _chunk_to_matrix(chunk)
    if matrix.size == 0:
        return BandPowers()

    fs = chunk.sample_rate_hz
    per_channel = [_bandpowers_1d(matrix[:, c], fs) for c in range(matrix.shape[1])]

    agg = {band: float(np.mean([pc[band] for pc in per_channel])) for band in BANDS}
    return BandPowers(**agg)


def frontal_alpha_asymmetry(chunk: EEGChunk) -> float:
    """Frontal alpha asymmetry (FAA), a classic approach/withdrawal proxy.

    Computed as ``ln(alpha_right) - ln(alpha_left)`` over frontal channels.
    More positive => relatively less right-frontal alpha => approach/positive
    affect; more negative => withdrawal/negative affect (fear, anxiety).
    Returns 0.0 when the required channels are absent.
    """
    labels, matrix = _chunk_to_matrix(chunk)
    label_idx = {label: i for i, label in enumerate(labels)}

    left = [c for c in ("AF3", "F3") if c in label_idx]
    right = [c for c in ("AF4", "F4") if c in label_idx]
    if not left or not right:
        return 0.0

    fs = chunk.sample_rate_hz

    def _alpha(cols: list[str]) -> float:
        vals = [_bandpowers_1d(matrix[:, label_idx[c]], fs)["alpha"] for c in cols]
        return float(np.mean(vals))

    eps = 1e-6
    return float(np.log(_alpha(right) + eps) - np.log(_alpha(left) + eps))
