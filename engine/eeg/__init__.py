"""EEG signal processing: band powers, affect inference, and a simulator."""

from engine.eeg.affect import infer_affect
from engine.eeg.bands import compute_band_powers
from engine.eeg.simulator import EEGSimulator

__all__ = ["compute_band_powers", "infer_affect", "EEGSimulator"]
