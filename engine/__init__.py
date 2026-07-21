"""EEG Horror Engine.

An adaptive horror-experience engine driven by live EEG affect signals.

Pipeline overview:
    1. Calibration + pre-generation: build an asset bank (sounds, characters,
       maps) from a seed profile using generative models (Google Cloud Vertex).
    2. Live loop: ingest EEG -> compute band powers -> infer affect
       (fear / stress / arousal / valence / engagement) -> the orchestrator
       emits directives that reshape the running experience in real time.

Everything is exposed over an HTTP + WebSocket API so a game client
(Unity / Unreal / web) can drive it.
"""

__version__ = "0.1.0"
