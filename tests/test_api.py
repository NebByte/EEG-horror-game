"""API smoke tests over the full request lifecycle using the mock provider."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from engine.main import app

PREFIX = "/v1"


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _fake_chunk(arousal_channels: float = 20.0):
    now = time.time()
    samples = []
    for i in range(256):
        # A simple oscillation so the FFT has content.
        val = arousal_channels
        samples.append(
            {
                "t": now + i / 256.0,
                "channels": {ch: val for ch in ("AF3", "AF4", "F3", "F4")},
            }
        )
    return {"sample_rate_hz": 256.0, "samples": samples}


def test_health(client):
    r = client.get(f"{PREFIX}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_full_lifecycle(client):
    # Create
    r = client.post(f"{PREFIX}/sessions", json={"seed": {"theme": "cabin"}})
    assert r.status_code == 201
    sid = r.json()["id"]

    # Generate assets (background task runs inline within TestClient).
    r = client.post(f"{PREFIX}/sessions/{sid}/generate")
    assert r.status_code == 200

    r = client.get(f"{PREFIX}/sessions/{sid}/assets")
    assert r.status_code == 200
    assert len(r.json()["characters"]) >= 1

    # Push EEG -> get affect + directive.
    r = client.post(f"{PREFIX}/sessions/{sid}/eeg", json=_fake_chunk())
    assert r.status_code == 200
    body = r.json()
    assert "affect" in body
    assert 0.0 <= body["intensity"] <= 1.0


def test_missing_session_404(client):
    r = client.get(f"{PREFIX}/sessions/does-not-exist")
    assert r.status_code == 404
