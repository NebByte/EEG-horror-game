"""Live Vertex AI verification — run this where GCP credentials exist.

It forces the Vertex provider, builds a real asset bank (Gemini specs + Imagen
concept art uploaded to GCS), and prints what came back so you can confirm the
media URIs are real ``gs://`` paths.

Prereqs (see docs/CREDENTIALS_SETUP.md):
    pip install -r requirements-vertex.txt
    export ASSET_PROVIDER=vertex
    export GCP_PROJECT=eeg-horror
    export GCS_BUCKET=<your-bucket>
    # auth: either GOOGLE_APPLICATION_CREDENTIALS=/path/sa.json
    #       or run in Cloud Shell / after `gcloud auth application-default login`

Run:
    python scripts/verify_vertex.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.config import get_settings
from engine.generative.pipeline import AssetPipeline, get_provider
from engine.schemas import SeedProfile


async def main() -> None:
    settings = get_settings()
    if settings.asset_provider != "vertex":
        print("ASSET_PROVIDER is not 'vertex'. Set it and re-run.")
        return

    print(
        f"project={settings.gcp_project} location={settings.gcp_location} "
        f"bucket={settings.gcs_bucket or '(none — specs only)'}"
    )

    provider = get_provider(settings)
    print(f"provider resolved to: {provider.name}")
    if provider.name != "vertex":
        print(
            "Provider fell back to mock — SDK or credentials are missing. "
            "Install requirements-vertex.txt and configure auth."
        )
        return

    seed = SeedProfile(
        theme="abandoned asylum",
        fears=["darkness", "being watched"],
        intensity=0.5,
    )

    print("\nGenerating a real asset bank (this calls Gemini + Imagen)...\n")
    bank = await AssetPipeline(provider).build_bank(seed)

    print("CHARACTERS:")
    for c in bank.characters:
        print(f"  [{','.join(c.tags)}] {c.name}")
        print(f"    uri: {c.uri}")
        print(f"    behavior={c.payload.get('behavior')} "
              f"aggression={c.payload.get('aggression')}")

    print("\nSOUNDS:")
    for s in bank.sounds:
        print(f"  [{','.join(s.tags)}] {s.name}: bpm={s.payload.get('bpm')} "
              f"layers={s.payload.get('layers')}")

    print("\nMAPS:")
    for m in bank.maps:
        print(f"  [{','.join(m.tags)}] {m.name}: rooms={m.payload.get('rooms')} "
              f"lighting={m.payload.get('lighting')}")

    with_media = [c for c in bank.characters if c.uri and c.uri.startswith("gs://")]
    print(
        f"\n✔ {len(with_media)}/{len(bank.characters)} characters have live "
        f"gs:// media URIs."
        if with_media
        else "\n⚠ No gs:// media URIs — check GCS_BUCKET and the Imagen model access."
    )


if __name__ == "__main__":
    asyncio.run(main())
