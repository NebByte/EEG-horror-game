"""End-to-end demo: exercise the whole engine with simulated EEG, no server.

Run:  python scripts/demo_loop.py

It creates a session, pre-generates the asset bank, then feeds increasingly
"aroused" synthetic EEG through the affect model and prints the directives the
orchestrator produces — so you can watch the experience escalate and then hit
the safety back-off.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Allow running as `python scripts/demo_loop.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.eeg.affect import infer_affect
from engine.eeg.simulator import EEGSimulator
from engine.experience.state import session_store
from engine.generative.pipeline import AssetPipeline
from engine.schemas import SeedProfile


async def main() -> None:
    seed = SeedProfile(
        theme="abandoned asylum",
        fears=["darkness", "being watched"],
        intensity=0.4,
        player_name="demo",
    )
    session = session_store.create(seed)
    print(f"session: {session.id}  theme={seed.theme!r}\n")

    bank = await AssetPipeline().build_bank(seed)
    session_store.set_bank(session.id, bank)
    print(
        f"asset bank: {len(bank.sounds)} sounds, "
        f"{len(bank.characters)} characters, {len(bank.maps)} maps\n"
    )

    sim = EEGSimulator(channels=seed_channels(), sample_rate_hz=256.0)

    # Sweep arousal from calm -> terrified to watch the director react.
    for step, arousal in enumerate([0.1, 0.3, 0.5, 0.7, 0.9, 0.98, 0.99]):
        chunk = sim.chunk(seconds=2.0, arousal=arousal)
        affect = infer_affect(chunk)
        directive = session_store.update_affect(session.id, affect)
        print(
            f"[{step}] arousal_in={arousal:.2f} | "
            f"fear={affect.fear:.2f} stress={affect.stress:.2f} "
            f"eng={affect.engagement:.2f} -> "
            f"intensity={directive.intensity:.2f} "
            f"spawn={directive.spawn_character_id} "
            f"{'BACKOFF' if directive.safety_backoff else ''}"
        )


def seed_channels() -> list[str]:
    return ["AF3", "AF4", "F3", "F4", "T7", "T8", "O1", "O2"]


if __name__ == "__main__":
    asyncio.run(main())
