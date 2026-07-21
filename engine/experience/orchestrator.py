"""Affect -> Directive orchestration (Phase 2 of the experience).

This is the "director": it turns the live affect estimate into a concrete
:class:`Directive` that the game client acts on — which mood the scene should
target, which character to spawn, ambient/stinger sounds, and map mutations.

Two ideas drive it:

* **Tension curve.** If the player isn't scared/engaged enough, ramp intensity
  toward the target mood. If they're highly engaged, hold and let dread build.
* **Safety back-off.** If stress stays above the configured ceiling, deliberately
  ease off (calmer mood, gentler assets) so the experience never becomes harmful.
"""

from __future__ import annotations

from engine.config import Settings, get_settings
from engine.schemas import AffectState, AssetBank, Directive, SeedProfile


def _pick(bank_list, mood: str, fallback_index: int = 0):
    """Pick an asset tagged with ``mood``; fall back to a default."""
    for asset in bank_list:
        if mood in asset.tags:
            return asset
    return bank_list[fallback_index] if bank_list else None


class Orchestrator:
    """Stateful director for one session."""

    def __init__(self, seed: SeedProfile, settings: Settings | None = None) -> None:
        self.seed = seed
        self.settings = settings or get_settings()
        # Smoothed intensity so directives don't flip-flop frame to frame.
        self.intensity = seed.intensity
        # Rolling stress used for the safety rail.
        self._stress_ema = 0.0

    def _target_mood(self, affect: AffectState, safety: bool) -> str:
        if safety:
            return "relief"
        if affect.fear >= 0.66 or affect.stress >= 0.7:
            return "panic"
        if affect.fear >= 0.4 or affect.arousal >= 0.55:
            return "dread"
        return "unease"

    def step(self, affect: AffectState, bank: AssetBank | None) -> Directive:
        """Advance the director one tick and return the next directive."""
        # Update rolling stress (exponential moving average).
        self._stress_ema = 0.7 * self._stress_ema + 0.3 * affect.stress
        safety = self._stress_ema >= self.settings.stress_safety_ceiling

        # Tension control: if the player is under-engaged, push intensity up;
        # if over-stressed, pull it down.
        if safety:
            self.intensity = max(0.1, self.intensity - 0.2)
        elif affect.engagement < 0.4 and affect.fear < 0.4:
            self.intensity = min(1.0, self.intensity + 0.1)
        elif affect.fear > 0.75:
            self.intensity = max(0.2, self.intensity - 0.05)

        mood = self._target_mood(affect, safety)

        directive = Directive(
            intensity=round(self.intensity, 3),
            reason=(
                f"mood={mood} fear={affect.fear:.2f} stress={affect.stress:.2f} "
                f"engagement={affect.engagement:.2f}"
                + (" [SAFETY BACKOFF]" if safety else "")
            ),
            safety_backoff=safety,
        )

        if bank is not None:
            char = _pick(bank.characters, mood)
            ambient = _pick(bank.sounds, mood)
            level = _pick(bank.maps, mood)

            directive.ambient_sound_id = ambient.id if ambient else None
            directive.map_mutation = level.name if level else None

            # Only spawn a stalker / stinger once we've crossed into dread+.
            if mood in ("dread", "panic") and char is not None:
                directive.spawn_character_id = char.id
                stinger = _pick(bank.sounds, "panic")
                directive.stinger_sound_id = stinger.id if stinger else None

        return directive
