"""Google Cloud Vertex AI provider.

Uses Gemini for text/structured generation (character + map design) and Imagen
for concept art. The ``google-cloud-aiplatform`` SDK is imported lazily so the
package installs and the mock path runs even when the SDK is absent.

Auth follows Application Default Credentials (ADC): set
``GOOGLE_APPLICATION_CREDENTIALS`` to a service-account key, or run in a GCP
environment / ``gcloud auth application-default login`` locally.

NOTE: This is a reference implementation for the prototype. Image bytes are
written to GCS in a real deployment; here we return the model's structured
output and leave media upload as a documented TODO (see ROADMAP).
"""

from __future__ import annotations

import json
import logging

from engine.generative.base import AssetProvider
from engine.schemas import Asset, AssetKind, SeedProfile

logger = logging.getLogger(__name__)


class VertexProvider(AssetProvider):
    """Vertex AI-backed generation."""

    name = "vertex"

    def __init__(
        self,
        project: str | None,
        location: str,
        text_model: str,
        image_model: str,
    ) -> None:
        if not project:
            raise ValueError(
                "GCP_PROJECT must be set to use the Vertex asset provider."
            )
        self.project = project
        self.location = location
        self.text_model_name = text_model
        self.image_model_name = image_model
        self._text_model = None  # lazy init

    def _text_model_or_init(self):
        if self._text_model is None:
            # Imported lazily so the dependency is optional.
            import vertexai
            from vertexai.generative_models import GenerativeModel

            vertexai.init(project=self.project, location=self.location)
            self._text_model = GenerativeModel(self.text_model_name)
        return self._text_model

    async def _generate_json(self, instruction: str) -> dict:
        """Ask Gemini for a JSON object and parse it defensively."""
        model = self._text_model_or_init()
        prompt = (
            instruction
            + "\n\nRespond with a single minified JSON object and nothing else."
        )
        # The SDK call is synchronous; for the prototype we accept that. A
        # production build would offload to a threadpool or use the async client.
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        try:
            # Strip markdown fences if the model added them.
            if text.startswith("```"):
                text = text.split("```")[1].removeprefix("json").strip()
            return json.loads(text)
        except (ValueError, IndexError):
            logger.warning("Vertex returned non-JSON; wrapping raw text.")
            return {"raw": text}

    async def generate_soundscape(self, seed: SeedProfile, mood: str) -> Asset:
        instruction = (
            f"Design an ambient horror soundscape spec for a {seed.theme}. "
            f"Target mood: {mood}. Player fears: {', '.join(seed.fears)}. "
            "Include keys: bpm (int), layers (list of strings), loopable (bool), "
            "description (string)."
        )
        payload = await self._generate_json(instruction)
        return Asset(
            id=f"snd-{mood}-{abs(hash((seed.theme, mood))) % 10**8}",
            kind=AssetKind.sound,
            name=f"{mood}-ambience",
            prompt=instruction,
            payload=payload,
            tags=[mood, "ambient"],
        )

    async def generate_character(self, seed: SeedProfile, mood: str) -> Asset:
        instruction = (
            f"Design a horror character for a {seed.theme} that embodies the "
            f"player's fear of {', '.join(seed.fears)}. Mood: {mood}. Include "
            "keys: name, description, behavior, aggression (0..1), "
            "image_prompt (a vivid concept-art prompt)."
        )
        payload = await self._generate_json(instruction)
        # A production build calls Imagen with payload['image_prompt'] and
        # uploads the PNG to GCS, setting `uri` to the gs:// path. TODO.
        return Asset(
            id=f"chr-{mood}-{abs(hash((seed.theme, mood))) % 10**8}",
            kind=AssetKind.character,
            name=payload.get("name", f"{mood}-stalker"),
            prompt=instruction,
            payload=payload,
            tags=[mood, "enemy"],
        )

    async def generate_map(self, seed: SeedProfile, mood: str) -> Asset:
        instruction = (
            f"Design a level layout for a {seed.theme}, tuned for a {mood} mood. "
            "Include keys: rooms (int), lighting (string), hazards (list), "
            "graph (list of {from, to} room connections)."
        )
        payload = await self._generate_json(instruction)
        return Asset(
            id=f"map-{mood}-{abs(hash((seed.theme, mood))) % 10**8}",
            kind=AssetKind.map,
            name=f"{mood}-wing",
            prompt=instruction,
            payload=payload,
            tags=[mood, "level"],
        )
