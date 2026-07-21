"""Google Cloud Vertex AI provider (google-genai SDK).

Uses the current ``google-genai`` SDK (the ``vertexai.generative_models`` SDK is
deprecated / removed) to call Gemini for structured generation and Imagen for
concept art. When a GCS bucket is configured, generated images are uploaded and
the object's ``gs://`` URI is stamped onto the asset.

SDKs (``google-genai``, ``google-cloud-storage``) are imported lazily so the
package installs and the mock path runs even when they're absent.

Auth follows Application Default Credentials (ADC): a service-account key via
``GOOGLE_APPLICATION_CREDENTIALS``, or an authenticated environment such as
Google Cloud Shell / ``gcloud auth application-default login``.
"""

from __future__ import annotations

import json
import logging

from engine.generative.base import AssetProvider
from engine.schemas import Asset, AssetKind, SeedProfile

logger = logging.getLogger(__name__)


class VertexProvider(AssetProvider):
    """Vertex AI-backed generation via the google-genai SDK."""

    name = "vertex"

    def __init__(
        self,
        project: str | None,
        location: str,
        text_model: str,
        image_model: str,
        gcs_bucket: str | None = None,
    ) -> None:
        if not project:
            raise ValueError(
                "GCP_PROJECT must be set to use the Vertex asset provider."
            )
        self.project = project
        self.location = location
        self.text_model_name = text_model
        self.image_model_name = image_model
        self.gcs_bucket = gcs_bucket
        self._client = None  # lazy genai client
        self._storage_client = None  # lazy GCS client

    # --- lazy SDK initialisers ----------------------------------------- #
    def _genai(self):
        """Return a cached google-genai client bound to Vertex."""
        if self._client is None:
            from google import genai

            self._client = genai.Client(
                vertexai=True, project=self.project, location=self.location
            )
        return self._client

    def _bucket(self):
        if self._storage_client is None:
            from google.cloud import storage

            self._storage_client = storage.Client(project=self.project)
        return self._storage_client.bucket(self.gcs_bucket)

    # --- generation helpers -------------------------------------------- #
    async def _generate_json(self, instruction: str) -> dict:
        """Ask Gemini for a JSON object (forced via response_mime_type)."""
        from google.genai import types

        client = self._genai()
        resp = client.models.generate_content(
            model=self.text_model_name,
            contents=instruction,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=1.0,
            ),
        )
        text = (resp.text or "").strip()
        try:
            return json.loads(text)
        except ValueError:
            logger.warning("Vertex returned non-JSON; wrapping raw text.")
            return {"raw": text}

    def _generate_and_upload_image(self, prompt: str, object_name: str) -> str | None:
        """Generate one image with Imagen, upload to GCS, return its gs:// URI.

        Returns ``None`` (and logs) if no bucket is configured or generation
        fails, so a media hiccup never breaks the whole asset bank.
        """
        if not self.gcs_bucket:
            logger.info("No GCS bucket configured; skipping image for %s", object_name)
            return None
        try:
            from google.genai import types

            client = self._genai()
            resp = client.models.generate_images(
                model=self.image_model_name,
                prompt=prompt,
                config=types.GenerateImagesConfig(number_of_images=1),
            )
            if not resp.generated_images:
                logger.warning("Imagen returned no images for %s", object_name)
                return None
            image_bytes = resp.generated_images[0].image.image_bytes

            blob = self._bucket().blob(object_name)
            blob.upload_from_string(image_bytes, content_type="image/png")
            uri = f"gs://{self.gcs_bucket}/{object_name}"
            logger.info("Uploaded generated image -> %s", uri)
            return uri
        except Exception:  # noqa: BLE001 - media is best-effort
            logger.exception("Image generation/upload failed for %s", object_name)
            return None

    # --- provider API --------------------------------------------------- #
    async def generate_soundscape(self, seed: SeedProfile, mood: str) -> Asset:
        instruction = (
            f"Design an ambient horror soundscape spec for a {seed.theme}. "
            f"Target mood: {mood}. Player fears: {', '.join(seed.fears)}. "
            "Return JSON with keys: bpm (int), layers (list of strings), "
            "loopable (bool), description (string)."
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
            f"player's fear of {', '.join(seed.fears)}. Mood: {mood}. Return "
            "JSON with keys: name, description, behavior, aggression (0..1), "
            "image_prompt (a vivid concept-art prompt)."
        )
        payload = await self._generate_json(instruction)

        asset_id = f"chr-{mood}-{abs(hash((seed.theme, mood))) % 10**8}"
        image_prompt = payload.get("image_prompt") or (
            f"Concept art of a horror character in a {seed.theme}, mood {mood}."
        )
        uri = self._generate_and_upload_image(
            image_prompt, f"characters/{asset_id}.png"
        )

        return Asset(
            id=asset_id,
            kind=AssetKind.character,
            name=payload.get("name", f"{mood}-stalker"),
            prompt=instruction,
            uri=uri,
            payload=payload,
            tags=[mood, "enemy"],
        )

    async def generate_map(self, seed: SeedProfile, mood: str) -> Asset:
        instruction = (
            f"Design a level layout for a {seed.theme}, tuned for a {mood} mood. "
            "Return JSON with keys: rooms (int), lighting (string), "
            "hazards (list), graph (list of {from, to} room connections)."
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
