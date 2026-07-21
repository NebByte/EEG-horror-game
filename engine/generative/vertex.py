"""Google Cloud Vertex AI provider.

Uses Gemini for text/structured generation (character + map + soundscape specs)
and Imagen for concept art. When a GCS bucket is configured, generated images are
uploaded and the object's ``gs://`` URI is stamped onto the asset.

The ``google-cloud-aiplatform`` / ``google-cloud-storage`` SDKs are imported
lazily so the package installs and the mock path runs even when they're absent.

Auth follows Application Default Credentials (ADC): set
``GOOGLE_APPLICATION_CREDENTIALS`` to a service-account key, or run in a GCP
environment / ``gcloud auth application-default login`` locally.
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
        self._text_model = None  # lazy init
        self._image_model = None  # lazy init
        self._storage_client = None  # lazy init

    # --- lazy SDK initialisers ----------------------------------------- #
    def _ensure_vertex_init(self) -> None:
        import vertexai

        vertexai.init(project=self.project, location=self.location)

    def _text_model_or_init(self):
        if self._text_model is None:
            from vertexai.generative_models import GenerativeModel

            self._ensure_vertex_init()
            self._text_model = GenerativeModel(self.text_model_name)
        return self._text_model

    def _image_model_or_init(self):
        if self._image_model is None:
            from vertexai.preview.vision_models import ImageGenerationModel

            self._ensure_vertex_init()
            self._image_model = ImageGenerationModel.from_pretrained(
                self.image_model_name
            )
        return self._image_model

    def _bucket(self):
        if self._storage_client is None:
            from google.cloud import storage

            self._storage_client = storage.Client(project=self.project)
        return self._storage_client.bucket(self.gcs_bucket)

    # --- generation helpers -------------------------------------------- #
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
            if text.startswith("```"):
                text = text.split("```")[1].removeprefix("json").strip()
            return json.loads(text)
        except (ValueError, IndexError):
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
            model = self._image_model_or_init()
            result = model.generate_images(prompt=prompt, number_of_images=1)
            if not result:
                logger.warning("Imagen returned no images for %s", object_name)
                return None
            image_bytes = result[0]._image_bytes  # SDK exposes raw PNG bytes

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
