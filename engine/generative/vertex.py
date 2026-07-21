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

# Field names Vertex prediction responses have used for base64 audio bytes.
_AUDIO_KEYS = ("audioContent", "bytesBase64Encoded", "audio", "content")


def _extract_audio_b64(preds: list) -> str | None:
    """Pull base64 audio from a predictions list, tolerating field-name drift."""
    if not preds:
        return None
    pred = preds[0]
    if isinstance(pred, str):  # some models return the base64 string directly
        return pred
    if isinstance(pred, dict):
        for key in _AUDIO_KEYS:
            if pred.get(key):
                return pred[key]
    return None


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
        text_location: str = "global",
        music_model: str = "lyria-002",
        audio_provider: str = "none",
    ) -> None:
        if not project:
            raise ValueError(
                "GCP_PROJECT must be set to use the Vertex asset provider."
            )
        self.project = project
        # Regional endpoint (Imagen + Lyria + GCS).
        self.location = location
        # Gemini endpoint ("global" by default).
        self.text_location = text_location
        self.text_model_name = text_model
        self.image_model_name = image_model
        self.music_model_name = music_model
        self.audio_provider = audio_provider
        self.gcs_bucket = gcs_bucket
        self._text_client = None  # lazy genai client (global)
        self._image_client = None  # lazy genai client (regional)
        self._storage_client = None  # lazy GCS client

    # --- lazy SDK initialisers ----------------------------------------- #
    def _genai_text(self):
        """genai client for Gemini text, bound to the global endpoint."""
        if self._text_client is None:
            from google import genai

            self._text_client = genai.Client(
                vertexai=True, project=self.project, location=self.text_location
            )
        return self._text_client

    def _genai_image(self):
        """genai client for Imagen, bound to the regional endpoint."""
        if self._image_client is None:
            from google import genai

            self._image_client = genai.Client(
                vertexai=True, project=self.project, location=self.location
            )
        return self._image_client

    def _bucket(self):
        if self._storage_client is None:
            from google.cloud import storage

            self._storage_client = storage.Client(project=self.project)
        return self._storage_client.bucket(self.gcs_bucket)

    # --- generation helpers -------------------------------------------- #
    async def _generate_json(self, instruction: str) -> dict:
        """Ask Gemini for a JSON object (forced via response_mime_type)."""
        from google.genai import types

        client = self._genai_text()
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

    def _render_image_bytes(self, prompt: str) -> bytes | None:
        """Return PNG bytes for ``prompt`` using the configured image model.

        Two backends, selected by the model name:
        * ``imagen-*``      -> Imagen ``generate_images``.
        * ``gemini-*-image`` -> Gemini ``generate_content`` (inline image data).
        """
        from google.genai import types

        client = self._genai_image()
        model = self.image_model_name

        if "imagen" in model.lower():
            resp = client.models.generate_images(
                model=model,
                prompt=prompt,
                config=types.GenerateImagesConfig(number_of_images=1),
            )
            if not resp.generated_images:
                return None
            return resp.generated_images[0].image.image_bytes

        # Gemini image model: image comes back as inline_data on a part.
        resp = client.models.generate_content(model=model, contents=prompt)
        for cand in resp.candidates or []:
            content = getattr(cand, "content", None)
            for part in getattr(content, "parts", None) or []:
                inline = getattr(part, "inline_data", None)
                if inline and inline.data:
                    return inline.data
        return None

    def _generate_and_upload_image(self, prompt: str, object_name: str) -> str | None:
        """Generate one image, upload to GCS, return its gs:// URI.

        Returns ``None`` (and logs) if no bucket is configured or generation
        fails, so a media hiccup never breaks the whole asset bank.
        """
        if not self.gcs_bucket:
            logger.info("No GCS bucket configured; skipping image for %s", object_name)
            return None
        try:
            image_bytes = self._render_image_bytes(prompt)
            if not image_bytes:
                logger.warning("Image model returned no bytes for %s", object_name)
                return None

            blob = self._bucket().blob(object_name)
            blob.upload_from_string(image_bytes, content_type="image/png")
            uri = f"gs://{self.gcs_bucket}/{object_name}"
            logger.info("Uploaded generated image -> %s", uri)
            return uri
        except Exception:  # noqa: BLE001 - media is best-effort
            logger.exception("Image generation/upload failed for %s", object_name)
            return None

    def _generate_and_upload_music(
        self, prompt: str, negative_prompt: str, object_name: str
    ) -> str | None:
        """Generate a WAV soundscape with Lyria, upload to GCS, return its URI.

        Lyria is accessed via the regional ``:predict`` REST endpoint. Best-effort
        like image generation: any failure returns ``None`` and is logged, so a
        single audio hiccup never breaks the asset bank.
        """
        if self.audio_provider != "lyria" or not self.gcs_bucket:
            return None
        try:
            import base64

            import google.auth
            import google.auth.transport.requests
            import httpx  # bundled with google-genai

            creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            creds.refresh(google.auth.transport.requests.Request())

            loc = self.location  # Lyria is regional
            url = (
                f"https://{loc}-aiplatform.googleapis.com/v1/projects/"
                f"{self.project}/locations/{loc}/publishers/google/models/"
                f"{self.music_model_name}:predict"
            )
            body = {
                "instances": [
                    {"prompt": prompt, "negative_prompt": negative_prompt}
                ],
                "parameters": {"sample_count": 1},
            }
            resp = httpx.post(
                url,
                json=body,
                headers={"Authorization": f"Bearer {creds.token}"},
                timeout=180.0,
            )
            resp.raise_for_status()
            data = resp.json()
            preds = data.get("predictions", [])
            b64 = _extract_audio_b64(preds)
            if not b64:
                # Log what actually came back so the cause is visible: empty
                # predictions usually mean safety-filtering; a different key
                # means a schema mismatch.
                logger.warning(
                    "Lyria returned no audio for %s. status=%s top_keys=%s "
                    "pred0_keys=%s snippet=%s",
                    object_name,
                    resp.status_code,
                    list(data.keys()),
                    list(preds[0].keys()) if preds else "[]",
                    resp.text[:400],
                )
                return None
            audio_bytes = base64.b64decode(b64)

            blob = self._bucket().blob(object_name)
            blob.upload_from_string(audio_bytes, content_type="audio/wav")
            uri = f"gs://{self.gcs_bucket}/{object_name}"
            logger.info("Uploaded generated music -> %s", uri)
            return uri
        except Exception:  # noqa: BLE001 - media is best-effort
            logger.exception("Music generation/upload failed for %s", object_name)
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

        asset_id = f"snd-{mood}-{abs(hash((seed.theme, mood))) % 10**8}"
        # Turn the Gemini spec into a Lyria music prompt.
        music_prompt = (
            f"{payload.get('description', '')} Ambient horror score for a "
            f"{seed.theme}, {mood} mood, instrumental, atmospheric, loopable."
        ).strip()
        uri = self._generate_and_upload_music(
            music_prompt,
            negative_prompt="vocals, lyrics, spoken word, upbeat, cheerful",
            object_name=f"sounds/{asset_id}.wav",
        )

        return Asset(
            id=asset_id,
            kind=AssetKind.sound,
            name=f"{mood}-ambience",
            prompt=instruction,
            uri=uri,
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
