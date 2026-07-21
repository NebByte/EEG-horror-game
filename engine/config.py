"""Runtime configuration.

Settings are read from environment variables (or a local ``.env`` file) so the
prototype can run in three modes without code changes:

* ``ASSET_PROVIDER=mock``   -> fully local, no cloud creds needed (default).
* ``ASSET_PROVIDER=vertex`` -> Google Cloud Vertex AI (Gemini / Imagen).

See ``.env.example`` for the full list.
"""

from __future__ import annotations

from functools import lru_cache

try:
    # pydantic-settings is the v2 home for BaseSettings.
    from pydantic_settings import BaseSettings, SettingsConfigDict

    _HAS_SETTINGS = True
except Exception:  # pragma: no cover - fallback when dependency is missing
    _HAS_SETTINGS = False


if _HAS_SETTINGS:

    class Settings(BaseSettings):
        """Engine settings, populated from the environment."""

        model_config = SettingsConfigDict(
            env_file=".env", env_file_encoding="utf-8", extra="ignore"
        )

        # --- API ---
        app_name: str = "eeg-horror-engine"
        environment: str = "development"
        api_prefix: str = "/v1"

        # --- Generative asset provider ---
        # "mock" (default, offline) or "vertex" (Google Cloud).
        asset_provider: str = "mock"

        # --- Google Cloud / Vertex AI ---
        gcp_project: str | None = None
        # Regional endpoint used for Imagen + GCS (Imagen is region-bound).
        gcp_location: str = "us-central1"
        # Gemini text models are served from the "global" endpoint on Vertex;
        # a regional endpoint returns NOT_FOUND for them on many projects.
        vertex_text_location: str = "global"
        # Use fully-versioned GA model IDs; bare aliases (e.g. "gemini-2.0-flash")
        # are not resolvable through the aiplatform SDK on all projects.
        vertex_text_model: str = "gemini-2.0-flash-001"
        vertex_image_model: str = "imagen-3.0-generate-002"
        # Lyria music model (regional, us-central1) used for real audio when
        # audio_provider="lyria".
        vertex_music_model: str = "lyria-002"
        # Audio backend: "none" (Gemini sound specs only, default) or "lyria"
        # (generate real WAV soundscapes and upload to GCS).
        audio_provider: str = "none"
        # Path to a service-account JSON key (optional; ADC is used otherwise).
        google_application_credentials: str | None = None
        # GCS bucket where generated media (images/audio) is uploaded. When
        # empty, the Vertex provider skips media upload and returns specs only.
        gcs_bucket: str | None = None

        # --- EEG signal processing ---
        eeg_sample_rate_hz: float = 256.0
        eeg_window_seconds: float = 2.0
        # Channel labels expected from the headset, in order.
        eeg_channels: list[str] = ["AF3", "AF4", "F3", "F4", "T7", "T8", "O1", "O2"]

        # --- Experience safety rails ---
        # Above this sustained stress we back off to protect the player.
        stress_safety_ceiling: float = 0.9

else:  # pragma: no cover - minimal shim used only when deps are absent

    import os

    class Settings:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.app_name = os.getenv("APP_NAME", "eeg-horror-engine")
            self.environment = os.getenv("ENVIRONMENT", "development")
            self.api_prefix = os.getenv("API_PREFIX", "/v1")
            self.asset_provider = os.getenv("ASSET_PROVIDER", "mock")
            self.gcp_project = os.getenv("GCP_PROJECT")
            self.gcp_location = os.getenv("GCP_LOCATION", "us-central1")
            self.vertex_text_location = os.getenv("VERTEX_TEXT_LOCATION", "global")
            self.vertex_text_model = os.getenv(
                "VERTEX_TEXT_MODEL", "gemini-2.0-flash-001"
            )
            self.vertex_image_model = os.getenv(
                "VERTEX_IMAGE_MODEL", "imagen-3.0-generate-002"
            )
            self.vertex_music_model = os.getenv("VERTEX_MUSIC_MODEL", "lyria-002")
            self.audio_provider = os.getenv("AUDIO_PROVIDER", "none")
            self.google_application_credentials = os.getenv(
                "GOOGLE_APPLICATION_CREDENTIALS"
            )
            self.gcs_bucket = os.getenv("GCS_BUCKET")
            self.eeg_sample_rate_hz = float(os.getenv("EEG_SAMPLE_RATE_HZ", "256"))
            self.eeg_window_seconds = float(os.getenv("EEG_WINDOW_SECONDS", "2"))
            self.eeg_channels = ["AF3", "AF4", "F3", "F4", "T7", "T8", "O1", "O2"]
            self.stress_safety_ceiling = float(
                os.getenv("STRESS_SAFETY_CEILING", "0.9")
            )


@lru_cache
def get_settings() -> "Settings":
    """Return a cached ``Settings`` instance."""
    return Settings()
