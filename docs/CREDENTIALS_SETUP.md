# Google Cloud Credentials — Step-by-Step Setup

This guide walks you through obtaining every value the Vertex AI asset provider
needs. All steps are in the Google Cloud Console
(<https://console.cloud.google.com>) unless noted. When you're done you'll have a
filled-in `.env` and a service-account key on the machine.

> **Never commit secrets.** Put the key file under `secrets/` (gitignored) and
> keep real values in `.env` (also gitignored). Only the *placeholders* in
> `.env.example` are committed.

---

## 1. `GCP_PROJECT` — project ID
1. Open <https://console.cloud.google.com>.
2. Click the **project dropdown** in the top bar (left of the search box).
3. Pick an existing project, or **New Project** → name it (e.g. `eeg-horror`) →
   **Create**.
4. Reopen the dropdown — your project's **ID** (e.g. `eeg-horror-472013`) is the
   value. ⚠️ Use the **ID**, not the display name.

## 2. `GCP_LOCATION` — region
Use **`us-central1`** (best model availability). Only change for a data-residency
requirement. No console step.

## 3. Enable required APIs
Select your project first, then in the top **search bar**:
1. Search **"Vertex AI API"** → open → **Enable**.
2. Search **"Cloud Storage API"** → open → **Enable**.

## 4. `GCS_BUCKET` — bucket for generated media
1. Search **"Cloud Storage"** → **Buckets** → **Create**.
2. **Name**: globally unique, lowercase, e.g. `eeg-horror-assets-<suffix>`.
3. **Region**: `us-central1` (match your location).
4. Accept defaults → **Create**. The name you chose is `GCS_BUCKET`.

## 5. Service account, roles, and key
1. Search **"IAM & Admin"** → **Service Accounts** → **Create service account**.
2. **Name**: `horror-engine` → **Create and continue**.
3. **Grant roles** (Add another role for each):
   - **Vertex AI User** (`roles/aiplatform.user`)
   - **Storage Object Admin** (`roles/storage.objectAdmin`)

   → **Continue** → **Done**.
4. Click the new `horror-engine@…` account → **Keys** → **Add key** →
   **Create new key** → **JSON** → **Create**. A `.json` file downloads.
5. Place it on the machine and note the path:
   ```bash
   mkdir -p /home/user/EEG-horror-game/secrets
   mv ~/Downloads/eeg-horror-*.json /home/user/EEG-horror-game/secrets/sa.json
   ```
   That path is `GOOGLE_APPLICATION_CREDENTIALS`.

   *Alternative (no key file):* run
   `gcloud auth application-default login` and leave
   `GOOGLE_APPLICATION_CREDENTIALS` empty — ADC is used automatically.

## 6. Model values (defaults — leave as-is)
- `VERTEX_TEXT_MODEL=gemini-2.0-flash`
- `VERTEX_IMAGE_MODEL=imagen-3.0-generate-001`

## 7. Audio (decision needed)
Vertex AI has no music/SFX model. Choose:
- **(a)** Keep Gemini-generated sound *specs* (bpm/layers/description) for now.
  No extra credentials.
- **(b)** Add real audio via a third-party (e.g. ElevenLabs). You'd create an
  account and supply that API key as `AUDIO_API_KEY`.

## 8. Write your `.env`
Copy `.env.example` to `.env` and fill in:
```
ASSET_PROVIDER=vertex
GCP_PROJECT=<your-project-id>
GCP_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/home/user/EEG-horror-game/secrets/sa.json
GCS_BUCKET=<your-bucket-name>
```

## 9. Verify (optional quick check)
```bash
pip install -r requirements-vertex.txt
python -c "from google.cloud import aiplatform; aiplatform.init(); print('auth OK')"
```

---

Once `.env` is filled and you've picked audio **(a)** or **(b)**, the engine's
`VertexProvider` will be finished to call Imagen, upload media to your bucket,
and stamp each `Asset.uri` with its `gs://` path.
