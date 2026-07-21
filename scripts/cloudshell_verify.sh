#!/usr/bin/env bash
# Keyless live Vertex verification for Google Cloud Shell.
#
# Cloud Shell is already authenticated as your user (Owner), so no service
# account key is needed — this works even with iam.disableServiceAccountKeyCreation
# enforced. Run it from the repo root inside Cloud Shell.
#
#   bash scripts/cloudshell_verify.sh
#
# Override defaults via env if needed:
#   GCP_PROJECT=eeg-horror GCS_BUCKET=dduhwycdgcdg bash scripts/cloudshell_verify.sh

set -euo pipefail

export ASSET_PROVIDER="vertex"
export GCP_PROJECT="${GCP_PROJECT:-eeg-horror}"
export GCP_LOCATION="${GCP_LOCATION:-us-central1}"
export GCS_BUCKET="${GCS_BUCKET:-dduhwycdgcdg}"
# Real audio via Lyria (set AUDIO_PROVIDER=none to get Gemini specs only).
export AUDIO_PROVIDER="${AUDIO_PROVIDER:-lyria}"

echo "==> Project: $GCP_PROJECT   Bucket: $GCS_BUCKET   Region: $GCP_LOCATION"
echo "==> Audio provider: $AUDIO_PROVIDER"

echo "==> Setting active project + Application Default Credentials"
gcloud config set project "$GCP_PROJECT" >/dev/null
# Cloud Shell usually already has ADC; this is a no-op if so. If it prompts,
# follow the browser link once.
gcloud auth application-default set-quota-project "$GCP_PROJECT" >/dev/null 2>&1 || true

echo "==> Enabling required APIs (idempotent)"
gcloud services enable aiplatform.googleapis.com storage.googleapis.com >/dev/null

echo "==> Installing Python deps in a venv"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements-vertex.txt

echo "==> Running live verification (Gemini + Imagen -> GCS)"
python scripts/verify_vertex.py
