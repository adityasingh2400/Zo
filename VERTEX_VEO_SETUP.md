# Vertex AI Veo 3.1 Setup (~25 min, $300 free credit)

Goal: unblock Veo so we can render speaking-pose variants of each idle clip
without burning the daily Gemini API quota.

## 1. Create a GCP project (5 min)

1. Go to https://console.cloud.google.com/
2. Top-left dropdown → **New Project** → name it `empire-veo` (or similar)
3. Once created, note the **Project ID** (it'll look like `empire-veo-12345`)

## 2. Enable billing + free credit (5 min)

1. Top-left menu → **Billing**
2. **Link a billing account** to the project — use your normal credit card
3. If this is your first GCP account, you'll be prompted to claim the
   **$300 free credit** valid for 90 days. Take it.

## 3. Enable Vertex AI API (1 min)

1. Top-left menu → **APIs & Services → Enable APIs and Services**
2. Search "Vertex AI API" → **Enable**
3. Also enable "Cloud Storage API" while you're there

## 4. Create a service account + key (5 min)

1. Top-left menu → **IAM & Admin → Service Accounts**
2. **Create Service Account**:
   - Name: `empire-veo-runner`
   - Roles: `Vertex AI User`, `Storage Object User`
3. After creation, click into the service account → **Keys → Add Key →
   Create new key → JSON**. Download the JSON file.
4. Save it somewhere private, e.g. `/Users/aditya/.gcp/empire-veo.json`

## 5. Install gcloud CLI (one-time, 5 min)

```bash
# macOS via brew:
brew install --cask google-cloud-sdk

# Authenticate with the service account:
gcloud auth activate-service-account --key-file=/Users/aditya/.gcp/empire-veo.json
gcloud config set project YOUR_PROJECT_ID
```

## 6. Add to .env (1 min)

Append to `.env`:

```
GOOGLE_APPLICATION_CREDENTIALS=/Users/aditya/.gcp/empire-veo.json
GCP_PROJECT_ID=YOUR_PROJECT_ID
GCP_LOCATION=us-central1
```

## 7. Tell me when done

Once `.env` is updated, ping me. I'll switch the Veo client in
`phase0/scripts/veo_idle_library.py` from `genai.Client` to the
Vertex AI SDK so it uses the production model
(`veo-3.1-generate-001`, 50 RPM, no daily preview cap).
