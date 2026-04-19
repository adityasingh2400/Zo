# EMPIRE

**24/7 multilingual AI livestream-shopping seller, built at the Cactus × Google DeepMind × YC Gemma 4 Voice Agents hackathon.**

Voice in → product analyzed → avatar pitches → comments answered. Local-first routing on Gemma 4 + Cactus keeps ~90% of comments on-device for $144/mo unit economics vs. $9,628/mo for a live human team.

See [`EMPIRE-PITCH.md`](./EMPIRE-PITCH.md) for the thesis and [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the build. [`DESIGN_PRINCIPLES.md`](./DESIGN_PRINCIPLES.md) indexes the Lidwell PDF for agents reasoning about UI decisions.

## Run the demo

### Prerequisites

- macOS with Apple Silicon (for Cactus)
- Cactus + Gemma 4 E4B + whisper-base downloaded at `~/cactus/`
- RunPod pod with Wav2Lip server on port 8010 (see `scripts/provision_pod.sh`)
- `.env` with `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, `AWS_*`, `RUNPOD_*` — template in `.env.example`

### Boot

```bash
# 1. Open SSH tunnel to the Wav2Lip pod (keeps running)
bash phase0/scripts/open_tunnel.sh

# 2. Regenerate pre-rendered local answers (see below) — required on fresh clones

# 3. Backend
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000

# 4. Dashboard
cd dashboard && npm install && npm run dev
```

## Regenerating pre-rendered local answers

`backend/local_answers/*.mp4` is gitignored (~50 MB of generated MP4). A fresh clone has no pre-rendered clips, so every local-index question falls through to the cloud escalate path. To regenerate:

```bash
# 1. Confirm ELEVENLABS_API_KEY + ELEVENLABS_VOICE_ID are set in .env
# 2. Confirm the Wav2Lip tunnel is up: curl http://127.0.0.1:8010/health
# 3. Run the pre-render pipeline (10 clips, ~45 s warm):
python scripts/render_local_answers.py
# Use --force to re-render after changing ELEVENLABS_VOICE_ID.
```

## Smoke test

With backend + dashboard + tunnel running:

```bash
# Local path — expect router in ~0 ms, pre-rendered MP4 plays immediately
curl -F "text=is it real leather" http://127.0.0.1:8000/api/comment

# Escalate path — expect Bedrock + TTS + Wav2Lip, ~4.5 s warm
curl -F "text=how does this compare to the Apple Watch" http://127.0.0.1:8000/api/comment

# Router unit tests
pytest backend/tests/test_router.py -v
```
