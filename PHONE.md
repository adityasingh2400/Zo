# PHONE — integration contract

How the phone client uploads a recorded product video and what the backend
does with it. Two transports, both call the same pipeline under the hood.

## TL;DR

- Connect: `ws://<backend-host>:8000/ws/phone`
- Send: `{type: "sell_video", video_b64: "<base64 mp4/mov>"}`
- The pitch fires on the dashboard within ~1s of the script being ready
  (audio-first via `Director.dispatch_audio_first_pitch`).

> **Backend boot reminder:** start uvicorn with `--ws-max-size 67108864`
> (64MB) so a 15-30MB video b64 frame doesn't bounce off the default
> 16MB limit. The repo's `scripts/demo_prewarm.sh` reminder string
> includes the flag; if you start uvicorn manually, copy it from there.

## Transport options

### Option A — WebSocket (recommended for the phone)

Connect once, send a message per recording. The connection stays open
across recordings so you don't pay handshake cost on every upload.

```
ws://<backend-host>:8000/ws/phone
```

#### Send

```json
{
  "type": "sell_video",
  "video_b64": "<base64 of a .mp4 or .mov file, max ~10MB>",
  "filename": "clip.mp4",          // optional; only used to pick a temp suffix
  "voice_text": "sell this",       // optional; appended as seller intent context
  "mime": "video/mp4"              // optional; informational only today
}
```

#### Receive (acks)

```json
{"type": "phone_ack", "stage": "received",          "bytes": 1572864, "session_id": "phone_a1b2c3d4e5"}
{"type": "phone_ack", "stage": "pipeline_started", "session_id": "phone_a1b2c3d4e5"}
```

If the upload is malformed:

```json
{"type": "phone_ack", "stage": "error", "reason": "missing_video_b64"}
{"type": "phone_ack", "stage": "error", "reason": "b64_decode_failed: ..."}
{"type": "phone_ack", "stage": "error", "reason": "empty_after_decode"}
```

### Option B — HTTP multipart

If WebSocket is awkward, hit the existing endpoint:

```
POST http://<backend-host>:8000/api/sell-video
Content-Type: multipart/form-data
file=<the .mp4 or .mov binary>
voice_text=<optional, default "sell this">
```

Returns `{"status": "video_pipeline_started", "bytes": <int>}`. No
session id; the dashboard sees the same WS events as Option A.

### Phone side recommendations

- Record at 720p / 30fps, 5-15s — long enough for Claude vision to read
  the product, short enough for fast Deepgram transcription.
- Encode to H.264 + AAC inside an `.mp4` container. Anything ffmpeg can
  decode works, but `.mp4` is the most predictable.
- Auto-upload immediately on stop — no preview screen — to minimise
  perceived latency.
- Reuse the same WebSocket connection across recordings.

## What happens after upload

```
phone → /ws/phone {type: sell_video, video_b64}
  └─ _handle_phone_sell_video
     ├─ phone_ack "received" + phone_video_received WS to dashboards
     ├─ write video → temp file
     ├─ run_video_sell_pipeline(temp_path, voice_text)
     │   ├─ process_video (parallel)
     │   │   ├─ ffmpeg → audio.wav
     │   │   ├─ ffmpeg → key frames (15)
     │   │   ├─ Deepgram nova-3 → transcript        (broadcast voice_transcript)
     │   │   ├─ OpenCV Laplacian → 4 sharp frames
     │   │   └─ b64-encode best frames
     │   ├─ run_carousel_pipeline (parallel) → 3D spin
     │   ├─ transcript_extract (Cactus on-device, raced 1.5s)
     │   └─ run_sell_pipeline(best_frames_b64[0], voice_text + " " + transcript)
     │       ├─ analyze_and_script_claude (vision + script in one call)
     │       │   └─ broadcast product_data + sales_script
     │       ├─ remove_background (parallel)
     │       │   └─ broadcast product_photo
     │       ├─ run_3d_generation (background)
     │       └─ _run_audio_first_pitch(script)             ← USE_PITCH_VEO=1
     │           ├─ text_to_speech(script, return_word_timings=True)
     │           ├─ save audio → /response_audio/pitch_dispatch_<uuid>.mp3
     │           ├─ Director.dispatch_audio_first_pitch
     │           │   ├─ Tier 1: looped pitching-pose video, MUTED
     │           │   └─ broadcast pitch_audio (audio_url + word_timings + ms)
     │           ├─ broadcast pitch_video (legacy event, backend=audio_first)
     │           └─ schedule pitch_audio_end + fade_to_idle @ audio_ms+500
     └─ phone_ack "pipeline_started"
```

## WS events the dashboard receives (phone-relevant subset)

These also flow to any phone client connected to `/ws/phone` if you want
to mirror state on the phone UI (informational; the phone is upload-only
in the current design):

| event | when | useful payload |
|---|---|---|
| `phone_video_received` | sell_video accepted | `bytes`, `session_id`, `filename` |
| `transcript` | Deepgram returns | `text` |
| `transcript_extract` | on-device extract finishes | `data` (selling points hint) |
| `product_data` | Claude returns analysis | `data` (name, price, materials, …) |
| `sales_script` | Claude generates pitch | `script` |
| `product_photo` | bg-removal completes | `photo` (base64) |
| `pitch_audio` | TTS done, dispatch fires | `url`, `word_timings`, `expected_duration_ms`, `script` |
| `pitch_video` | legacy umbrella event | `url` (audio_url when audio-first), `tts_ms`, `audio_ms`, `backend` |
| `pitch_audio_end` | audio finished playing | `slug` |

## Failure modes the phone should handle

- WS closes mid-upload → reconnect + resend (no resume support).
- 2-second silence after `phone_ack "pipeline_started"` is normal; the
  pipeline is up to ~10s end-to-end on the demo MacBook before
  `pitch_audio` lands.
- If the dashboard never shows the pitch playing, check the dashboard
  agent_log for `[INTAKE] FAILED` / `[pitch] render failed`. The phone
  has done its job either way.

## Smoke test

```bash
backend/venv/bin/python scripts/smoke_phone_video.py \
  --port 8000 \
  --video backend/test_fixtures/watch_demo.mov
# expected: phone_ack received → pipeline_started → pitch_audio fires
```
