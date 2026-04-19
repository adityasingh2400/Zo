# Build notes — Avatar Realism push (Apr 18-19)

What landed across phases 1-4 for the YC submission.

## What changed (perceptually)

| gap | before | after |
|---|---|---|
| Mic press → "she heard me" | blank Tier 0 | `idle_reading_comments` listening pose ≤50ms after pointer-down |
| Comment → first audio | 5-7s (TTS+Wav2Lip serial) | <1s (audio fires the moment TTS finishes; Wav2Lip lands later under continuing audio) |
| "Sell this" → pitch starts | 8-15s (LatentSync render) | ≤1s (cached audio + cached video + karaoke + chip) |
| Idle ↔ speaking transition | random rotation | event-driven sip-after-long-response + voice_state=thinking → idle_thinking |

## What's new on disk

- `backend/static/silent_unlock.mp3` — 100ms silence the dashboard plays inside the StartDemoOverlay click handler to bank browser autoplay permission for all subsequent `<audio>` elements.
- `backend/pitch_assets/` — cached pitch MP3 + word timings + manifest (gitignored). Re-render with `python scripts/render_pitch_assets.py`.
- `backend/response_audio/` — runtime per-comment TTS dispatches (gitignored, regenerated on every comment).
- `dashboard/src/components/StartDemoOverlay.jsx` — one-tap autoplay-unlock ceremony.
- `dashboard/src/components/KaraokeCaptions.jsx` — word-by-word reveal driven by `<audio>.currentTime`.
- `dashboard/src/components/TranslationChip.jsx` — top-right "🔴 LIVE · auto-captioned" pill.
- `scripts/render_pitch_assets.py` — admin script for caching pitch audio + timings per product.
- `scripts/smoke_audio_first.py` — end-to-end WS smoke (3 flows: audio-first, pitch, mic-press).

## Env flags (all default ON)

```
USE_AUDIO_FIRST=1            # comment_response_audio + background Wav2Lip
USE_KARAOKE=1                # word-by-word captions on the stage
USE_PITCH_VEO=1              # 30s pre-rendered pitch + TTS overlay
USE_BACKCHANNEL=1            # listening_attentive on mic press
USE_SPECULATIVE_BRIDGE=1     # short ack clip after transcript, before router
LIPSYNC_PROVIDER=wav2lip     # forward-compat hook
```

Set any to `0` in `.env` to flip back to prior behaviour without touching code.

## How to verify (60-second smoke)

```bash
# 1. backend
cd backend && ./venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000

# 2. (re)cache the wallet pitch assets if you're running clean
python scripts/render_pitch_assets.py leather_wallet

# 3. WS smoke — exercises all 3 hot paths
backend/venv/bin/python scripts/smoke_audio_first.py
# expected: smoke: 3/3 passed

# 4. browser smoke
cd dashboard && npm run dev
# open http://localhost:5173/ → click "Start Demo" → click VoiceMic
# (or type "sell this for $40" into the chat to fire the pitch path
# without needing a live mic)
```

## Architecture cheat-sheet

```
mic press (VoiceMic.jsx)
    └─ ws.send({type: "mic_pressed"})
       └─ Director.play_listening_attentive()  → tier1 muted loop
       └─ voice_state="transcribing"

mic release → POST /api/voice_comment
    └─ transcribe (whisper-base on Cactus, ~200ms)
    └─ broadcast voice_transcript
    └─ _fire_speculative_bridge() if not a likely local match  ← USE_SPECULATIVE_BRIDGE
    └─ run_routed_comment(text)
       ├─ router._detect_pitch_command()
       │   └─ "sell this..." → tool=pitch_product
       │       └─ Director.play_pitch_veo(slug)
       │           ├─ tier1 emit: pitch_veo (looped, muted)
       │           ├─ broadcast pitch_audio (url + word_timings)
       │           └─ schedule pitch_audio_end + fade_to_idle @ audio_ms+500
       │       └─ dashboard: <audio>.play(), KaraokeCaptions tracks, chip shows
       └─ otherwise → existing 4-tool router
           └─ escalate_to_cloud → api_respond_to_comment
               ├─ TTS → save → broadcast comment_response_audio   ← USE_AUDIO_FIRST
               │   └─ dashboard: <audio>.play(), KaraokeCaptions tracks
               └─ asyncio.create_task(_render_and_broadcast_video)
                   └─ Wav2Lip → save → broadcast comment_response_video
                       (audio_already_playing=true, expected_duration_ms=...)
                       └─ dashboard tier1 muted, duration handshake on canplaythrough
                          (rejects video if duration drifts >150ms — let audio finish alone)

broadcast_to_dashboards(msg)
    └─ Director.observe(msg)  ← motivated idle rotation
       ├─ comment_response_audio dur≥3s → schedule misc_sip_drink after audio
       ├─ pitch_audio_end → schedule misc_sip_drink in 2s
       └─ voice_state=thinking >2s → tier0 swap to idle_thinking
```

## Known limitations

- Wav2Lip pod must be reachable at `RUNPOD_POD_IP`. When it's offline, the audio-first audio still plays (good degradation), but `comment_response_video_failed` fires instead of a video crossfade. Audience hears the answer but sees Tier 0 idle the whole time.
- Word timings are synthetic (whitespace split + audio duration ÷ char count), accuracy ~80-95%. Cartesia would give us native per-word timings for tighter sync. Re-add via `TTS_PROVIDER=cartesia` later.
- Veo pitch video is the existing 8s `state_pitching_pose_speaking_1080p.mp4` looped 4× under the 30s audio. Karaoke captions divert eye gaze from the loop — judge-tested-on-paper, not yet judge-tested-on-stage.
- Listening-attentive backchannel uses `idle_reading_comments` as the substrate. If a dedicated `idle_listening_attentive.mp4` ever ships, swap the constant in `agents/avatar_director.py:_LISTENING_ATTENTIVE_URL`.

## Tags shipped

- `v0.1-tts-and-bridge-green` — TTS word timings + speculative bridge + env flags
- `v0.2-audio-first-green` — audio-first dispatch + StartDemoOverlay + duration handshake
- `v0.3-karaoke-green` — KaraokeCaptions + TranslationChip components
- `v0.4-pitch-veo-green` — pitch_product router + Director.play_pitch_veo + listening_attentive
- `v0.5-A-locked` — Approach A complete (motivated idle observer)
- `v0.7-FREEZE` — code-complete, no further features (set when freezing)
