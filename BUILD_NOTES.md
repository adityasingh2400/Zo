# Build notes — Avatar Realism push (Apr 18-19)

What landed across phases 1-4 + the post-submission tightenings (v0.5-A-locked → b56324a).

## What changed (perceptually)

| gap | before | after |
|---|---|---|
| Mic press → "she heard me" | blank Tier 0 | `idle_reading_comments` listening pose ≤50ms after pointer-down |
| Comment → first audio | 5-7s (TTS+Wav2Lip serial) | reading_chat fires <50ms (Cody's 2c98beb), audio + lip-sync land together at ~7-15s under continuing reading visual |
| Phone-uploaded video → pitch starts | 8-15s (LatentSync render) | <1s after script ready (audio-first dispatch + looping speaking-pose video + karaoke captions) |
| Karaoke word boundaries | synthetic char-proportional | real ElevenLabs `/with-timestamps` per-character alignment, aggregated to words |
| First-frame pop on Tier 1 swaps | small head-jump on slo-mo | `prepareFirstFrame` seek(0)+await 'seeked' before opacity ramp (REVISIONS §17 closed) |
| Idle ↔ speaking transition | random rotation | event-driven sip-after-long-response + voice_state=thinking → idle_thinking via `Director.observe()` |

## What's new on disk

- `backend/static/silent_unlock.mp3` — 100ms silence the dashboard plays inside the StartDemoOverlay click handler to bank browser autoplay permission for all subsequent `<audio>` elements.
- `backend/response_audio/` — runtime per-comment + per-pitch TTS dispatches (gitignored, regenerated on every dispatch). Pitches use a `pitch_dispatch_<uuid>.mp3` filename prefix; comment responses use `resp_<uuid>.mp3`.
- `backend/agents/trace.py` — contextvar-based trace ids that propagate across `asyncio.create_task` so a comment's full lifecycle is greppable by one short id.
- `dashboard/src/components/StartDemoOverlay.jsx` — one-tap autoplay-unlock ceremony.
- `dashboard/src/components/KaraokeCaptions.jsx` — word-by-word reveal driven by `<audio>.currentTime`.
- `dashboard/src/components/TranslationChip.jsx` — top-right "🔴 LIVE · auto-captioned" pill (live + pitch variants).
- `dashboard/src/lib/dlog.js` — fire-and-forget POSTer mirroring browser audio/video lifecycle into backend.log under the same trace ids.
- `scripts/smoke_audio_first.py` — comment + mic smoke (pitch is exercised by smoke_phone_video.py, since pitches only fire from video upload).
- `scripts/smoke_phone_video.py` — uploads a real product video over `/ws/phone` and watches for the audio-first pitch dispatch on the dashboard WS.

## What was removed (intentionally)

- `pitch_product` router tool + `_detect_pitch_command` + `_PITCH_TRIGGERS` chat-trigger pattern. Pitches fire from the video-upload pipeline only — comments are audience reactions, never pitches.
- `Director.play_pitch_veo()` + manifest loader + `backend/pitch_assets/` cache + `scripts/render_pitch_assets.py`. The slug-based cached pitch path existed only to feed the chat-trigger flow; gone with it. `Director.dispatch_audio_first_pitch()` is the only remaining pitch entry point and takes its inputs directly from the live video pipeline.

## Env flags (all default ON)

```
USE_AUDIO_FIRST=1            # comment_response_audio + standalone <audio> dispatch (powers KaraokeCaptions)
USE_KARAOKE=1                # word-by-word captions on the stage
USE_PITCH_VEO=1              # video-upload triggers audio-first pitch dispatch (vs legacy Wav2Lip pitch)
USE_BACKCHANNEL=1            # listening_attentive on mic press
USE_SPECULATIVE_BRIDGE=1     # short ack clip after transcript, before router
LIPSYNC_PROVIDER=wav2lip     # forward-compat hook
```

Set any to `0` in `.env` to flip back to prior behaviour without touching code.

## How to verify (60-second smoke)

```bash
# 1. backend (ws-max-size needed so phone uploads up to ~50MB don't bounce off the default 16MB limit)
cd backend && ./venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000 \
  --no-access-log --ws-max-size 67108864

# 2. comment + mic smoke
backend/venv/bin/python scripts/smoke_audio_first.py
# expected: smoke: 2/2 passed

# 3. video pitch smoke (uploads test_fixtures/watch_demo.mov via /ws/phone)
backend/venv/bin/python scripts/smoke_phone_video.py
# expected: phone video → pitch_audio fired

# 4. browser
cd dashboard && npm run dev
# open http://localhost:5173/ → click "Start Demo" → type a chat comment
# (e.g. "is it real leather", "how does it ship") to see the audio-first
# response with lip-synced video + karaoke captions
```

## Architecture cheat-sheet

```
PHONE VIDEO PATH (production pitch entry)
─────────────────────────────────────────
phone records video → /ws/phone {type:"sell_video", video_b64}
    └─ _handle_phone_sell_video → run_video_sell_pipeline(temp_path)
       ├─ process_video → Deepgram transcript + best frames
       ├─ analyze_and_script_claude → product_data + sales_script
       └─ run_sell_pipeline(best_frame, voice_text + transcript)
           ├─ broadcast product_data + sales_script + product_photo
           └─ _run_audio_first_pitch(script)             ← USE_PITCH_VEO=1
               ├─ text_to_speech(return_word_timings=True) ← real ElevenLabs alignment
               ├─ save audio → /response_audio/pitch_dispatch_<uuid>.mp3
               └─ Director.dispatch_audio_first_pitch
                   ├─ tier1 emit: looping speaking-pose video, MUTED
                   └─ broadcast pitch_audio (audio_url + word_timings + ms)
                       └─ dashboard: <audio>.play(), KaraokeCaptions tracks,
                          TranslationChip mounts (pitch variant)

MIC PATH (audience question via voice)
──────────────────────────────────────
VoiceMic.jsx pointer-down
    └─ ws.send({type:"mic_pressed"})
       └─ Director.play_listening_attentive() → tier1 muted loop (<50ms)
       └─ voice_state="transcribing"

mic release → POST /api/voice_comment
    ├─ transcribe (Deepgram or whisper-base on Cactus)
    ├─ broadcast voice_transcript
    ├─ _fire_speculative_bridge() if not a likely local match     ← USE_SPECULATIVE_BRIDGE
    └─ run_routed_comment(text)  ← same as typed comment, see below

COMMENT PATH (audience reaction — typed OR voice)
─────────────────────────────────────────────────
comment received
    └─ run_routed_comment
       ├─ trace.new_trace + reading_chat_emitted_immediate (Cody 2c98beb: <50ms)
       ├─ classify_comment_gemma + router.decide
       └─ dispatch:
           respond_locally     pre-rendered .mp4 (sub-300ms)
           play_canned_clip    bridge clip (sub-300ms)
           block_comment       no visual, counter only
           escalate_to_cloud → api_respond_to_comment
               ├─ TTS + Wav2Lip render in parallel under reading_chat
               ├─ release reading_chat (1.5s minimum visible)
               ├─ broadcast comment_response_audio (powers karaoke)
               └─ Director.play_response(url, muted=True, expected_duration_ms)
                   └─ tier1 muted lip-sync video crossfades over standalone audio
                      ⇒ duration handshake at 250ms threshold (drift typically 30-80ms)

OBSERVER (motivated idle rotation)
──────────────────────────────────
broadcast_to_dashboards(msg) → Director.observe(msg)
    ├─ comment_response_audio dur≥3s → schedule misc_sip_drink after audio
    ├─ pitch_audio_end → schedule misc_sip_drink in 2s
    └─ voice_state=thinking >2s → tier0 swap to idle_thinking
```

## Known limitations

- Wav2Lip pod must be reachable at `RUNPOD_POD_IP`. When offline, audio-only fallback fires (`audio_only_fallback_dispatched`) — audience hears the answer but sees Tier 0 idle the whole time.
- Veo pitch video is the existing 8s `state_pitching_pose_speaking_1080p.mp4` looped under whatever audio length Claude+ElevenLabs produce. Karaoke captions divert eye gaze from the loop seam.
- Listening-attentive backchannel uses `idle_reading_comments` as the substrate. If a dedicated `idle_listening_attentive.mp4` ever ships, swap the constant in `agents/avatar_director.py:_LISTENING_ATTENTIVE_URL`.
- First-frame match (`prepareFirstFrame`) assumes the browser's `seeked` event accurately marks first-frame readiness. Chrome and Firefox honour this; Safari occasionally fires `seeked` slightly before the frame is composited (visible only on slo-mo).

## Tags shipped

- `v0.1-tts-and-bridge-green` — TTS word timings + speculative bridge + env flags
- `v0.2-audio-first-green` — audio-first dispatch + StartDemoOverlay + duration handshake
- `v0.3-karaoke-green` — KaraokeCaptions + TranslationChip components
- `v0.4-pitch-veo-green` — Director.dispatch_audio_first_pitch + listening_attentive
- `v0.5-A-locked` — Approach A complete (motivated idle observer)
- `v0.6-dry-run-1-passed` — smoke 3/3 PASS pre-Cody-restructure
- `v0.7-FREEZE` — first code-complete tag
- `v0.8-video-pipeline-green` — phone video upload pipeline wired into audio-first pitch
- `v0.9-transitions-tightened` — first-frame match (REVISIONS §17 closed) + real ElevenLabs character timings + smooth fade-out
- `2c98beb` (Cody) — instant reading_chat + lip-synced response with karaoke preserved
- `b56324a` — duration handshake threshold raised to 250ms post-2c98beb + smoke early-exit fix
