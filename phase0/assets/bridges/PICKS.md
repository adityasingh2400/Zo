# Bridge Library — Picks + Architecture

The shortlist + the full architecture this enables. Updated 2026-04-19.

---

## Final picks (ship these to the Director)

| Intent | Picked variants | Where they're used |
|---|---|---|
| `question` | **A**, **E** | Wav2Lip substrate when classifier returns `type: question` (or unclassified — neutral collapsed in) |
| `objection` | **E** | Wav2Lip substrate when classifier returns `type: objection` |
| `compliment` | **B** (default), **E** (special) | Wav2Lip substrate when classifier returns `type: compliment`. See routing note below for E. |
| `intro` | **C** | `/api/go_live` (G hotkey on /stage) — demo opener |
| `pitch` | **`pitch_chained.mp4`** (3-segment Veo chain, 20.6s, neutral-frame stitched at 6.2s + 12.4s) | Wav2Lip substrate looped under the post-video-upload pitch audio. Replaces `state_pitching_pose_speaking_1080p.mp4` in `_DEFAULT_PITCH_VIDEO_URL`. |
| `welcome` | **`welcome.mp4`** (Veo-native render with baked audio — see below) | Tier 1 ambient + `/api/go_live` (G hotkey on /stage) — demo opener |

## Welcome architecture — Veo-native render, no Wav2Lip needed

**Decision (2026-04-19):** Skip the Wav2Lip composite entirely. Veo 3.1
generates the video AND the spoken audio in a single diffusion pass,
which means the lip-sync is INHERENT to the rendering — no mouth-region
inpainting, no edge-frame artifacts, no runtime pod inference cost.

Rendered file: `phase0/assets/bridges/welcome/welcome.mp4`
- **Video:** 2.21s, 1080×1920, h264, locked-off 9:16 medium shot
- **Audio:** 2.20s, AAC 192k, single line: *"Welcome to the stream, guys!"*
- **Audio profile:**
  - 0.0s – 1.5s: clean speech with two-handed wave gesture
  - 1.5s – 2.0s: natural acoustic decay tail (full final word lands)
  - 2.0s – 2.1s: 100ms fade-out window (kills any click)
  - 2.1s – 2.2s: pure silence (-inf dB) — body language settles into closing pose
- **Visual end frame:** hands at waist, soft smile, eye contact (matches Tier 0
  idle anchor pose so the crossfade-to-idle on completion is invisible)

Render pipeline + post-process recipe:
```
1. Veo 3.1 native render (Vertex AI, 6s @ 1080p 9:16, generate_audio=True)
   prompt = explicit two-phase structure (PHASE 1 speak + wave 0-2.5s,
            PHASE 2 silence + closed-mouth + return-to-pose 2.5s-6s).
   Veo produces native lip-synced audio in Phase 1 + ~silent Phase 2,
   though often with residual breath/babble at -36 dB or extra
   vocalizations beyond the spoken line.
2. ffmpeg -t 2.4 + afade=out:st=2.0:d=0.1 trims the tail and replaces
   any post-speech Veo babble with deterministic pure silence.
3. Final trim to 2.2s so audio and video end essentially together,
   leaving 100ms of silent body language for the idle crossfade.
4. We tried ElevenLabs voice-changer (speech-to-speech) on top of the
   Veo audio to align the timbre with our canonical bridge voice
   (yj30vwTGJxSHezdAGsv9). Rejected — added artifacts that the Veo
   native voice did not have. Veo's voice was already close enough.
```

Director wiring:
- `/api/go_live` plays `welcome.mp4` directly (no `pick_bridge_clip`
  fallback needed — the asset is canonical and always on disk in
  shipped builds). 200ms tail buffer on `_release_after()` lets the
  video fully end before the idle crossfade kicks in.
- Tier 1 ambient pool also includes `welcome.mp4` (15% weight) so it
  re-fires occasionally during idle to re-greet new viewers, but the
  G hotkey is the primary entry point.

**Total in shipping library: 6 picked clips** (after pitch + welcome reviewed).
Everything else stays on disk under `phase0/assets/bridges/<intent>/` as
backup variants we can swap to without re-rendering. The original silent
`welcome_A.mp4` and `welcome_B.mp4` substrates are kept as backup in
case we ever want to revert to the Wav2Lip composite path.

---

## Special routing rule — `compliment_E` for personal-preference questions

When an audience comment is a **personal question about the seller's
experience** (rather than the product itself), the response should:

1. Use `compliment_E` as the Wav2Lip substrate (the slightly different
   body language sells "personally I love this thing" energy).
2. Have the LLM generate a first-person preference response in the form
   *"personally I really like [unique aspect of the product]"*.

**Trigger pattern examples** (router needs to detect):
- "why did you buy this"
- "what do you really like about it"
- "what's your favorite thing about it"
- "do you actually use this"
- "is this something you'd buy yourself"

Mechanically this becomes a sub-classification of `compliment`:
`compliment_personal` vs default `compliment`. Implementation pattern
similar to the existing rule-based router in `backend/agents/router.py` —
keyword regex over the comment text, runs after `classify_comment_gemma`
returns `type: compliment`.

When wired:

```python
# in the router, after Gemma returns classify.type == "compliment"
if _is_personal_preference_question(comment):
    bridge_label = "compliment_personal"  # → compliment_E specifically
    response_prompt_style = "first_person_preference"
else:
    bridge_label = "compliment"           # → compliment_B (default)
    response_prompt_style = "default"
```

The substrate→variant map in `pick_bridge_clip` would resolve:
- `compliment` → compliment_B
- `compliment_personal` → compliment_E

---

## Tier architecture (decided 2026-04-19)

Three rendering layers stacked z-index 0/1/2. Each owns its own ping-pong
A/B `<video>` pair on the dashboard. Six video elements total.

### Tier 0 — always-on idle floor (looping, muted)

Director rotates between these every 8–18 s. **600 ms** opacity crossfade
between two ping-ponged elements with `prepareFirstFrame` seek-to-t=0.

| Intent | When |
|---|---|
| `idle_calm` | Cold start + 75% rotation weight + reset target after every Tier 2 release |
| `idle_thinking` | 10% rotation weight + motivated-idle observer when `voice_state="thinking"` >2s |
| `misc_hair_touch` | 15% rotation weight |

### Tier 1 — ambient interjections (one-shot, NOT user-driven)

Same idle-rotate task rolls 35% chance per tick. Plays one-shot over
Tier 0 with **120 ms crossfade in / 500 ms fade out** at natural end.
**Cannot fire when Tier 2 is active** (structural rule; replaces today's
manual `_schedule_sip_after` guard list).

| Intent | When |
|---|---|
| `misc_sip_drink` | Idle roll (45% intra-weight) **OR** motivated observer ~600ms after a `comment_response_audio` ≥3s, debounced 30s |
| `misc_walk_off_return` | Idle roll only (25% intra-weight) |
| `misc_glance_aside` | Idle roll only (30% intra-weight) |
| **`welcome` (NEW)** | Idle roll, low probability — re-greets new viewers occasionally |

### Tier 2 — user-driven reactive states

Fires from explicit pipeline events. **120 ms crossfade in** over Tier 0/1.
**Auto-releases at natural end** — no explicit `tier2_release` event.

| Intent | When | Loop? | Substrate |
|---|---|---|---|
| `listening_attentive` | VoiceMic `mic_pressed` (USE_BACKCHANNEL) | yes, muted | `idle_reading_comments.mp4` (eyes-down listening pose) |
| `reading_chat` | Every routed comment, 3.5 s minimum hold | yes | `idle_reading_comments.mp4` (same listening pose) |
| `bridge_<intent>` (NEW: WAV2LIP SUBSTRATE) | Body language picked from question/compliment/objection pool, response audio overlaid via Wav2Lip | one-shot | one of the picks above |
| `pitch_veo` | `dispatch_audio_first_pitch` from phone-video upload | yes, muted | one of the new `pitch_*` picks |

**Cross-tier rule:** Tier 2 active → Tier 1 force-released (50ms quick fade)
+ no Tier 1 schedule until Tier 2 releases. Tier 1 active → trumped by
Tier 2 z-order; runs to natural end underneath.

---

## Routing flow (the conversation loop)

```
audience_comment WS broadcast (or operator simulate_comment)
        │
        ▼
run_routed_comment(text)
        │
        ├─► Director.emit_reading_chat()         ←── Tier 2: reading_chat fades in (120ms)
        │
        ├─► classify_comment_gemma(text)         ←── on-device, ~150ms
        │       ↓
        │   {type: question | compliment | objection | spam}
        │
        ├─► router.decide()
        │       ↓
        │   tool ∈ { respond_locally, play_canned_clip, block_comment, escalate_to_cloud }
        │
        ├─► [if compliment] _is_personal_preference(text) → compliment_personal vs compliment
        │
        ├─► pick_bridge_clip(label)              ←── from phase0/assets/bridges/<intent>/
        │
        ├─► TTS the response text → audio bytes + word_timings
        │
        ├─► Wav2Lip(bridge_clip, audio) → composite mp4 with mouth swapped
        │
        ▼
Director.play_response(composite_url, muted=True)
        │
        ├─► Tier 2 ping-pongs reading_chat → response (120ms crossfade)
        ├─► Standalone <audio> plays the TTS
        ├─► KaraokeCaptions track currentTime → word-by-word reveal
        │
        ▼
Response video natural end → onEnded
        │
        ├─► Tier 2 fades out (500ms), Tier 0 visible underneath
        ├─► Maybe motivated `misc_sip_drink` scheduled (Tier 1 one-shot)
        │
        ▼
back to idle floor, rotation resumes
```

---

## What still needs implementation (next sessions)

- [ ] **Re-tier the Director** — rename `play_response`/`play_bridge`/etc. to emit `layer="tier2"`; move `misc_*` interjections to `layer="tier1"`; add force-release-Tier-1-on-Tier-2-emit hook; delete dead `play_pitch` (already gone) + `play_judge_object_opener` (already gone).
- [ ] **Rewrite `pick_bridge_clip`** to read from `phase0/assets/bridges/<intent>/` only. Add `neutral` → `question` alias. Add `compliment_personal` → `compliment_E` mapping.
- [ ] **Wire Wav2Lip substrate selection** so `_run_escalate_to_cloud` (and friends) pass the picked bridge clip as the substrate instead of always using `state_pitching_pose_speaking_1080p.mp4`.
- [ ] **Personal-preference router rule** — keyword regex in `agents/router.py` after `type=compliment` classification. Plus a different LLM prompt template for first-person preference responses.
- [ ] **Update useAvatarStream + LiveStage** for the third tier — add `setTier2`, render tier2A/tier2B `<video>` elements at z-index 2, clone the Tier 1 driver for Tier 2 with shared crossfade machinery.
- [ ] **Director constants split** — TIER1_CROSSFADE_MS / TIER1_FADEOUT_MS for ambient (today's values), separate TIER2_CROSSFADE_MS / TIER2_FADEOUT_MS for reactive (likely same numbers but named differently for clarity).
- [ ] **iOS sync (Cody)** — heads-up that Swift `VideoDirector.swift` would need the same tier split if it ever shows reactive states. Today it's single-AVPlayer so the model maps to "Tier 2 only" implicitly.
