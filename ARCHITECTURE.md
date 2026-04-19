# EMPIRE — Architecture

> 24/7 multilingual AI livestream-shopping seller, built in 24 hours.
> Voice in → product analyzed → avatar pitches → comments answered → all on a budget.

## Not in scope today (read this first)

To set expectations cleanly:

- **No live platform integration.** No actual TikTok Shop / Instagram Live / WhatsApp connection. The dashboard chat panel simulates comments. The router + response pipeline is identical to what would feed a real platform integration; the integration adapter itself (planned Sprint 4 for TikTok Shop) is not yet built.
- **Single-tenant.** One product, one stream, one operator at a time. Multi-product is shipped (Sprint 1.3) but multi-stream isolation is roadmap (Sprint 5+).
- **English only.** ElevenLabs `language_code="en"` is hardcoded today. Multi-language is roadmap (Sprint 3).
- **CLOSER agent is not built.** The pitched DM auto-responder is planned for Sprint 4.
- **Cloud generative photo variants** (Vertex AI Imagen / Stability AI) not integrated. CREATOR v0 ships rembg + PIL + ffmpeg variants only.

Everything else in this doc is shipped and runnable.

## The local-first thesis

Every step a small seller does on TikTok Shop or Whatnot — analyzing the product, drafting the script, answering the same five questions an hour — costs money and a human if you do it on the cloud. We push the obvious ones onto the seller's own machine via on-device LLM (Cactus + Gemma 4), and only escalate to cloud (Claude, Veo, ElevenLabs, RunPod) when the question genuinely needs it.

**Result: $144/month per seller end-to-end vs ~$9,600/month if every step were cloud.** Same multilingual avatar quality. Same lipsync. Different unit economics.

**~90% of live-stream comments never leave the laptop.** That's the product.

## What runs where

```
LAPTOP (M-series Mac, on-device)         RUNPOD POD (5090, $0.40/hr)
├── Cactus + Gemma 4 E4B                 ├── Wav2Lip server (port 8010)
│   ├── whisper-base transcription       │   ├── In-process model load
│   ├── classify_comment_gemma           │   ├── Per-source face cache
│   ├── analyze_with_gemma (vision)      │   ├── Per-source frame memcache
│   └── parse_voice_intent_gemma         │   └── 1-2s warm for 4s response
│                                         │
├── FastAPI orchestrator (port 8000)     ├── LatentSync 1.6 server (port 8011)
│   ├── /api/voice_comment               │   ├── Diffusion-based, 512×512
│   ├── /api/respond_to_comment          │   ├── DeepCache enabled, fp16
│   ├── /ws/dashboard (broadcast bus)    │   └── ~50s for 4s — pitch only
│   ├── avatar director (FSM)            │
│   └── 4-tool comment router            └── (MuseTalk on 8012 — exploratory)
│
└── Vite + React dashboard (port 5173)
    ├── LiveStage (avatar + voice pills)
    ├── VoiceMic (PTT mic)
    ├── ProductPanel (Spin3D carousel)
    ├── ChatPanel (simulated comments)
    ├── RoutingPanel (live router telemetry)
    └── AgentLog (real-time pipeline trace)

CLOUD APIs (escalation-only)
├── Anthropic Claude (deep responses)    ├── ElevenLabs v3 (TTS + bridges)
├── Vertex AI Veo (idle library refresh) └── Gemini Speech (cloud STT fallback)
```

---

## The four-tool router

```
 VIEWER COMMENT ──► classify_comment_gemma (on-device) ──► router ──► one of four paths
                                                                     ├─ respond_locally      0 ms router + instant pre-rendered MP4
                                                                     ├─ play_canned_clip     0 ms router + instant bridge clip
                                                                     ├─ block_comment        0 ms router + silent (counter++)
                                                                     └─ escalate_to_cloud    ~5 s total (Bedrock + TTS + Wav2Lip)
```

| Agent | Model | Where | What it does |
|---|---|---|---|
| **Voice in** | whisper-base (Cactus) | `backend/agents/eyes.py:_cactus_transcribe_audio` | Push-to-talk → 16 kHz mono → text. **244 ms observed**, serialized via `_cactus_whisper_lock`. Fallback: Gemini 2.5 Flash. |
| **Classify** | Gemma 4 E4B (Cactus) | `backend/agents/eyes.py:classify_comment_gemma` | Comment → `{type: question\|compliment\|objection\|spam, draft_response}` JSON. Serialized via `_cactus_model_lock`. |
| **Product vision** | Gemma 4 E4B vision (Cactus) | `backend/agents/eyes.py:analyze_with_gemma` | Photo → product JSON (name, materials, price hints, selling points). |
| **Voice intent** | Gemma 4 E4B (Cactus) | `backend/agents/eyes.py:parse_voice_intent_gemma` | Voice text → `{action, price, target_audience}` structured extraction. |
| **Router** | Rule-based Python, fed by Gemma 4 classify | `backend/agents/router.py:_rule_based_decide` | 0 ms decision. 100% accurate on the demo test suite. 19 parametrized unit tests in `backend/tests/test_router.py`. |

### Why rule-based, not Gemma 4 tool-calling, for the router

We probed Cactus tool-calling on Gemma 4 E4B with our exact `TOOL_SCHEMA` (`backend/agents/router.py:46–105`): latency 3–31 s, accuracy 1/4 on demo inputs, occasional invented tool names. Rule-based fed by Gemma 4 classify gets us 0 ms routing + 100% accuracy on the same inputs. The Gemma 4 classification *output* is what informs the router — so we still run Gemma 4 on every comment, we just don't ask it to emit the final tool call. FunctionGemma (270M, purpose-built tool-caller) is the right model for that job and is a 1–2h wire-up post-demo.

---

## The pre-rendered local answer layer

The `respond_locally` path isn't a fast LLM — it's *no LLM at all*. Per product we pre-render 10–20 MP4s at alpha time (one per Q&A entry) using the same Wav2Lip + ElevenLabs pipeline we use for live escalates.

- **Source:** `backend/data/products.json` — per-product `qa_index` with keyword arrays + answer text + URL.
- **Match:** `backend/agents/router.py:_match_product_field` — best-hit keyword scoring, multi-word keys as substrings, single-word as tokens.
- **Pre-render script:** `scripts/render_local_answers.py` — iterates qa_index, TTS → Wav2Lip → `backend/local_answers/<slug>.mp4`. Idempotent (`--force` to re-render).
- **Serve:** `/local_answers/*` static mount at `backend/main.py`.
- **Observed render cost:** 4.6 s/clip warm, 10 clips in 46 s wall-time.
- **For the demo product (wallet, 10 entries) the disk footprint is 51 MB.**

`backend/local_answers/*` is `.gitignore`-d. Regenerate on fresh clones via [`README.md`](./README.md).

---

## Lipsync stack — what we landed on

| pipeline | warm latency (4s response) | quality | role |
|---|---|---|---|
| **wav2lip + unsharp** (production) | ~1.5s | clean, soft mouth | live comment responses |
| GFPGAN restoration (explored) | 5-7s | introduces color/jitter artifacts | shipped behind env flag, disabled |
| **LatentSync 1.6 stage2_512** | ~50s | studio-grade, indistinguishable | pre-rendered pitch only |
| MuseTalk 1.5 (deploying) | TBD | TBD | exploratory third option |

**The demo-time live path is wav2lip+unsharp because it's the only one that fits the sub-8s budget.** The pre-rendered pitch (the long "sell this product" video that plays once per product) goes through LatentSync 1.6 at full 512 resolution because we have ~60 s render time before the avatar needs to start talking.

GFPGAN on the wav2lip output looked promising in theory (face restorer can sharpen the soft 96 px wav2lip mouth) but in practice produced visible color-shift / face-jitter artifacts worse than the clean wav2lip baseline. Code is left in place behind `GFPGAN_ENABLED=1` for future re-evaluation.

### wav2lip server hot-path (`phase0/runpod/wav2lip_server_v2.py`)

- Loads Wav2Lip + RetinaFace **once** at startup, keeps in GPU memory
- Caches face-detection bboxes **per source video sha256** on disk → state videos detect once ever, not per-request
- In-memory cache of decoded frames keyed by `(sha256, out_height)` → warm call skips video I/O entirely
- `/lipsync_fast` endpoint: source video already on pod, only the audio is uploaded, saves ~1 s per call vs `/lipsync`
- `libx264 -preset ultrafast -crf 20` for mux (NVENC unavailable on this pod's ffmpeg build for sm_120)

---

## Gold path walkthrough — voice → avatar

What a judge sees when they hold the mic and say "is it real leather":

```
t=0       Release mic button
t=0.05    MediaRecorder flushes ~1.2 s of WebM/Opus → POST /api/voice_comment
t=0.15    FastAPI receives audio → ffmpeg normalize to 16 kHz mono WAV (50 ms)
t=0.39    Cactus whisper transcription: "is it real leather" (244 ms)
t=0.39    Broadcast voice_transcript → VoiceMic chip + voice-state pill → 'thinking'
t=0.55    classify_comment_gemma returns {type: question} (~150 ms, on-device)
t=0.55    router._rule_based_decide picks respond_locally (0 ms)
t=0.55    Broadcast routing_decision → RoutingPanel pulses green, pill → 'responding'
t=0.55    director.play_response('/local_answers/wallet_real_leather.mp4')
t=0.9     Avatar on screen is speaking the pre-rendered response.
```

**Under 1 second, end-to-end, with zero cloud spend.** That's what the local-first routing unlocks.

Compare the escalate path on "how does this compare to the Apple Watch":

```
t=0       Whisper transcribes (244 ms)
t=0.4     classify → question; router → escalate_to_cloud
          Director plays canned bridge ("great question...") in parallel
t=0.5     Bedrock Claude Haiku invoked
t=1.5     Claude returns 48-char response text
t=1.9     ElevenLabs turbo_v2_5 TTS (MP3 bytes in hand)
t=1.9     POST /lipsync_fast to pod with audio
t=6.4     Wav2Lip returns 2.5 MB MP4 (warm cache, 4.5 s)
t=6.4     Director plays the response MP4
```

Escalate is 6 s warm, 16 s cold. The bridge clip fills the silence. For the 10% of comments that require live reasoning, this is acceptable; the local path covers the other 90%.

---

## Latency budget per stage

| stage | warm target | what it is |
|---|---|---|
| voice transcription (Cactus local) | ≤500 ms | whisper-base on Cactus |
| comment classification | 2-4s today (target ≤500ms with NPU mlpackage) | Gemma 4 classify_comment_gemma |
| router decision (local) | ≤1 ms | rule-based dispatcher |
| local response (pre-rendered MP4 play) | ≤50 ms | disk read + HLS stream |
| cloud response (when escalated) | 1-2 s | Claude streaming |
| TTS | ≤500 ms TTFB | ElevenLabs v3 streaming |
| Wav2Lip render (~4 s audio) | 1.5-2 s warm | pod sub-8s budget |
| WS broadcast + crossfade | ≤200 ms | dashboard frame swap |
| **Total live response (local)** | **< 1 s** | comment → avatar speaking |
| **Total live response (escalate)** | **5-7 s warm** | comment → avatar speaking |

---

## Cost economics — why local-first

For a single seller running a 1000-viewer livestream, our routing telemetry projects **~90% of comments are routine** (price, materials, shipping, sizing, returns, warranty) and answer locally for free. Only ~10% require cloud LLM reasoning.

| approach | per-month cost (est) |
|---|---|
| 100% cloud (LLM + TTS + lipsync per comment) | ~$9,628 |
| Empire (local-first routing, cloud only on escalation) | ~$144 |

**Breakdown of the $144/month:**
- ~$60 ElevenLabs (TTS for ~10% of responses + bridges)
- ~$25 Claude (cloud responses for the 10% that escalate)
- ~$50 RunPod (5090 spot, ~3 active hours/day for live demos)
- ~$9 Anthropic spot for product analysis

**The on-device leg (Cactus + Gemma 4 voice + router + Q&A index) is $0.** That's the whole pitch.

---

## Fallback chain

```
Transcription:   Cactus whisper-base (local, 244 ms) → Gemini 2.5 Flash Speech
Classification:  Cactus Gemma 4 E4B → Ollama (gemma4:e4b) → {"type":"question"}
Router:          Rule-based dispatcher (always works; 0 ms; deterministic)
Response:        escalate → Claude Haiku → Claude Sonnet (richer) → degraded text-only
TTS:             ElevenLabs (Starter plan) → no-audio degraded path
Lip-sync:        Wav2Lip on pod → static substrate play-through (no lip-sync)
WiFi:            USB tethering as primary if venue WiFi is poor
```

**Zero step in the golden path requires cloud.** Everything cloud is a *fast-path upgrade*, not a dependency.

---

## Ownership matrix (who built what)

| component | owner | files |
|---|---|---|
| Cactus install + Gemma 4 voice | Cody | local install |
| Voice transcription (whisper-base) | Cody | `backend/agents/eyes.py` |
| `POST /api/voice_comment` + run_routed_comment | Cody | `backend/main.py` |
| 4-tool router | Cody | `backend/agents/router.py` |
| Local product Q&A index + pre-render pipeline | Cody | `backend/data/products.json`, `scripts/render_local_answers.py` |
| VoiceMic UI | Cody | `dashboard/src/components/VoiceMic.jsx` |
| Routing telemetry panel | Cody | `dashboard/src/components/RoutingPanel.jsx` |
| Router unit tests | Cody | `backend/tests/test_router.py` |
| Wav2Lip + GFPGAN integration | Aditya | `phase0/runpod/wav2lip_server_v2.py` |
| LatentSync 1.6 deploy + server | Aditya | `phase0/runpod/deploy_latentsync.sh`, `latentsync_server.py` |
| MuseTalk exploration | Aditya | `phase0/runpod/deploy_musetalk.sh` |
| Voice-state pill + Spin3D voice reactivity | Aditya | `dashboard/src/components/ProductPanel.jsx`, `LiveStage.jsx` |
| Avatar director (FSM) | shared | `backend/agents/avatar_director.py` |
| 3D carousel | Aditya | `dashboard/src/components/Spin3D.jsx`, `backend/agents/threed.py` |
| Substrate video library + Veo pipeline | Aditya | `phase0/assets/states/` |

---

## Threat model / known edges

| Risk | Mitigation |
|---|---|
| Cactus C library is not re-entrant on one handle | `threading.Lock()` per handle (whisper + Gemma 4). Verified under parallel requests (no segfault). |
| Venue WiFi flaky on hackathon day | USB tethering; local path is pod-independent for 90% of answers. |
| Pod redeploy stales face cache | `/prewarm` POST per substrate at startup; backend re-warms after teammate signals redeploy. |
| ElevenLabs free tier blocked (abuse detector) | Starter plan ($5/mo) resolves; key validated on restart. |
| Gemma 4 vision hallucinates in sell pipeline | Claude Bedrock fallback on sparse output or parse error. Voice text acts as anchor. |
| Whisper returns empty on silence / noise | `_is_noise_transcript` filter + Gemini fallback; bare fillers like "Mhm." return `source: no_speech`. |
| React keys instability in RoutingPanel pulse | Monotonic `seq` counter in `useEmpireSocket` reducer. |
| NaN poisons `cost_saved_usd` running total | `Number.isFinite` guard in reducer. |
| Abrupt WS disconnect → `ValueError: list.remove` | try/except around `.remove()` in `broadcast_to_dashboards`. |

---

## Honest tradeoffs

1. **Wav2Lip mouth is soft.** 96 px native output; even with unsharp mask the teeth/lip detail isn't studio-grade. We tested GFPGAN restoration (worse — color artifacts), GFPGAN mouth-only with temporal interp (still worse than baseline), and LatentSync 1.6 (gorgeous but ~50 s render). For sub-8 s live we accept wav2lip's quality floor and use LatentSync only for the pre-rendered pitch.

2. **Single-pod single-GPU.** All renders serialized through one 5090. For a real seller deployment, you'd shard sources across multiple pods keyed by source video sha256 (the face cache shards naturally).

3. **Cactus model size.** Gemma 4 E4B is fast on M-series silicon but its reasoning is shallow — anything beyond simple intent classification + Q&A field match falls through to Claude.

4. **State video library is fixed.** Idle / speaking / bridge clips are pre-rendered via Veo. Adding a new pose requires a new Veo render cycle (~2 min/clip). Not a runtime constraint, but it caps how "personalized" the avatar can be without re-rendering its body library.

5. **No actual TikTok Shop integration in this build.** The chat panel simulates comments; we'd plug into the real chat WS in production.

---

## What's not in the demo but is on the roadmap

1. **FunctionGemma-270m-it as router primary.** Purpose-built tool-calling model; schema already exists; probe showed the 270M model needs its own eval before going live.
2. **Gemma 4 vision fast-path for product analysis.** Current sell pipeline uses Claude Bedrock; front with Gemma 4 vision and fall back on sparse output.
3. **BRAIN agent — conversion tracking + offline fine-tune loop.** Per-stream telemetry (which Q&A answer → which add-to-cart) feeding a weekly retrain. The defensible moat.
4. **CLOSER agent — outbound DMs.** "Comment asked about sizing → DM them 24 h later with a sizing chart + checkout link."
5. **Mobile capture surface.** Expo Go shell for phone-side camera/mic → laptop over WebSocket. Currently the dashboard runs on the same laptop that handles inference.
