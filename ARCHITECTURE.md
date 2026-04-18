# EMPIRE — Architecture

> 24/7 multilingual AI livestream-shopping seller, built in 24 hours.
> Voice in → product analyzed → avatar pitches → comments answered → all on a budget.

## The local-first thesis

Every step a small seller does on TikTok Shop or Whatnot — analyzing the
product, drafting the script, answering the same five questions an hour —
costs money and a human if you do it on the cloud. We push the obvious
ones onto the seller's own machine via on-device LLM (Cactus + Gemma 4),
and only escalate to cloud (Claude, Veo, ElevenLabs, RunPod) when the
question genuinely needs it.

Result: $144/month per seller end-to-end vs ~$9,600/month if every step
were cloud. Same multilingual avatar quality. Same lipsync. Different
unit economics.

## What runs where

```
LAPTOP (M-series Mac, on-device)         RUNPOD POD (5090, $0.40/hr)
├── Cactus + Gemma 4 (270M-it)           ├── Wav2Lip server (port 8010)
│   ├── voice transcription              │   ├── In-process model load
│   ├── product Q&A index                │   ├── Per-source face cache
│   └── FunctionGemma router             │   ├── Per-source frame memcache
│                                         │   └── 1-2s warm for 4s response
├── FastAPI orchestrator (port 8000)     │
│   ├── /api/voice_comment               ├── LatentSync 1.6 server (port 8011)
│   ├── /api/respond_to_comment          │   ├── Diffusion-based, 512×512
│   ├── /ws/live (broadcast bus)         │   ├── DeepCache enabled, fp16
│   └── avatar director (FSM)            │   └── ~50s for 4s — pitch only
│                                         │
└── Vite + React dashboard (port 5173)   └── (MuseTalk on 8012 — exploratory)
    ├── LiveStage (3D carousel + avatar)
    ├── VoiceMic (PTT, waveform viz)
    ├── RoutingPanel (live router telemetry)
    ├── ChatPanel (simulated comments)
    ├── AgentLog (real-time pipeline trace)
    └── Spin3D (auto-rotating product carousel)

CLOUD APIs (escalation-only)
├── Anthropic Claude (deep responses)    ├── ElevenLabs v3 (TTS + bridges)
├── Vertex AI Veo (idle library refresh) └── Gemini Speech (cloud STT fallback)
```

## Lipsync stack — what we landed on

| pipeline | warm latency (4s response) | quality | role |
|---|---|---|---|
| **wav2lip + unsharp** (production) | ~1.5s | clean, soft mouth | live comment responses |
| GFPGAN restoration (explored) | 5-7s | introduces color/jitter artifacts | shipped behind env flag, disabled |
| **LatentSync 1.6 stage2_512** | ~50s | studio-grade, indistinguishable | pre-rendered pitch only |
| MuseTalk 1.5 (deploying) | TBD | TBD | exploratory third option |

The honest tradeoff: **the demo-time live path is wav2lip+unsharp because it's
the only one that fits the sub-8s budget**. The pre-rendered pitch (the long
"sell this product" video that plays once per product) goes through LatentSync
1.6 at full 512 resolution because we have ~60s render time before the avatar
needs to start talking.

GFPGAN on the wav2lip output looked promising in theory (face restorer can
sharpen the soft 96px wav2lip mouth) but in practice produced visible
color-shift / face-jitter artifacts that the user judged worse than the clean
wav2lip baseline. Code is left in place behind `GFPGAN_ENABLED=1` for future
re-evaluation.

### wav2lip server hot-path (`phase0/runpod/wav2lip_server_v2.py`)

- Loads Wav2Lip + RetinaFace **once** at startup, keeps in GPU memory
- Caches face-detection bboxes **per source video sha256** on disk → state
  videos detect once ever, not per-request
- In-memory cache of decoded frames keyed by `(sha256, out_height)` → warm
  call skips video I/O entirely
- `/lipsync_fast` endpoint: source video already on pod, only the audio is
  uploaded, saves ~1s per call vs `/lipsync`
- `libx264 -preset ultrafast -crf 20` for mux (NVENC unavailable on this
  pod's ffmpeg build for sm_120)

## Demo data flow — comment response

```
1. [DASHBOARD] User holds VoiceMic, asks: "Is this real leather?"
   → MediaRecorder webm → POST /api/voice_comment

2. [LAPTOP] Cactus + Gemma 4 voice transcription
   → 320ms on-device, transcript broadcast on /ws/live

3. [LAPTOP] FunctionGemma router classifies the comment
   → "respond_locally" tool with high confidence
   → Local product Q&A index returns answer
   → Routing telemetry broadcast: tool=respond_locally, was_local=true,
     ms=47, cost_saved_usd=$0.012

4. [LAPTOP] ElevenLabs streams TTS audio for the response

5. [POD] Wav2Lip renders the response over the substrate clip the
   carousel is currently showing → mp4 bytes back

6. [DASHBOARD] LiveStage crossfades the response over the bridge clip
   that was playing while we waited

Total: ~5-7s warm path end-to-end
```

## Ownership matrix (who built what)

| component | owner | files |
|---|---|---|
| Cactus install + Gemma 4 voice | Cody | local install |
| Voice transcription module | Cody | `backend/agents/voice.py` |
| `POST /api/voice_comment` | Cody | `backend/main.py` |
| FunctionGemma router | Cody | `backend/agents/router.py` |
| Local product Q&A index | Cody | `backend/data/products.json` |
| VoiceMic UI + waveform | Cody | `dashboard/src/components/VoiceMic.jsx` |
| Routing telemetry panel | Cody | `dashboard/src/components/RoutingPanel.jsx` |
| Wav2Lip + GFPGAN integration | (You) | `phase0/runpod/wav2lip_server_v2.py` |
| LatentSync 1.6 deploy + server | (You) | `phase0/runpod/deploy_latentsync.sh`, `latentsync_server.py` |
| MuseTalk exploration | (You) | `phase0/runpod/deploy_musetalk.sh` |
| Avatar director (FSM) | shared | `backend/agents/avatar_director.py` |
| Item rotation 3D carousel | (You) | `dashboard/src/components/Spin3D.jsx`, `agents/threed.py` |

## Latency budget per stage

| stage | warm target | what it is |
|---|---|---|
| voice transcription (Cactus local) | ≤500ms | Cactus + Gemma 4 STT |
| comment classification | ≤100ms | FunctionGemma router decision |
| local response (when routed local) | ≤50ms | products.json field match |
| cloud response (when escalated) | 1-2s | Claude streaming |
| TTS | ≤500ms TTFB | ElevenLabs v3 streaming |
| Wav2Lip render (~4s audio) | 1.5-2s warm | pod sub-8s budget |
| WS broadcast + crossfade | ≤200ms | dashboard frame swap |
| **Total live response** | **5-7s** | comment → avatar speaking |

## Cost economics — why local-first

For a single seller running a 1000-viewer livestream, our routing telemetry
projects **~90% of comments are routine** (price, materials, shipping,
sizing, returns, warranty) and answer locally for free. Only ~10% require
cloud LLM reasoning.

| approach | per-month cost (est) |
|---|---|
| 100% cloud (LLM + TTS + lipsync per comment) | ~$9,628 |
| Empire (local-first routing, cloud only on escalation) | ~$144 |

Breakdown of the $144/month:
- ~$60 ElevenLabs (TTS for ~10% of responses + bridges)
- ~$25 Claude (cloud responses for the 10% that escalate)
- ~$50 RunPod (5090 spot, ~3 active hours/day for live demos)
- ~$9 Anthropic spot for product analysis

The on-device leg (Cactus + Gemma 4 voice + router + Q&A index) is **$0**.
That's the whole pitch.

## Honest tradeoffs

1. **Wav2Lip mouth is soft.** 96px native output, even with unsharp mask
   the teeth/lip detail isn't studio-grade. We tested GFPGAN restoration
   (worse — color artifacts), GFPGAN mouth-only with temporal interp
   (still worse than baseline by user judgment), and LatentSync 1.6
   (gorgeous but ~50s render). For sub-8s live we accept wav2lip's
   quality floor and use LatentSync only for the pre-rendered pitch.

2. **Single-pod single-GPU.** All renders serialized through one 5090.
   For a real seller deployment, you'd shard sources across multiple pods
   keyed by source video sha256 (the face cache shards naturally).

3. **Cactus model size.** Gemma 4 270M-it is small enough to be fast on
   M-series silicon but its reasoning is shallow — anything beyond
   simple intent classification + Q&A field match falls through to
   Claude.

4. **State video library is fixed.** Idle / speaking / bridge clips are
   pre-rendered via Veo. Adding a new pose requires a new Veo render
   cycle (~2 min/clip). Not a runtime constraint, but it caps how
   "personalized" the avatar can be without re-rendering its body
   library.

5. **No actual TikTok Shop integration in this build.** The chat panel
   simulates comments; we'd plug into the real chat WS in production.
```
