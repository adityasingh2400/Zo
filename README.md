# EMPIRE

**A 24/7 multilingual AI livestream-shopping seller, run end-to-end by Gemma 4 on Cactus.** Built at the Cactus × Google DeepMind × YC Gemma 4 Voice Agents hackathon.

> Voice in → Gemma sees the product → Gemma writes the pitch → Gemma classifies every viewer comment → Gemma picks which pre-rendered clip to stitch in next. ~90% of comments never leave the laptop. $144/mo unit economics vs. ~$9,628/mo for an equivalent cloud + human team.

See [`EMPIRE-PITCH.md`](./EMPIRE-PITCH.md) for the full thesis, [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the build, and [`EMPIRE-COST-ANALYSIS.md`](./EMPIRE-COST-ANALYSIS.md) for the cost model. [`DESIGN_PRINCIPLES.md`](./DESIGN_PRINCIPLES.md) indexes the Lidwell PDF for agents reasoning about UI decisions.

---

## Why this exists — the market and the gap

Live commerce is selling products through livestreams: a host shows the item, talks about it, answers questions in chat, and viewers buy directly from the stream. QVC for the TikTok generation, run by individual sellers instead of TV networks.

| Metric | Number |
|---|---|
| US live commerce market (2026) | **$68 billion** |
| Global live commerce (China) | **$500 billion** |
| TikTok Shop global sales (last quarter) | **$19 billion** |
| US TikTok Shop YoY growth | **+125%** |
| Live commerce conversion rate | **30%** (vs. 2-3% for traditional e-commerce — **10× the rate**) |

Live converts at a different category than static product pages. But going live is brutal — it's the reason 95% of sellers are locked out of a $68B market:

| Problem | What it actually costs |
|---|---|
| **Languages** | A human host speaks 1–2 languages. Your customers are in 140+. |
| **Time zones** | A human host can do ~8 hours/day. Your buyers are awake 24/7 — you miss two-thirds of the world. |
| **Team cost** | Solo host: $5,160/mo. Agency: **$8,000–$12,500/mo**. Plus camera, lighting, producer, comment moderator, sales coach. |
| **Cloud-AI alternative** | If you naively replace the team with cloud LLM + cloud TTS + cloud avatar, you trade a $10K/mo human bill for a **$9,628/mo cloud bill**. The unit economics still don't work. |
| **Empire** | **$144/mo, 140+ languages, 24/7, every comment answered.** |

That's the wedge. The only way the unit economics close is if the on-device model handles ~90% of the work. **That model is Gemma 4 on Cactus.**

---

## How Gemma 4 runs the entire seller

Gemma 4 E4B (via the Cactus runtime on Apple Silicon) is the brain of every decision EMPIRE makes. Not a fallback, not an accelerator — the **primary intelligence layer**. Every other component (Wav2Lip, ElevenLabs, the pre-rendered clip library, the dashboard) is plumbing that Gemma drives.

### Gemma's five jobs in this app

All five run **on the seller's laptop**. Zero cloud spend, zero IP leak, zero latency to a datacenter.

| # | Gemma job | Code | What it does |
|---|---|---|---|
| **1** | **Pitch generation (vision + script)** | `backend/agents/eyes.py:analyze_and_script_gemma` — fused single call | Looks at the product photo + the seller's narration. Returns one JSON: `{product:{name, materials, selling_points, visual_details}, script:"<70-word TikTok-energy pitch>"}`. **Replaces a Claude vision call.** Falls back to Claude only if Gemma's JSON is malformed. |
| **2** | **Comment classification (intent routing)** | `backend/agents/eyes.py:classify_comment_gemma` | Every viewer comment → `{type: question\|compliment\|objection\|spam, draft_response}`. This is the input the router uses to pick which pre-rendered clip plays next. |
| **3** | **Voice intent parsing** | `backend/agents/eyes.py:parse_voice_intent_gemma` | The seller says *"sell this for $89, target young professionals."* Gemma extracts `{action:"sell", price:89, target_audience:"young professionals"}` — structured, no regex. |
| **4** | **On-device Whisper transcription** | `backend/agents/eyes.py:_cactus_transcribe_audio` | Whisper-base via Cactus, **244 ms observed** for a ~1.2 s utterance. Push-to-talk mic → text without a Gemini API round-trip. |
| **5** | **Drafting the comment response itself** | Same `classify_comment_gemma` call returns a `draft_response` field that becomes the spoken audio in the dispatch path | When the local Q&A index doesn't match, Gemma's draft_response is what gets TTS'd and lip-synced onto the next pre-rendered substrate clip. **No second LLM call** — one Gemma forward pass produces both the routing decision *and* the words the avatar will say. |

**Without Gemma, EMPIRE doesn't have a product.** The cloud fallback chain (Claude / Gemini Speech) exists only as a safety net for the demo — every shipped path runs Gemma first.

---

## The illusion of a live AI avatar — actually intelligent stitching of pre-rendered clips

**Read this section carefully — it's the whole trick.**

EMPIRE does **not** generate video in real time. We are not running a diffusion video model on your laptop, and we are not asking Gemma 4 to do something it can't do. What looks to the audience like a live, reactive AI avatar is actually a **library of pre-rendered MP4 clips that Gemma 4 intelligently stitches together** in real time.

The audience perceives liveness. The compute is constant-time clip selection + a lightweight lip-sync overlay. **Gemma is the choreographer.**

### The three-tier stage — pre-rendered everything, Gemma stitches it live

The dashboard composites **three tiers of pre-rendered MP4** onto the screen at all times. None of them are generated frame-by-frame at runtime. Each tier is a different *kind* of pre-rendered content, and **Gemma 4 is the router that decides which tier fires, when, and what audio rides on top of it.**

```
                ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
                ┃        G E M M A   4   E 4 B           ┃
                ┃        ─────────────────────           ┃
                ┃        on-device · ~150 ms / call      ┃
                ┗━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┛
                                     │
                                     │  routes EVERY state
                                     │  transition on stage
                                     ▼
                            ┌─────────────────┐
                            │    DIRECTOR     │   choreographer FSM
                            │  (per-process)  │   (backend/agents/
                            └────────┬────────┘    avatar_director.py)
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
        ▼                            ▼                            ▼
  ┌───────────┐                ┌───────────┐                ┌───────────┐
  │  TIER 0   │                │  TIER 1   │                │  TIER 2   │
  │  ───────  │                │  ───────  │                │  ───────  │
  │   IDLE    │  ◀── Gemma ──  │  BRIDGE   │  ◀── Gemma ──  │ RESPONSE  │
  │           │   swaps pose   │           │   picks bucket │           │
  │ always on │                │ comment-  │                │ the actual│
  │ silent    │                │ intent    │                │   answer  │
  │ Veo loop  │                │ substrate │                │           │
  └─────┬─────┘                └─────┬─────┘                └─────┬─────┘
        │                            │                            │
        │ ◀──── crossfade IN ─────── │                            │
        │                            │                            │
        │                            │ ◀──── crossfade IN ─────── │
        │                            │                            │
        ▼                            ▼                            ▼
  always painting          plays for ~8 s while         plays once TTS +
  underneath every-        Gemma's drafted              Wav2Lip finishes;
  thing else; never        response renders             body language inherits
  goes black               in the background            from the Tier 1 pose
```

| Tier | Content | When it fires | Gemma's role |
|---|---|---|---|
| **0 — Idle** | Looping muted Veo render of the avatar's resting pose. The safety net — never goes black. | Continuously. Every other tier crossfades on top of this. | Gemma's `voice_state` signal swaps the active idle pose (calm ↔ thinking) when the operator pauses to think. |
| **1 — Bridge** | 8-second pre-rendered intent substrate matching the comment's emotional register (warm-smile, thoughtful-nod, or "actually here's the deal"). | The moment a comment lands. Crossfades onto Tier 0. | **Gemma's `classify_comment` returns `type` — that field IS the bucket selector.** The Director picks a random clip from the matching bucket. |
| **2 — Response** | Either a fully pre-rendered Q&A MP4 (when Gemma's intent matches a known question) or a fresh Wav2Lip lip-sync onto the Tier 1 substrate (when Gemma drafts a novel response). | Once TTS + Wav2Lip finishes. Crossfades onto Tier 1. | Gemma's `draft_response` is the audio source. Gemma's `qa_index` match decides whether a fresh render is even needed. |

**The "live AI avatar" is Gemma 4 driving three pre-rendered tiers in sequence.** That's it.

### Gemma 4 is the state router

Every state transition on stage is driven by Gemma 4 output. The crucial point: **a single Gemma forward pass produces multiple outputs that fan out and drive multiple tiers in parallel.** No second LLM call, no stacked inference, no extra latency.

```
   ┌─────────────────────────────────────────────────────────────────┐
   │   GEMMA.classify_comment("is it real leather")                  │
   │   ────────────────────────────────────────────                  │
   │                                                                 │
   │   returns ONE JSON object:                                      │
   │     {                                                           │
   │       type:           "question",                               │
   │       draft_response: "Yes, full-grain vegetable-tanned         │
   │                        leather. See the natural grain?",        │
   │       latency_ms:     147                                       │
   │     }                                                           │
   └─────────────────────────────────┬───────────────────────────────┘
                                     │
              ┌──────────────────────┴──────────────────────┐
              │                                             │
              ▼                                             ▼
    TIER 1 BUCKET PICK                            TIER 2 AUDIO + RENDER
    ──────────────────                            ─────────────────────
    uses Gemma's `type` field                     uses Gemma's `draft_response`
    → indexes the matching                        → ElevenLabs TTS streams the
      intent bucket (compliment /                   text (translated to one of
      question / objection)                         6 supported languages on
    → Director picks 1 random                       cache miss)
      pre-rendered substrate                      → Wav2Lip overlays mouth
                                                    pixels onto the Tier 1
                                                    substrate the Director
                                                    just chose. Body language
                                                    is inherited; only the
                                                    mouth region is generated
                                                    live.

   ┌─────────────────────────────────────────────────────────────────┐
   │  (concurrent third channel — same Gemma model, different prompt)│
   │                                                                 │
   │  GEMMA.parse_voice_intent(seller_audio) ──►  drives TIER 0 pose │
   │     voice_state = "thinking" for >2 s  →  Tier 0 crossfades to  │
   │     idle_thinking pose; reverts on next state change.           │
   └─────────────────────────────────────────────────────────────────┘
```

### How the bridge state (Tier 1) is routed

This is the part that does the heavy lifting on every comment: **Gemma's `type` field is the entire routing signal.** No rule engine on top, no heuristic, no second classifier. Whatever Gemma says is the bucket the Director draws from.

```
              ┌─────────────────────────────────────────────────┐
              │   GEMMA.classify_comment("...")                 │
              │   ─────────────────────────────                 │
              │                                                 │
              │   returns:                                      │
              │     type           →  "compliment"              │
              │                    |  "question"                │
              │                    |  "objection"               │
              │                    |  "spam"   (block silently) │
              │     draft_response →  "Yes, full-grain ..."     │
              │                                                 │
              └───────────────┬─────────────────────────────────┘
                              │
                              │  type IS the bucket key
                              ▼
                ┏━━━━━━━━━━━━━╋━━━━━━━━━━━━━┓
                ┃             ┃             ┃
                ▼             ▼             ▼
          ┌───────────┐ ┌───────────┐ ┌───────────┐
          │COMPLIMENT │ │ QUESTION  │ │ OBJECTION │
          │───────────│ │───────────│ │───────────│
          │warm smile │ │thoughtful │ │"actually, │
          │+ slow nod │ │nod, chin- │ │here's the │
          │           │ │hold       │ │deal…"     │
          │           │ │           │ │           │
          │ 5 clips   │ │ 5 clips   │ │ 5 clips   │
          └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
                │             │             │
                └─────────────┼─────────────┘
                              │
                              ▼
        Director picks 1 random clip from the chosen bucket
        and crossfades it onto TIER 1 (above the always-
        painting TIER 0 idle layer).

        Empty bucket / Gemma timeout → default speaking-pose
        substrate; the Director never goes silent.
```

The chosen bridge clip plays for ~8 seconds — comfortably long enough for ElevenLabs to TTS Gemma's `draft_response` and for Wav2Lip to lip-sync that audio onto the **same substrate clip the Director just picked**. So when Tier 2 crossfades over Tier 1, the body language is preserved and only the mouth pixels change. The audience reads it as one continuous gesture: thoughtful nod → start speaking → finish speaking → ease back to idle.

### The fast path — Gemma matches the seller's pre-authored Q&A index

If Gemma's `type` is `question` AND the comment matches a key in the seller's `qa_index`, the router **skips Tier 1 entirely and fires Tier 2 directly with a fully pre-rendered MP4** — sub-600 ms total, zero cloud spend, zero render. The MP4 was rendered once at product onboarding by the same Wav2Lip + ElevenLabs pipeline used live; it lives on disk forever. **Gemma's only job here is to recognize the question well enough to fire the right clip.**

This is the path ~90% of comments take in the demo.

### What's pre-rendered, where, and what it costs

| Tier | Asset | Quantity | Source | Cost |
|---|---|---|---|---|
| **0** | Idle pose loops | 5–8 per avatar | Vertex AI Veo (offline) | Generated offline |
| **0** | Speaking-pose substrates (paired with each idle) | 1 per idle pose | Veo (offline) | Generated offline |
| **0** | Sip / glance / walk-off ambient interjections | A few each | Veo (offline) | Generated offline |
| **1** | Intent bridge substrates (compliment / question / objection) | 3–5 per bucket | Veo (offline), uploaded to pod `/workspace/bridges/<intent>/` | Generated offline |
| **2** | Local Q&A answer MP4s | 10–20 per product | Wav2Lip + ElevenLabs at onboarding | ~4.6 s/clip warm. 10 clips in ~46 s. ~50 MB per product. |
| **2** | Pitch clip (per product) | 1 per product | LatentSync 1.6 (~50 s render) | Played once per stream |
| **2** | Live response stitching | On-demand, ~10% of comments | Wav2Lip lip-sync onto Tier 1 substrate (~1.5–2 s warm on the 5090) | Per Gemma-drafted comment that doesn't match `qa_index` |

**At runtime, the only thing being generated is the mouth region of Tier 2 — and only when Gemma can't reuse an existing clip.** Pose, lighting, eyes, hands, hair, body — every other pixel on screen was rendered before the stream started. Wav2Lip is a *lip-region overlay*, not a face generator.

### Why this matters for what Gemma can do

A judge looking at the demo will reasonably ask: *"How can a 4B-parameter on-device model produce a live, reacting AI avatar?"* The answer is that **it doesn't have to**. Gemma's job is the part Gemma is genuinely good at — **understanding language and routing decisions in <500 ms** — and the pre-rendered clip library handles everything else. We didn't pick this architecture to hide a limitation. We picked it because it's the only architecture where the unit economics close at $144/mo.

Live generative video on-device is currently impossible at acceptable quality and latency on consumer hardware. **Pre-rendered clips + Gemma routing is the production architecture for AI video commerce in 2026.** This is what HeyGen, Khaby Lame's avatar deal, and TikTok's AI Seller Assistant will all converge to within 12 months.

---

## How a single comment flows — three tiers in motion

This is what the audience sees end-to-end. Watch which tier is "live" at each moment — that's the choreography.

### Path A — Gemma matches the seller's pre-authored Q&A index (~90% of comments)

```
   TIME →     0 ms       150 ms      200 ms              600 ms

   TIER 0  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
            idle_calm loop (always painting underneath)

   TIER 1                                  (skipped — fast path)

   TIER 2                          [████████ pre-rendered Q&A MP4 ████████]

              ▲           ▲            ▲                   ▲
              │           │            │                   │
              │           │            │                   └─ Avatar speaks
              │           │            │                      the answer.
              │           │            │                      Tier 2 plays.
              │           │            │
              │           │            └─ Director.play_response()
              │           │               fires Tier 2 directly.
              │           │
              │           └─ Gemma.classify_comment → {type: "question"}.
              │              Router matches qa_index → respond_locally.
              │
              └─ Viewer types "is it real leather"

   Total: < 600 ms.  Zero cloud spend.  Zero render.
   The MP4 was rendered once, at product onboarding, by the same
   Wav2Lip + ElevenLabs pipeline used live.
```

### Path B — Gemma drafts a novel response, Wav2Lip stitches it onto a Tier 1 substrate (~10% of comments)

```
   TIME →   0 s     0.2 s     0.5 s              2.0 s              7.0 s

   TIER 0  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
            idle_calm loop (always painting underneath the whole time)

   TIER 1                  [▓▓▓ thoughtful-nod bridge substrate ▓▓▓]
                             ▲                                    ▲
                             │                                    │
                             │                                    fades out as
                             Gemma's type="question" → Director   Tier 2 fades in
                             picks bucket, crossfades the
                             chosen bridge clip onto Tier 0.

   TIER 2                                              [████ Wav2Lip lip-sync ████]
                                                        ▲
                                                        │
                                                        Wav2Lip done (~2 s warm).
                                                        Audio = Gemma's draft_response
                                                          via ElevenLabs TTS.
                                                        Body language = the Tier 1
                                                          pose, preserved.
                                                        Mouth pixels = the only
                                                          thing rendered live.

      ▲       ▲           ▲                          ▲                  ▲
      │       │           │                          │                  │
   viewer   Gemma       TTS streams                audience          Tier 2 ends,
   types    classify    starts; bridge             hears answer      Tier 0 alone
            returns     visible immediately        speaking +
            type +                                 gesturing in
            draft +                                coherent body
            latency                                language
```

**The "live AI avatar" the audience perceives is three pre-rendered tiers crossfading in sequence under Gemma's command.**

1. **Tier 0** keeps painting the resting pose, no matter what.
2. **Tier 1** crossfades in the intent body language Gemma chose, the instant Gemma classifies the comment.
3. **Tier 2** crossfades over Tier 1 once Gemma's drafted words have been TTS'd and lip-synced.

Three crossfades. One Gemma forward pass. Zero generative video. That's the whole trick.

---

## The cost case — why local-first Gemma is the entire moat

| Approach | Monthly cost (modeled) | Languages | Hours/day | Comment coverage |
|---|---|---|---|---|
| Hire a human team | **$5,160 – $12,500** | 1–2 | 8 | Misses 60–80% during busy streams |
| 100% cloud AI (Claude/GPT for everything) | **~$9,628** | 10–20 | 24 | Every one |
| **Empire (Gemma 4 + Cactus on-device, cloud only on escalation)** | **~$144** | **140+** | **24** | **Every one** |

The on-device leg — Cactus + Gemma 4 voice + classify + Q&A index + draft_response + the pre-rendered clip library — costs **$0** in marginal spend. The only cloud bills are:

- ~$60 ElevenLabs (TTS for the ~10% that escalate + bridge audio)
- ~$25 Bedrock Claude Haiku (cloud responses for the 10% Gemma defers)
- ~$50 RunPod (5090 spot, ~3 active hours/day)
- ~$9 Anthropic spot for product analysis fallback

**This isn't a tech-flex.** It's the only architecture where 24/7 multilingual AI live commerce is economically viable. Without Gemma 4 on-device, the unit economics are the same as a human team — and you've added an AI startup's complexity tax on top.

---

## Run the demo

### Prerequisites

- macOS with Apple Silicon
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

### Regenerating the pre-rendered local-answer library

`backend/local_answers/*.mp4` is gitignored (~50 MB of pre-rendered MP4 per product). A fresh clone has no clips, so every local-index question falls through to the cloud-escalate path. To regenerate:

```bash
# 1. Confirm ELEVENLABS_API_KEY + ELEVENLABS_VOICE_ID are set in .env
# 2. Confirm the Wav2Lip tunnel is up: curl http://127.0.0.1:8010/health
# 3. Run the pre-render pipeline (10 clips, ~46 s wall-time):
python scripts/render_local_answers.py
# Use --force to re-render after changing ELEVENLABS_VOICE_ID.
```

The bridge clips (compliment / question / objection intent substrates) are uploaded to the pod separately:

```bash
bash phase0/scripts/upload_bridge_clips.sh
```

### Smoke test

With backend + dashboard + tunnel running:

```bash
# Local Gemma + pre-rendered MP4 path — Gemma classifies + matches Q&A + plays clip in <1s
curl -F "text=is it real leather" http://127.0.0.1:8000/api/comment

# Escalate path — Gemma classifies + drafts + Wav2Lip stitches onto pre-rendered substrate
curl -F "text=how does this compare to the Apple Watch" http://127.0.0.1:8000/api/comment

# Router unit tests (19 parametrized cases)
pytest backend/tests/test_router.py -v
```

---

## What we built and what we didn't

- **Built and shipped:** Gemma-driven pitch generation, Gemma comment classifier + router, on-device Whisper transcription, two-tier Director with pre-rendered clip choreography, Wav2Lip lip-sync overlay onto intent-aware substrates, pre-rendered local Q&A library, multilingual TTS path (6 languages today, architecture supports 140+), BRAIN telemetry panel showing local-vs-cloud routing rate + cost saved.
- **Out of scope for the demo:** real TikTok Shop / Instagram Live integration (the chat panel simulates comments, the router itself is identical to what would feed a real platform integration). CLOSER agent (DM auto-responder). Generative photo variants. Multi-stream isolation.

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) §"Not in scope today" for the full honest list.
