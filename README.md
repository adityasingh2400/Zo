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

### The two-tier stage

The dashboard runs two crossfading video layers. Both are always playing pre-rendered MP4s. Nothing on screen is ever generated frame-by-frame at runtime.

```
TIER 0  ──  always-on looping idle pose          (pre-rendered, muted, never stops)
            └─ idle_calm / idle_thinking / hair_touch / reading_comments — Veo-rendered library

TIER 1  ──  reactive overlay, crossfaded in       (pre-rendered, with audio)
            └─ pitch clip (LatentSync, 50s render, played once per product)
            └─ Q&A response clip (pre-rendered MP4 from the local answer cache)
            └─ Intent bridge clip (pre-rendered "compliment / question / objection" pose,
               with fresh TTS audio lip-synced over it)
            └─ Sip / glance / walk-off interjections (pre-rendered, muted, ambient)
```

**The "live avatar" is Gemma 4 deciding which pre-rendered clip to crossfade onto Tier 1, and what audio (if any) to lip-sync on top of it.** That's it.

### Where Gemma applies to the stitching — explicitly

This is the apparent loop the judges should see:

```
VIEWER COMMENT
  │
  ▼
Gemma.classify_comment ──►  type: "compliment"        ──┐
                            draft_response: "Aww thank │
                            you, I love that you said  │
                            that — it took me forever  │  Gemma's classification IS the
                            to find this leather"       │  routing signal. The router is
                                                        │  rule-based on TOP of Gemma's
                                                        │  output (0 ms decision, 100%
                                                        │  accurate on the test suite).
Router.decide(...)  ──────────────────────────────────►─┘
  │
  ▼
DIRECTOR.choreograph
  │
  ├─ Gemma said "compliment" → pick a clip from the COMPLIMENT bucket of pre-rendered substrates
  │  (warm-smile + nod gesture, 8s, already on the GPU pod)
  │
  ├─ Gemma's draft_response → ElevenLabs TTS (or one of 6 supported languages,
  │  cached) → ~5s of new audio bytes
  │
  ├─ Wav2Lip lip-syncs the new audio ONTO the pre-rendered compliment substrate
  │  (warm cache: ~1.5–2s on the 5090). The substrate's body language is preserved;
  │  only the mouth region is regenerated to match the new words.
  │
  └─ Director crossfades the lip-synced clip onto Tier 1 over the always-playing
     Tier 0 idle. After the audio ends, fade Tier 1 back out and the idle returns.
```

If Gemma classifies as `question` AND the comment matches a key in the seller's pre-authored `qa_index`, the router skips the lip-sync round trip entirely and plays a **fully pre-rendered MP4 answer** in **<1 second total**. That MP4 was rendered once, at product onboarding time, and lives on disk forever. **Gemma's job there is to recognize the question well enough to fire the right clip.**

### What's pre-rendered, where, and what it costs

| Asset | Quantity | When rendered | Where it lives | Cost |
|---|---|---|---|---|
| Idle pose loops (Tier 0) | 5–8 per avatar | Once, via Vertex AI Veo | `/states/idle/*.mp4` | Generated offline |
| Speaking-pose substrates | 1 per idle pose | Once, paired with each idle | Pod `/workspace/idle_speaking/` | Generated offline |
| Intent bridge clips (compliment/question/objection) | 3–5 per intent | Once, via Veo | Pod `/workspace/bridges/<intent>/` | Generated offline |
| Local Q&A answer MP4s | 10–20 per product | Once, at onboarding | `backend/local_answers/<slug>.mp4` | ~4.6 s/clip warm. 10 clips in ~46 s wall-time. ~50 MB on disk per product. |
| Pitch clip (per product) | 1 per product | Once, via LatentSync 1.6 | Pod render | ~50 s render, played once per stream |
| Walk-off / sip / glance interjections | A few each | Once, via Veo | `/states/idle/misc_*.mp4` | Generated offline |

**At runtime there is no video synthesis.** Wav2Lip is a *lip-region overlay* — it takes a pre-existing video and modifies the mouth pixels to match new audio. It is not generating a face. It is not generating a body. The pose, the lighting, the eyes, the hands — all pre-rendered. Mouth pixels only.

### Why this matters for what Gemma can do

A judge looking at the demo will reasonably ask: *"How can a 4B-parameter on-device model produce a live, reacting AI avatar?"* The answer is that **it doesn't have to**. Gemma's job is the part Gemma is genuinely good at — **understanding language and routing decisions in <500 ms** — and the pre-rendered clip library handles everything else. We didn't pick this architecture to hide a limitation. We picked it because it's the only architecture where the unit economics close at $144/mo.

Live generative video on-device is currently impossible at acceptable quality and latency on consumer hardware. **Pre-rendered clips + Gemma routing is the production architecture for AI video commerce in 2026.** This is what HeyGen, Khaby Lame's avatar deal, and TikTok's AI Seller Assistant will all converge to within 12 months.

---

## How a single comment flows — end to end

```
t=0       Viewer types: "is it real leather"
t=0.05    FastAPI /api/comment receives it
t=0.20    Gemma.classify_comment_gemma → {type: "question"}      [on-device, ~150 ms]
t=0.20    Router checks product.qa_index → MATCH on "is_it_real_leather"
t=0.20    Router emits {tool: "respond_locally", answer_id: "is_it_real_leather"}
t=0.20    Director.play_response("/local_answers/wallet_real_leather.mp4")
t=0.55    Avatar is ON SCREEN speaking the answer.

  Total: < 600 ms. Zero cloud spend. Zero render. The clip was rendered once,
  weeks ago, when the seller authored that Q&A entry.
```

Compare a comment that doesn't match the local index — Gemma still drives every step, it just stitches a fresh lip-sync onto a pre-rendered intent substrate:

```
t=0       Viewer: "how does this compare to the Apple Watch"
t=0.20    Gemma.classify_comment_gemma → {type: "question",
                                          draft_response: "Different category — this is..."}
          (Note: ONE Gemma call returns both the routing intent AND the words to speak.)
t=0.20    Router → escalate_to_cloud (no local Q&A match)
t=0.20    Director.emit_reading_chat() — Tier 1 swaps to "reading comments" pose, looped
t=0.50    ElevenLabs TTS on Gemma's draft_response → audio bytes
t=0.50    Wav2Lip /lipsync_fast: takes a pre-rendered "question" intent substrate
          (8 s clip, thoughtful-nod pose) + the new audio → returns lip-synced MP4
t=2.0     Director.play_response(lipsynced_url) — crossfade onto Tier 1
t=2.0     Audience sees the avatar visibly nodding + speaking the response in
          intent-coherent body language. Mouth alignment is Wav2Lip-soft, body
          is Veo-quality.
```

The "live AI" effect comes from three things stitched together:
1. **Gemma's classification picks the right substrate** (warm smile vs. thoughtful nod vs. "actually..." beat).
2. **Gemma's draft_response is the audio**, with no second LLM call — the same forward pass that classifies *also drafts*.
3. **Director's two-tier crossfade hides the seam** between idle, reading, and response.

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
