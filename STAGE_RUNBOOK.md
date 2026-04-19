# STAGE RUNBOOK

Solo operator playbook. From "5 minutes to stage" through "off stage". When
something breaks, ctrl-F for the symptom in the [Failure modes](#failure-modes)
section instead of debugging.

If you're reading this 30 seconds before going on stage: jump to
[Showtime sequence](#showtime-sequence) and do the steps in order.

---

## URLs (single source of truth)

| What | URL | Notes |
| --- | --- | --- |
| **Stage view** | `http://localhost:5173/stage` | Press `F` for fullscreen, `G` to fire intro, `R` to reset cost ticker |
| **Operator dashboard** | `http://localhost:5173/` | Drag-drop the judge's item video here |
| **Audience comment form** | `https://<random>.trycloudflare.com/comment` | Printed by `start_audience_tunnel.sh`. QR PNG at `/tmp/empire_audience_qr.png` |
| **Backend health** | `http://localhost:8000/api/state` | JSON dump; sanity check |
| **Backend API root** | `http://localhost:8000` | FastAPI |

If any of those URLs change, update this section first, fix everything else
second.

## Hotkeys (single source of truth)

All bound to the window — they work even when no field has focus.

| Key | Where | What it does |
| --- | --- | --- |
| `G` | `/stage` | Fire intro clip (POST `/api/go_live`); shows `INTRO FIRED` ping for 1.4s |
| `F` | `/stage` | Toggle browser fullscreen |
| `R` | anywhere on dashboard | Reset cost ticker to `$0.00000` |
| `Esc` | `/` | Close telemetry overlay |

`Cmd-F` / `Ctrl-F` are explicitly NOT intercepted — browser find still works.

---

## Pre-show checklist (T-15 min)

Run these in order. If any FAILs, see [Failure modes](#failure-modes) before
walking on stage.

### 1. Backend up

```bash
cd backend
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --ws-max-size 67108864
```

Watch for the startup banner. Required env in `backend/.env`:

```
ELEVENLABS_API_KEY=...           # cloud TTS
ELEVENLABS_VOICE_ID=...          # the seller's voice
AWS_ACCESS_KEY_ID=...            # Bedrock Claude
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-3-5-haiku-20241022-v1:0
LIPSYNC_HOST=127.0.0.1           # only needed if using local lipsync
WAV2LIP_PORT=8010
GEMINI_API_KEY=...               # only if using cloud Gemma
```

`AWS_*` and `ELEVENLABS_*` are mandatory for cloud-escalate to work. The
others are nice-to-have.

### 2. Dashboard up

```bash
cd dashboard
npm run dev
```

Wait for `Local: http://localhost:5173/`. Open `http://localhost:5173/stage`
in a clean browser window (separate from the operator dashboard).

### 3. Prewarm

```bash
./scripts/demo_prewarm.sh
```

Should print `DEMO READY` (green). WARNs are OK if you understand what they
mean — read each one. FAILs are stop-the-world.

For a fuller test that exercises the cloud path end-to-end (~10s, costs
$0.00035):

```bash
./scripts/demo_prewarm.sh --with-cloud
```

### 4. Audience tunnel + QR

In a third terminal:

```bash
./scripts/start_audience_tunnel.sh
```

It prints:
- The trycloudflare URL (drop into intro slide)
- A QR PNG at `/tmp/empire_audience_qr.png` (drop into intro slide; print
  + tape to back of laptop as backup)
- A terminal QR (you can scan it from your phone right now to test)

Leave this terminal running. Ctrl-C tears down the tunnel.

### 5. Test one comment from your phone

Scan the QR with your phone, send "test ignore me". On `/stage` you should
see:
- `audience_comment` toast
- Avatar transitions to `idle_reading_comments` (visibly reading)
- Within 3-8s, audio plays + KaraokeCaptions render the answer
- Cost ticker increments by `$0.00035` (one routing decision)

If any of those four things doesn't happen, jump to
[Failure modes](#failure-modes) — do NOT walk on stage with a broken
audience loop.

### 6. Cosmetic prep

- Press `R` on `/stage` → cost ticker back to `$0.00000`
- Press `F` on `/stage` → fullscreen
- Reload `/stage` once → fresh state, no leftover comments
- Make sure `idle_calm` (or another idle Tier 0) is playing — face is
  visible and not stuck on a previous response

---

## Showtime sequence

Roughly the 2-minute beat sheet from the design doc. Numbers are
approximate; let the audience pace it.

### 0:00 — Walk on stage. `/stage` already fullscreen.

The avatar is in idle. Cost ticker reads `$0.00000`.

Open with the hook:
> "Live commerce is a $68 billion market. The barrier to entry is hiring
> a host team — about $9,000 a month. We just made it free."

### 0:15 — Press `G` to fire intro

Avatar plays the intro bridge clip. While she's talking:
> "Meet Zo — your 24/7 AI livestream seller. Multilingual, sub-eight-second
> response, voice in, video out. Watch."

### 0:35 — Take the judge's item

Walk down, hand them your phone. (Or: pre-recorded video of you walking
down to take the item from the judge — depends on what your friend has
shipped on the mobile app.)

For now (drag-drop fallback): take their item, photograph or screen-record
it on your phone, AirDrop to the demo laptop. Drop into the operator
dashboard at `http://localhost:5173/` (the home page).

The pipeline kicks off automatically:
- Gemma vision → product_data
- Claude → 4-beat script
- ElevenLabs → TTS
- Wav2Lip → lipsync (cloud, ~5-10s)
- Avatar plays the pitch

### 1:00 — Audience interaction

While the pitch finishes, point at the QR on screen / your laptop:
> "QR's been live the whole time. Comment now — Zo answers in real time,
> on her phone, on stage, in front of you."

You'll get 3-15 comments instantly. The router classifies each — most
will route through cloud-escalate (because product_data is
judge-item-specific, no qa_index hits). Each takes 5-10s to render and
play.

### 1:30 — Cost ticker kill shot

After 3-5 comments have processed, point at the ticker:
> "Five comments answered. That cost three quarters of a cent. A human
> host team would have charged us $375 for that hour."

### 1:45 — Closing line (memorize this verbatim)

> "Zo replaces a $9,000-a-month live host team with a Mac and thirty-five
> hundredths of a cent per question. TikTok Shop is a $68 billion market.
> Two hundred million small businesses can't afford to be on it.
> We just made it free."

Walk off.

---

## Failure modes

### "I pressed G and nothing happened"

- Backend not running, or backend is running but no bridge clips rendered
  (no intro clip to play).
- Check terminal: did POST `/api/go_live` return 200 or 503?
- 503 = no clip available. Run
  `python -m backend.agents.bridge_clips render` (slow, LatentSync) or
  `python scripts/render_generic_clips.py` (fast, Wav2Lip).
- 200 but visually nothing = browser tab lost focus or `/stage` socket
  disconnected. Reload `/stage`, try again.

### "Comment came in, avatar never reacted"

- Open backend terminal. Look for `[trace=...]` entries near the time the
  comment arrived. There should be a `phase: classify` then `phase: respond`.
- If you see `phase: classify` but no `phase: respond`: classify (Cactus)
  is still grinding. CPU, can be 5-10s. Audience will see
  `idle_reading_comments` visibly hold for that whole window — designed
  behavior, NOT broken.
- If you see no trace at all: the comment never reached the backend.
  Check `start_audience_tunnel.sh` terminal — is the tunnel still up?
  trycloudflare URLs can drop after ~30 min idle. Restart the tunnel and
  REGENERATE THE QR (the URL will change!).

### "Avatar's lips don't move on cloud responses"

- Expected, by design. Cloud-escalated comments play standalone audio +
  KaraokeCaptions over a looping speaking-pose idle clip. There's no live
  Wav2Lip on the cloud-escalate path (it was removed for quality reasons —
  Wav2Lip's 96px patches looked artifacted). The avatar visibly mouths
  through the speaking-pose loop, which reads as "talking" from a
  distance.
- If you NEED real lipsync for a specific question, pre-render it as a
  local answer for whatever product is loaded (see
  `scripts/render_local_answers.py`).

### "TTS is silent / audience hears nothing"

- ElevenLabs API down or out of credits. Check
  `https://api.elevenlabs.io/v1/user` with your key.
- Backend logs: search for `[TTS]` or `text_to_speech`. You'll see a stack
  trace if it's failing.
- Recovery: the dashboard already shows the response text via
  KaraokeCaptions even when TTS fails. You can verbally read the captions
  yourself. Awkward but recoverable.

### "Cost ticker is stuck / showing wrong amount"

- Press `R` (with `/stage` focused) to reset to `$0.00000`. Each
  `routing_decision` increments by `$0.00035`.
- If R doesn't work, the keyboard handler in `CostTicker.jsx` may not be
  bound — reload `/stage`.

### "Two audio streams playing at once"

- Should be fixed (see commits around the audio fade rAF clamp). If it
  recurs: open browser devtools console, look for the `[dlog] tier1` and
  `[dlog] audio` events. Send screenshots after the demo to whoever's
  on call.
- Mid-demo recovery: reload `/stage`. You'll lose any in-flight comment
  state but the next comment will render cleanly.

### "Pending comment chip stuck on screen"

- Cloud-escalate path emits a synthetic `comment_response_video`
  (`url=null`) specifically to clear pending chips. If a chip is stuck,
  the synthetic event didn't broadcast — usually means TTS failed AND
  the fallback `comment_failed` broadcast also failed.
- Manual recovery: reload `/stage`. Pending state lives entirely
  client-side, so a reload clears it.

### "Wrong product name in BUY card"

- `backend/data/products.json` should be `{}` for judge-item demos. If
  it's not empty, that product loads on backend boot and shows up in the
  BUY card. Empty it: `echo "{}" > backend/data/products.json`.

### "Cloudflared tunnel died mid-demo"

- The tunnel terminal will show an error. Ctrl-C, restart it.
- The new URL will be DIFFERENT. Generate a new QR.
- If you've lost the audience: go straight to the cost-ticker beat. The
  judge-item pitch is the kill shot, not the audience comments.

### "RunPod tunnel down (cloud Wav2Lip unreachable)"

- Doesn't matter for the cloud-escalate path (it doesn't use Wav2Lip
  anymore). Only matters if you're trying to render a live pitch with
  lipsync.
- For the demo: skip lipsync, the pitch will still play with audio +
  speaking idle.
- Backend will log a single-line `httpx.ConnectError` warning per
  attempt. Not a crash.

---

## Post-show

```bash
# Stop the tunnel
# In its terminal: Ctrl-C

# Reset state for the next rehearsal / demo
echo "{}" > backend/data/products.json
rm backend/renders/resp_*.mp4    # purge stale renders (gitignored)
git status                        # should be clean
```

If you committed mid-demo for some reason, rotate the build forward:

```bash
git push                          # to zo/main
```

(That's literally the entire push workflow — see SHIP.md.)

---

## When to update this file

- Any time you add a new hotkey
- Any time you change a port or URL
- Any time you discover a new failure mode mid-rehearsal — write it down
  in [Failure modes](#failure-modes) before you forget

This document is the contract between "what's actually wired up" and
"what the operator thinks is wired up". If those drift, demos break on
stage.
