# PITCH RUNBOOK — EMPIRE on stage

Single document. Read top-to-bottom at **T-30 min**. Don't improvise.

Context: Cactus × Google DeepMind × YC Gemma 4 Voice Agents Hackathon, 2026-04-19. Cody is the operator. 3-5 minute live demo.

---

## T-30 min — network + Mac setup

```bash
cd ~/Desktop/Zo
ipconfig getifaddr en0
```

Note the IP. Options:
- **iPhone hotspot** (recommended): iPhone → Settings → Personal Hotspot ON. Mac joins the hotspot. `ipconfig getifaddr en0` shows `172.20.10.x`.
- **Shared WiFi**: both on same SSID, confirm no client isolation.

If the IP differs from last build → **long-press iPhone status pill** (or GEMMA card if visible) → enter new IP → Save. **No Swift rebuild needed** — P0.2 shipped this runtime override. Clear the host-override ("Reset to default") only if going back to a previously-baked IP.

Also confirm: Mac plugged into power, Mac sleep disabled (System Settings → Lock Screen → Prevent automatic locking).

## T-25 min — cold boot the stack

```bash
cd ~/Desktop/Zo && make bootstrap
```

All green. If any red, fix before proceeding — do NOT walk on with a failing bootstrap.

## T-20 min — start backend + dashboard

```bash
make start
```

Wait for these log lines (takes ~30-60s):
- `EMPIRE backend running on ...`
- `Cactus/Gemma 4 model ready` (Cactus cold-load)
- `Uvicorn running on http://0.0.0.0:8000`

Dashboard auto-opens in browser at http://localhost:5173.

## T-18 min — stage gate

```bash
bash scripts/demo_prewarm.sh --stage
```

Must exit 0. Expected output today:
- `PASS ≥ 12` — backend + active product + classify + Bedrock + ElevenLabs + BRAIN prefire
- `WARN = 0-1` — `bridge clips manifest empty` is acceptable (not used in iPhone pitch)
- `FAIL = 0` — **any FAIL is a stop-ship until resolved**

Known issue: if AWS creds in `.env` are invalid, Bedrock check FAILs with `UnrecognizedClientException`. Two paths:
1. Fix `.env` with valid AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY before stage
2. Accept the failure and tell the pure-on-device story: every demo question hits the local Q/A index; no cloud needed. Honest narrative.

BRAIN prefire seeds 20 curated comments; `/api/brain/stats.total` should be ≥ 25 after this step.

## T-15 min — dashboard smoke

Open http://localhost:5173 fullscreen (press F). Verify:

- **Top header**: `EMPIRE` logo, `◎ Telemetry` button with event count badge (≥20)
- **Controls row**: `PRODUCT` dropdown (2 products: wallet + backpack) — **leave on wallet** for demo
- **Cinema grid — sideCol (3 rows)**:
  - ProductPanel: wallet photo + product info
  - **CreatorPanel**: "▶ BUILD 3 PHOTOS + 15s PROMO" button
  - ChatPanel: comment input
- **Click `◎ Telemetry`** → overlay shows Routing, BRAIN (total ≥ 25), AgentLog. Close overlay.

## T-12 min — iPhone setup

1. Settings → Privacy & Security → **Local Network** → EmpirePhone **ON**. If the toggle isn't there → delete + reinstall the app to force the permission prompt.
2. Open **EmpirePhone**. Wait for status pill to go green ("READY") — Cactus whisper-base cold-loads in ~2s.
3. **Long-press status pill** → verify the host field matches the Mac IP from T-30. If not → update.
4. Test: hold-to-speak "hi", release. Should see WHISPER transcript card within ~350ms.

## T-10 min — fire a dry run

```
iPhone: hold-to-speak: "Is this real leather?"
```

Expected in <1s:
- WHISPER card: `"is this real leather?" · ~350ms`
- ROUTER card (green): `respond_locally · Matched product.qa_index key: is_it_real_leather · 1ms`
- MP4 plays (wallet_real_leather.mp4): avatar speaks the answer

Expected 2-3s later:
- GEMMA card lights magenta: `question · cactus · ~2500ms`

If anything fails → iterate. If still failing at T-7, go to backup video path.

## T-7 min — rehearse 3 phrases from the validated set

Pull 3 phrases from `docs/REHEARSAL_PHRASES.md` that scored ≥9/10 during reps. Fire each on the iPhone. All 3 must hit `respond_locally` + play the right MP4 + GEMMA card lights magenta.

**Rule: any phrase that fails now gets replaced from the backup bank, not retried.** Stage isn't the time to gamble.

## T-5 min — final operator setup

- **Terminal 1** visible on the second monitor (if multi-monitor): tailing backend logs (`tail -f ~/Desktop/Zo/logs/*.log` or whatever `make start` writes to). Helps narrate "look, it's on device."
- **Terminal 2** ready with a fallback `bash scripts/demo_prewarm.sh` command in history (in case something dies mid-pitch).
- **iPhone** charger plugged in. Screen brightness at max. Do Not Disturb ON.
- **Dashboard** on stage monitor, fullscreen.

## T-2 min — insurance

```bash
ls -la ~/Desktop/EMPIRE_DEMO_BACKUP.mp4
```

Confirm it's there and plays (open in QuickTime, scrub through briefly).

---

## STAGE — walk-on sequence (3-5 min)

1. **Opener beat** (~20s): click **▶ BUILD 3 PHOTOS + 15s PROMO** on the dashboard. "Watch the laptop generate product marketing in 3 seconds." Photos + promo appear.
2. **Local routing beat** (~60s): pick up iPhone, hold-to-speak phrase 1 from the validated set. WHISPER → ROUTER → MP4 plays. Narrate: "Whisper on device. Router on device. Answer pre-rendered, plays instantly. No network call."
3. **Gemma verification beat** (~30s): while MP4 is playing, point at the magenta GEMMA 4 card. "And while that's happening, the Mac's Gemma 4 is also classifying — on-device LLM, zero cloud. They agree."
4. **Moat beat** (~30s): open `◎ Telemetry`. BRAIN panel. "Every comment we've routed so far — 25+ events, 100% local, zero cloud spend. That's the moat: on-device inference at consumer scale."
5. **Multi-product beat** (~15s): mention the PRODUCT dropdown. "EMPIRE isn't one-SKU — every product a seller lists becomes a qa_index." Don't actually switch; backpack has no rendered MP4s.
6. **Close** (~30s): "4 of 5 agents shipped on a real iPhone today. CLOSER is next sprint. Everything you saw ran off-cloud."

## FAILURE RECOVERY

| Symptom | Fix |
|---|---|
| iPhone "Mac unreachable" | Long-press status pill → re-enter IP → save. No rebuild needed. |
| Backend not responding | Second Terminal: `bash scripts/demo_prewarm.sh` → find the FAIL. |
| Whisper mis-transcribes on stage | Say phrase 2 of the validated set instead. If persistent, pivot to backup video. |
| CreatorPanel times out | Skip that beat — go straight to the iPhone-driven flow. |
| **Any uncontained failure** | **Play `~/Desktop/EMPIRE_DEMO_BACKUP.mp4`** on the stage laptop. Narrate over it in real time. Don't apologize — own it. |

## POST-STAGE

```bash
# Dump BRAIN for the post-mortem
curl -s http://localhost:8000/api/brain/stats | python3 -m json.tool > ~/Desktop/stage_brain_$(date +%s).json
```

Keep the terminal output open until you're off stage. Screenshot `◎ Telemetry` for the follow-up thread.
