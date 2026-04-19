# Cody — comment interactions sandbox

You own the **comment / live-reaction** layer end-to-end. Aditya owns the
intake → pitch path. The handoff between you is this `phase0/fixtures/`
folder: real captures from a real pipeline run, so you never have to drop
a video to start working.

## Your mission

Build `**/dev/comments`** — a developer page mounted in the dashboard
that lets you iterate on the comment routing + bridge picker without
running the full sell pipeline every time.

Pattern: same as the existing `/dev/transitions` tuning page. Probably
add the route in `dashboard/src/main.jsx` (or wherever the route table
lives) and put the component at `dashboard/src/dev/CommentTester.jsx`.

## What the page needs

1. **Hydrate from fixture, not live state.** Read
  `phase0/fixtures/post_pitch_state.json` (or fetch from a new
   `/api/dev/fixture/post_pitch` endpoint if you'd rather keep it
   server-side) so the page boots in a "post-pitch, ready for
   comments" state without any pipeline run.
2. **Comment input** — text box → `POST /api/comment` (form-encoded,
  `text=...`). See `COMMENT_API.md` §1.
3. **Routing decision panel.** Subscribe to `/ws/dashboard`, render the
  incoming `routing_decision` event: which tool fired, why, latency,
   cost saved, was-local flag.
4. **Avatar response preview.** Show the resulting clip URL — local
  answer mp4, bridge clip, or pitch render. Use the existing
   `<video>` rendering pattern from `TikTokShopOverlay.jsx`.
5. **Bridge picker iteration UI.** This is the meat. Load
  `bridge_library.json`. For each intent bucket, show the 6-12 clips
   side by side with thumbnails. Let yourself force-pick a specific
   variant for testing (debug override via querystring, or a dropdown
   per intent).
6. **FunctionGemma prompt editor.** The `play_canned_clip` tool today
  does random pick. The roadmap (`router.py:13`) has FunctionGemma on
   Gemma 4 doing intent-aware picks ("compliment_C feels warmer for
   first-time viewers"). Build a textarea where you can edit the
   FunctionGemma prompt and re-run dispatch with the edited prompt.
   Wire it through a new `/api/dev/comment_with_prompt` debug endpoint
   if needed, or add a prompt-override param to `/api/comment`.

## Read these first (in order)

1. `phase0/fixtures/COMMENT_API.md` — the contract you're building against.
2. `phase0/fixtures/post_pitch_state.json` — your hydration fixture.
3. `phase0/fixtures/bridge_library.json` — the picker's universe.
4. `backend/main.py:1991` (`run_routed_comment`) — the dispatcher.
5. `backend/agents/router.py` — the 4-tool definitions + cost model.
6. `backend/agents/eyes.py:185` (`classify_comment_gemma`) — the on-device classifier.

## Ownership boundaries — do not touch


| Path                                                          | Owner                | Why                            |
| ------------------------------------------------------------- | -------------------- | ------------------------------ |
| `backend/main.py:run_video_sell_pipeline / run_sell_pipeline` | Aditya               | intake + pitch generation      |
| `backend/agents/eyes.py:analyze_and_script_`*                 | Aditya               | Gemma vision + text-only paths |
| `backend/agents/threed.py`                                    | Aditya               | carousel + raw_heroes + Spin3D |
| `backend/agents/seller.py:render_pitch_*`                     | Aditya               | TTS + pitch Wav2Lip            |
| `dashboard/src/App.jsx`                                       | Aditya               | the main stage shell           |
| `dashboard/src/components/TikTokShopOverlay.jsx`              | Aditya               | live phone surface             |
| `phase0/assets/bridges/processing/`                           | Aditya               | the upload-bridge clip         |
| `ios/EmpirePhone/`, `phone-quickdrop/`                        | Cody (active branch) | iOS phone surface              |


You **may** touch:

- `backend/main.py:run_routed_comment` and `_run_`* dispatch helpers
- `backend/agents/router.py` — the picker logic, cost model, FunctionGemma prompt
- `backend/agents/eyes.py:classify_comment_gemma` — classifier improvements
- `backend/agents/seller.py:render_comment_response_wav2lip` — comment-render path (already has the retry loop Aditya added)
- new files under `dashboard/src/dev/` — your test page
- new files under `phase0/fixtures/` — additional captures you generate

## Open questions to flag, not solve

1. `**question/` bucket has 12 clips but no tool routes there.**
  Currently `play_canned_clip` accepts `compliment | objection | neutral`
   only. Should `question` become a 5th label, or do questions always
   route to `respond_locally` / `escalate_to_cloud`? The 12 clips are
   curated and ready — they want a home.
2. `**neutral` label has no dedicated bucket.** Today the picker would
  404 on `play_canned_clip(neutral)`. Should it borrow from compliment
   or objection? Or use `processing/processing.mp4`?
3. **Active product mismatch in fixture.** `post_pitch_state.json` shows
  `product_data` = the watch (just pitched) but `active_product_id` =
   `leather_wallet` (the catalog default with the live qa_index). Comments
   route through the catalog's qa_index, not `product_data`. Confirm
   with Aditya whether the watch should also become the active product
   after a sell-video run.

## Known bug — don't fix, Aditya owns

`backend/agents/avatar_director.py:121` has
`misc_glance_aside_speaking.mp4` in the `TIER1_INTERJECTIONS` pool with
weight 0.30. The clip has visible mouth movement but plays muted in idle
context, creating a silent-mouthing uncanny effect. Also: the Tier 1
rotation is **not** suppressed during the processing.mp4 bridge, so it
overlays/interrupts. Both are Aditya's to fix in the avatar_director
refactor. If you see weird overlay behavior in your test page, that's
why — not your code.

## How to start

```bash
# 1. Pull
git pull zo main

# 2. Confirm services are up (Aditya keeps these running)
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/api/state   # 200
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:5173/             # 200
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8010/health       # 200

# 3. Smoke-test the existing dispatcher
curl -X POST http://localhost:8000/api/comment -d 'text=is this real leather'
# expect: routing_decision WS event with tool=respond_locally,
#         clip_emit for /local_answers/wallet_real_leather.mp4

curl -X POST http://localhost:8000/api/comment -d 'text=this looks amazing'
# expect: routing_decision WS event with tool=play_canned_clip label=compliment

# 4. Build the page. Land per SHIP.md (one commit per logical chunk,
#    push to zo main directly). Tests must keep passing:
cd backend && PYTHONPATH=. venv/bin/python -m pytest tests/ -q
```

Holler in chat with progress / blockers / the answers to the three open
questions above.