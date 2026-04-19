# Comment API — reference for the test page

Everything Cody needs to build `/dev/comments` without re-running the sell
pipeline. Source of truth lives in `backend/main.py` and `backend/agents/router.py`;
this doc stays in sync by hand — if you change the contract, update this file.

---

## 1. The HTTP entry point

```
POST /api/comment
Content-Type: application/x-www-form-urlencoded
Body: text=<comment string>

→ 200 {"status": "processing"}
```

Fire-and-forget. The actual response streams over the dashboard WebSocket.
Source: `backend/main.py:1032` → kicks `asyncio.ensure_future(run_routed_comment(text))`.

## 2. Dispatch flow

`run_routed_comment` (main.py:1991) is the spine:

```
text  ──►  classify_comment_gemma(text)          # eyes.py:185 — on-device Gemma
       ├►  comment_router.decide(text, classify, product)   # router.py
       ├►  WS broadcast {type: "routing_decision", ...}     # for the panel
       └►  dispatch into one of 4 _run_* helpers (below)
```

Classifier returns `{type, source, ...}` — never raises. `decide()` returns
`{tool, args, reason, ms, was_local, cost_saved_usd}`.

## 3. The 4 tools (Cody's picker)

From `backend/agents/router.py:50`:

| tool | when to pick | args |
|---|---|---|
| `respond_locally` | routine product Q&A the seller pre-authored (price/shipping/returns/sizing/warranty/etc.) | `{answer_id: <key from product.qa_index>}` |
| `play_canned_clip` | compliment / objection / neutral acknowledgement, no new generation | `{label: "compliment" \| "objection" \| "neutral"}` |
| `escalate_to_cloud` | cross-product compares, opinions, anything needing live reasoning | `{comment: <original text>}` |
| `block_comment` | spam, abuse, off-topic promo — no visual response | `{reason: <short tag>}` |

Cost model: every non-escalating decision saves ~$0.00035 (Bedrock Claude Haiku averaged). `escalate_to_cloud` saves $0.

There is **no `pitch_product` tool** by design — the 30s pitch only fires from the video-upload pipeline. Comments are audience reactions only.

## 4. WebSocket events Cody subscribes to

Connect: `ws://localhost:8000/ws/dashboard`

### Events fired during a comment run

| `type` | payload | source |
|---|---|---|
| `routing_decision` | `{comment, tool, reason, ms, was_local, cost_saved_usd}` | main.py:2012 |
| `comment_failed` | `{comment, response, reason}` | main.py:2063 — only on dispatch error |
| `clip_emit` (via Director) | `{layer, intent, url, mode, fade_ms, ...}` | avatar_director.py emit() |

### Events fired by the sell pipeline (for the post-pitch fixture state)

| `type` | payload | when |
|---|---|---|
| `pipeline_step` | `{request_id, step, status}` | each phase of intake |
| `transcript` | `{text}` | after Deepgram |
| `phone_frame` | `{frame: <truncated b64>}` | after frame extraction |
| `transcript_extract` | `{data: {...}}` | only if `EMPIRE_TRANSCRIPT_EXTRACT=1` |
| `product_data` | `{data: <product dict>}` | after Gemma analysis |
| `sales_script` | `{script: <pitch text>}` | after script generation |
| `product_photo` | `{photo: <b64>}` | after rembg |
| `view3d` | `{...}` (heroes/raw_heroes/spin URL) | after threed agent |
| `status` | `{status: "idle" \| "creating" \| "selling" \| "live"}` | phase changes |
| `force_phase` | `{phase: "INTRO" \| "BRIDGE" \| "PITCH" \| "LIVE", status}` | operator override |
| `on_air` | `{on: bool}` | toggle |

For mocking the post-pitch state without running the pipeline, replay
these events in order from `pipeline_run_log.txt` (or just hydrate from
`post_pitch_state.json` and skip the replay).

## 5. respond_locally — the qa_index

Each product in `backend/data/products.json` has a `qa_index`:

```json
{
  "is_it_real_leather": {
    "keywords": ["real leather", "genuine", "material", "leather"],
    "text": "Yes — full-grain vegetable-tanned leather. The real thing.",
    "url": "/local_answers/wallet_real_leather.mp4"
  },
  "..."
}
```

Router matches comment text against `keywords[]` (substring match) to
pick `answer_id`. Cody's panel should expose this so he can verify the
match logic from the page.

When the dispatcher picks `respond_locally`, the Director crossfades to
the `url` clip directly — no Wav2Lip render, no cloud round-trip.
Sub-300ms in the warm path.

## 6. play_canned_clip — the bridge picker

Currently the helper picks a random clip from the matching intent
bucket. See `bridge_library.json` for the full inventory. Cody's
FunctionGemma picker should replace random selection with intent-aware
choice (e.g., "compliment_C feels warmer for first-time viewers,
compliment_E for repeat").

## 7. escalate_to_cloud — the cloud path

Forwards to `api_respond_to_comment` (existing endpoint) which runs
Claude → ElevenLabs → Wav2Lip → Director crossfade. Same render path
as the pitch, just driven by comment text instead of script. ~5-10s.

## 8. block_comment — silent

No clip emit. Increments a counter for the routing panel and logs
a `routing_decision` WS event. Done.

---

## Test it from the CLI

```bash
# Fire one comment
curl -X POST http://localhost:8000/api/comment -d 'text=is this real leather'

# Watch WS in another terminal
websocat ws://localhost:8000/ws/dashboard | jq -c '.type, .'
```

The `respond_locally` tool should fire (`is_it_real_leather` keyword match
on the leather wallet's qa_index) and play `/local_answers/wallet_real_leather.mp4`.
