#!/usr/bin/env bash
# scripts/prefire_brain.sh — seed the BRAIN panel before stage.
#
# Fires 20 canned comments at /api/comment so the dashboard's BRAIN panel
# shows real stats (total, by_tool, top_answers) from the moment the
# operator opens it, instead of the empty "fire a comment to begin" state.
#
# Pick is intentional — every comment routes to respond_locally,
# play_canned_clip, or block_comment (all zero-cost beyond Gemma classify).
# Zero escalate_to_cloud calls, so prefire spends $0 on Bedrock/ElevenLabs/
# Wav2Lip. The live on-stage demo populates top_misses organically as real
# audience questions come in, which is the whole point.
#
# Usage:
#   ./scripts/prefire_brain.sh
#
# Env overrides:
#   BACKEND_URL        default http://localhost:8000
#   PREFIRE_SLEEP_MS   default 150 — delay between posts to let Gemma settle.
#                      Cactus Gemma 4 classify is lock-serialized; sending
#                      faster than ~10/sec queues requests in uvicorn.

set -uo pipefail
cd "$(dirname "$0")/.."

BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"
SLEEP_MS="${PREFIRE_SLEEP_MS:-150}"

# 15 product questions — each hits a wallet qa_index keyword.
# 3 compliments + 1 objection → play_canned_clip via cue words.
# 1 URL spam → block_comment via _SPAM_CUES.
COMMENTS=(
  "how much does it cost"
  "what's the price"
  "is it real leather"
  "is this genuine leather"
  "what material is it"
  "does it ship to canada"
  "when will it arrive"
  "can i get a refund"
  "what's your return policy"
  "how long is the warranty"
  "what colors does it come in"
  "how big is it"
  "does it have rfid blocking"
  "is it waterproof"
  "how many cards does it hold"
  "this is so beautiful"
  "love the design"
  "amazing work"
  "this looks too expensive"
  "check out my store at https://shady.example"
)

# Baseline BRAIN total so we can report delta at the end — lets the operator
# distinguish "20 new events" from "20 events already there."
start_total=$(curl -fsS "$BACKEND_URL/api/brain/stats" --max-time 3 2>/dev/null \
  | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("total",0))
except Exception: print(0)' 2>/dev/null || echo 0)

echo "prefire_brain: seeding ${#COMMENTS[@]} comments at $BACKEND_URL (start total=$start_total)"

fired=0
failed=0
for c in "${COMMENTS[@]}"; do
  if curl -fsS -X POST "$BACKEND_URL/api/comment" \
       -F "text=$c" --max-time 10 >/dev/null 2>&1; then
    fired=$((fired+1))
    printf "  · %s\n" "$c"
  else
    failed=$((failed+1))
    printf "  ✗ failed: %s\n" "$c"
  fi
  # Sleep between posts — Gemma classify is lock-serialized; machine-gunning
  # just queues requests in uvicorn and doesn't meaningfully speed things up.
  if [ "$SLEEP_MS" -gt 0 ]; then
    python3 -c "import time; time.sleep($SLEEP_MS/1000.0)"
  fi
done

# /api/comment is fire-and-forget (run_routed_comment is asyncio.ensure_future),
# and classify_comment_gemma is lock-serialized around Cactus Gemma 4 — so
# firing 20 posts enqueues 20 classify tasks that run ~2.5s each = ~50s
# wall-clock before BRAIN is fully populated. Poll /api/brain/stats until
# the delta lands, with a generous timeout so we surface a Cactus stall
# (e.g., model wasn't loaded) rather than silently timing out.
target=$((start_total + fired))
deadline=$(($(date +%s) + 90))
final_total=$start_total
final_local=0
final_pct=0
while true; do
  stats_out=$(curl -fsS "$BACKEND_URL/api/brain/stats" --max-time 3 2>/dev/null \
    | python3 -c 'import sys,json
try:
    d = json.load(sys.stdin)
    print(d.get("total", 0), d.get("local_n", 0), d.get("pct_local", 0))
except Exception:
    print("0 0 0")' 2>/dev/null || echo "0 0 0")
  read -r final_total final_local final_pct <<< "$stats_out"
  if [ "$final_total" -ge "$target" ]; then
    break
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "prefire_brain: timeout — only $((final_total - start_total)) of $fired events landed in 90s (Cactus Gemma 4 stalled?)"
    break
  fi
  sleep 2
done

delta=$((final_total - start_total))

echo ""
echo "prefire_brain: fired=$fired failed=$failed delta=$delta (total=$final_total, local=$final_local, pct_local=${final_pct}%)"

# Exit 1 if we couldn't seed at least 20 new events — demo_prewarm.sh --stage
# relies on this to fail-fast before walking on stage.
if [ "$delta" -lt 20 ]; then
  echo "prefire_brain: FAIL — expected delta ≥ 20, got $delta (fired=$fired failed=$failed)"
  exit 1
fi

echo "prefire_brain: OK"
exit 0
