#!/usr/bin/env bash
# Pre-warm the Wav2Lip pod's face-detect cache for every bridge clip.
#
# Why: the first /lipsync_fast call against a substrate runs
# face-detect on the entire 8 s source (~12-15 s on the pod's GPU) and
# THEN does the actual lip-sync render (~5-6 s). Subsequent calls hit
# the cache and only pay the ~5-6 s render cost. Without this script
# the operator's first comment of each intent (compliment / objection /
# question) takes 18-22 s — visibly painful and indistinguishable from
# a stuck pipeline.
#
# This is a one-shot AFTER upload_bridge_clips.sh — run once per pod
# session, takes ~6 min total (30 substrates × ~12 s each, sequential).
# Can be re-run safely; /prewarm is idempotent (returns cached entry
# instantly on second hit).
#
# Usage:
#   bash phase0/scripts/prewarm_bridge_caches.sh
#
# Requires the Wav2Lip SSH tunnel to be open at localhost:8010 (the
# keep_tunnels_alive.sh script handles that).

set -euo pipefail

WAV2LIP_URL="${WAV2LIP_URL:-http://127.0.0.1:8010}"
OUT_HEIGHT="${OUT_HEIGHT:-1920}"  # matches render_comment_response_wav2lip

# Health check first — fail loud if the tunnel is down.
if ! curl -s -m 4 -f "$WAV2LIP_URL/health" >/dev/null; then
  echo "ERROR: $WAV2LIP_URL/health unreachable. Is the SSH tunnel up?"
  echo "  bash phase0/scripts/keep_tunnels_alive.sh &"
  exit 1
fi

# Same intents + filenames the upload script pushed. Mirror of
# pick_intent_substrate's filesystem walk; if the directory tree
# changes, both scripts need to follow.
INTENTS=(compliment objection question intro)

TOTAL=0
WARMED=0
SKIPPED=0
FAILED=0
T_GLOBAL=$SECONDS

for intent in "${INTENTS[@]}"; do
  src_dir="phase0/assets/bridges/$intent"
  if [ ! -d "$src_dir" ]; then
    echo "  $intent — no source dir, skipping"
    continue
  fi
  for f in "$src_dir"/*.mp4; do
    [ -e "$f" ] || continue
    TOTAL=$((TOTAL + 1))
    name=$(basename "$f")
    pod_path="/workspace/bridges/$intent/$name"
    printf "  [%2d] %-40s " "$TOTAL" "$pod_path"
    t0=$SECONDS
    body="{\"path\":\"$pod_path\",\"out_height\":$OUT_HEIGHT}"
    # 90 s timeout per call — cold cache build is usually 12-15 s but
    # GPU contention spikes can push to 30+ s. --retry-connrefused
    # absorbs SSH-tunnel hiccups (the RENDER_LOCK serializes prewarm
    # calls on the pod side; rapid back-to-back POSTs can race the
    # tunnel's connection reuse). 250 ms inter-call sleep lets the
    # previous lock release before we knock again.
    resp=$(curl -s -m 90 --retry 2 --retry-connrefused --retry-delay 1 \
                -X POST "$WAV2LIP_URL/prewarm" \
                -H 'Content-Type: application/json' -d "$body" || echo "")
    elapsed=$((SECONDS - t0))
    if echo "$resp" | grep -q '"frames"'; then
      frames=$(echo "$resp" | sed -n 's/.*"frames":\([0-9]*\).*/\1/p')
      boxes=$(echo "$resp" | sed -n 's/.*"box_count":\([0-9]*\).*/\1/p')
      printf "OK   %3d frames %3d boxes  (%2ds)\n" "$frames" "$boxes" "$elapsed"
      WARMED=$((WARMED + 1))
    elif [ -z "$resp" ]; then
      printf "TIMEOUT (%ds)\n" "$elapsed"
      FAILED=$((FAILED + 1))
    elif echo "$resp" | grep -q '"detail"'; then
      printf "ERROR  %s\n" "$resp"
      FAILED=$((FAILED + 1))
    else
      printf "SKIP   %s\n" "$resp"
      SKIPPED=$((SKIPPED + 1))
    fi
    sleep 0.25
  done
done

T_TOTAL=$((SECONDS - T_GLOBAL))
echo
echo "=== Done ==="
printf "  %d total, %d warmed, %d skipped, %d failed (%dm %ds)\n" \
  "$TOTAL" "$WARMED" "$SKIPPED" "$FAILED" "$((T_TOTAL / 60))" "$((T_TOTAL % 60))"

if [ "$FAILED" -gt 0 ]; then
  exit 1
fi
