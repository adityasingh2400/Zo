#!/usr/bin/env bash
# Upload Wav2Lip speaking-variant substrates to the pod's /workspace/idle_speaking/.
#
# These are the source frames Wav2Lip uses to stamp live response audio onto
# whichever Tier 0 idle is currently visible. They mirror the body language
# of the silent idles but with active mouth motion that settles to a closed
# anchor pose in the final 2.5s — so the response crossfades back to the
# silent idle layer without a body-pose jump.
#
# The Director (backend/agents/avatar_director.py) references these by their
# absolute pod path:
#     /workspace/idle_speaking/idle_calm_speaking.mp4
#     /workspace/idle_speaking/idle_thinking_speaking.mp4
#     /workspace/idle_speaking/misc_hair_touch_speaking.mp4
#     /workspace/idle_speaking/misc_glance_aside_speaking.mp4
#     /workspace/idle_speaking/idle_reading_comments_speaking.mp4
# (Currently only idle_calm_speaking + idle_thinking_speaking are referenced
#  by TIER0_LIBRARY; reading_comments and hair_touch consolidate onto calm.
#  Uploading all five anyway so future wiring changes don't need a re-push.)
#
# Usage:
#   ./phase0/scripts/upload_speaking_variants.sh
#
# Requires .env with: RUNPOD_POD_IP, RUNPOD_SSH_PORT, RUNPOD_SSH_KEY
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOCAL_DIR="$ROOT/phase0/assets/states/idle"
POD_DIR="/workspace/idle_speaking"

# Load .env (skip blank/comment lines). Read line-by-line so values with
# spaces or special chars survive without needing process substitution.
if [ -f "$ROOT/.env" ]; then
  while IFS= read -r line; do
    case "$line" in
      ''|\#*) continue ;;
      *=*) export "$line" ;;
    esac
  done < "$ROOT/.env"
fi

: "${RUNPOD_POD_IP:?missing RUNPOD_POD_IP in .env}"
: "${RUNPOD_SSH_PORT:?missing RUNPOD_SSH_PORT in .env}"
: "${RUNPOD_SSH_KEY:?missing RUNPOD_SSH_KEY in .env}"

# Resolve any ~ in the key path.
SSH_KEY="${RUNPOD_SSH_KEY/#\~/$HOME}"

# ssh uses lowercase -p for port; scp uses uppercase -P. Build them separately
# as arrays so spaces in the key path or future flags don't blow up via word
# splitting.
SSH_OPTS=(-i "$SSH_KEY" -p "$RUNPOD_SSH_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
SCP_OPTS=(-i "$SSH_KEY" -P "$RUNPOD_SSH_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)

VARIANTS=(
  idle_calm_speaking.mp4
  idle_reading_comments_speaking.mp4
  idle_thinking_speaking.mp4
  misc_glance_aside_speaking.mp4
  misc_hair_touch_speaking.mp4
)

echo "→ Ensuring pod dir exists: $POD_DIR"
ssh "${SSH_OPTS[@]}" "root@$RUNPOD_POD_IP" "mkdir -p $POD_DIR"

uploaded=0
skipped=0
for v in "${VARIANTS[@]}"; do
  src="$LOCAL_DIR/$v"
  if [ ! -e "$src" ]; then
    echo "  [skip] $v not present locally"
    skipped=$((skipped+1))
    continue
  fi
  # Resolve symlink so scp uploads the actual file, not a link.
  real="$(readlink -f "$src" 2>/dev/null || python3 -c "import os,sys;print(os.path.realpath(sys.argv[1]))" "$src")"
  size_kb=$(($(stat -f%z "$real" 2>/dev/null || stat -c%s "$real") / 1024))
  echo "  [up ] $v (${size_kb} KB)  src=$(basename "$real")"
  scp "${SCP_OPTS[@]}" -q "$real" "root@$RUNPOD_POD_IP:$POD_DIR/$v"
  uploaded=$((uploaded+1))
done

echo
echo "── pod dir state ──"
ssh "${SSH_OPTS[@]}" "root@$RUNPOD_POD_IP" "ls -la $POD_DIR"

echo
echo "uploaded=$uploaded  skipped=$skipped"
