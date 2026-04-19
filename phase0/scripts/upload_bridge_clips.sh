#!/usr/bin/env bash
# Upload all bridge clips (compliment / objection / question / intro) to the
# RunPod GPU pod under /workspace/bridges/<intent>/<file>.mp4 so the
# Wav2Lip server's /lipsync_fast endpoint can use them as source substrates
# via `source_path`.
#
# This is the substrate-side prerequisite for the new comment dispatch
# path (_run_bridge_with_wav2lip) which lip-syncs Gemma-drafted responses
# on top of intent-specific bridge clips instead of the generic speaking
# pose. Without this upload the new path 404s on /lipsync_fast.
#
# Idempotent: re-running just refreshes anything that changed locally;
# scp always overwrites.
#
# Usage:
#   bash phase0/scripts/upload_bridge_clips.sh
#
# Reads RUNPOD_POD_IP / RUNPOD_SSH_PORT / RUNPOD_SSH_KEY from .env.
# Pass --dry-run as the only argument to test SSH/path resolution
# without uploading anything.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# .env shouldn't be committed; allow override via env vars too.
if [ -f .env ]; then
  set -a; source .env; set +a
fi

: "${RUNPOD_POD_IP:?RUNPOD_POD_IP unset (check .env)}"
: "${RUNPOD_SSH_PORT:?RUNPOD_SSH_PORT unset (check .env)}"
: "${RUNPOD_SSH_KEY:?RUNPOD_SSH_KEY unset (check .env)}"

# Expand ~ if present.
SSH_KEY="${RUNPOD_SSH_KEY/#\~/$HOME}"

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
fi

SSH_COMMON=(-i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
            -o ServerAliveInterval=20 -o ServerAliveCountMax=10 -o ConnectTimeout=15)

# Intent buckets to upload. Each maps to phase0/assets/bridges/<intent>/.
INTENTS=(compliment objection question intro neutral)

# Count + size summary first.
echo "=== Source inventory ==="
TOTAL_BYTES=0
TOTAL_FILES=0
for intent in "${INTENTS[@]}"; do
  src_dir="phase0/assets/bridges/$intent"
  if [ ! -d "$src_dir" ]; then
    echo "  $intent — (no source dir, skipping)"
    continue
  fi
  count=$(find "$src_dir" -maxdepth 1 -name '*.mp4' | wc -l | tr -d ' ')
  bytes=$(du -sk "$src_dir" 2>/dev/null | awk '{print $1}')
  TOTAL_FILES=$((TOTAL_FILES + count))
  TOTAL_BYTES=$((TOTAL_BYTES + bytes))
  printf "  %-12s %3d files  %5d KB\n" "$intent" "$count" "$bytes"
done
printf "  ------------\n  TOTAL        %3d files  %5d KB\n" "$TOTAL_FILES" "$TOTAL_BYTES"

if [ "$DRY_RUN" = "1" ]; then
  echo
  echo "=== DRY RUN — testing SSH connectivity only ==="
  ssh -p "$RUNPOD_SSH_PORT" "${SSH_COMMON[@]}" "root@$RUNPOD_POD_IP" \
    "echo OK from \$(hostname); ls /workspace/ | head -10"
  exit 0
fi

echo
echo "=== Creating /workspace/bridges/<intent>/ on pod ==="
mkdir_cmd=""
for intent in "${INTENTS[@]}"; do
  mkdir_cmd+="mkdir -p /workspace/bridges/$intent; "
done
ssh -p "$RUNPOD_SSH_PORT" "${SSH_COMMON[@]}" "root@$RUNPOD_POD_IP" "$mkdir_cmd echo done"

echo
echo "=== Uploading clips ==="
for intent in "${INTENTS[@]}"; do
  src_dir="phase0/assets/bridges/$intent"
  if [ ! -d "$src_dir" ]; then
    continue
  fi
  files=("$src_dir"/*.mp4)
  if [ ! -e "${files[0]}" ]; then
    echo "  $intent — no .mp4 files"
    continue
  fi
  echo "  Uploading $intent (${#files[@]} files)..."
  scp -P "$RUNPOD_SSH_PORT" "${SSH_COMMON[@]}" -q \
    "${files[@]}" "root@$RUNPOD_POD_IP:/workspace/bridges/$intent/"
done

echo
echo "=== Verifying pod-side inventory ==="
ssh -p "$RUNPOD_SSH_PORT" "${SSH_COMMON[@]}" "root@$RUNPOD_POD_IP" \
  "for d in /workspace/bridges/*/; do printf '  %-30s %3d files\n' \"\$d\" \"\$(ls \"\$d\" | wc -l | tr -d ' ')\"; done"

echo
echo "Done. Wav2Lip can now use /workspace/bridges/<intent>/<file>.mp4 as source_path."
