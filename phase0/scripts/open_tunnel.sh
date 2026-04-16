#!/usr/bin/env bash
# Open SSH tunnel to the pod for lip-sync benchmarking/usage.
#  - 8010 -> Wav2Lip server
#  - 8766 -> LatentSync server
# Keeps running in the foreground; Ctrl-C to stop.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
set -a; source "$ROOT/.env"; set +a
exec ssh -i "$RUNPOD_SSH_KEY" -o StrictHostKeyChecking=no \
  -L 8010:localhost:8010 \
  -L 8766:localhost:8766 \
  -p "$RUNPOD_SSH_PORT" -N "root@$RUNPOD_POD_IP"
