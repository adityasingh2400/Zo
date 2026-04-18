#!/usr/bin/env bash
# Keep the SSH tunnels to the RunPod alive across wifi blips and pod restarts.
# Re-opens 8010 (Wav2Lip) and 8766 (LatentSync) any time they go dark.
# Run in a separate terminal:  bash phase0/scripts/keep_tunnels_alive.sh

set -u
cd "$(dirname "$0")/../.."

# Load .env so RUNPOD_* vars are set even from a fresh shell.
if [ -f .env ]; then
  set -a; source .env; set +a
fi

POD_IP="${RUNPOD_POD_IP:-74.2.96.25}"
POD_PORT="${RUNPOD_SSH_PORT:-17415}"
KEY="${RUNPOD_SSH_KEY:-$HOME/.ssh/id_ed25519_empire}"

echo "[tunnels] watching pod $POD_IP:$POD_PORT (key=$KEY)"
echo "[tunnels] press Ctrl+C to stop"

reopen() {
  pkill -f "ssh.*-L 8010" 2>/dev/null
  pkill -f "ssh.*-L 8766" 2>/dev/null
  sleep 1
  ssh -fN -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
      -o StrictHostKeyChecking=no \
      -p "$POD_PORT" -i "$KEY" \
      -L 8010:127.0.0.1:8010 -L 8766:127.0.0.1:8766 \
      "root@$POD_IP" \
      && echo "[tunnels] $(date +%H:%M:%S) reopened" \
      || echo "[tunnels] $(date +%H:%M:%S) reopen FAILED"
}

# Initial open
reopen

# Health-check loop. Both tunnels should respond on /health.
while true; do
  sleep 10
  w2l=$(curl -s -o /dev/null --max-time 4 -w "%{http_code}" http://127.0.0.1:8010/health 2>/dev/null)
  ls=$(curl -s -o /dev/null --max-time 4 -w "%{http_code}" http://127.0.0.1:8766/health 2>/dev/null)
  if [ "$w2l" != "200" ] || [ "$ls" != "200" ]; then
    echo "[tunnels] $(date +%H:%M:%S) health w2l=$w2l ls=$ls -> reopening"
    reopen
  fi
done
