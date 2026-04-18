#!/usr/bin/env bash
# Deploy TripoSR server to the active RunPod (port 8020). Idempotent.
#
# Required env (in .env or shell):
#   RUNPOD_POD_IP, RUNPOD_SSH_PORT, RUNPOD_SSH_KEY
#
# Cost estimate: ~5 min one-time setup (clone + pip install + 1.6GB weight DL).
# After deploy, the server holds ~5GB VRAM idle. Render = ~1s on RTX 5090.
#
# Co-residence with LatentSync: should fit in 32GB. Watch /health vram_free_gb.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -f .env ]; then
  set -a; source .env; set +a
fi

: "${RUNPOD_POD_IP:?missing RUNPOD_POD_IP}"
: "${RUNPOD_SSH_PORT:?missing RUNPOD_SSH_PORT}"
: "${RUNPOD_SSH_KEY:?missing RUNPOD_SSH_KEY}"

SSH="ssh -i $RUNPOD_SSH_KEY -p $RUNPOD_SSH_PORT -o StrictHostKeyChecking=no root@$RUNPOD_POD_IP"
SCP="scp -i $RUNPOD_SSH_KEY -P $RUNPOD_SSH_PORT -o StrictHostKeyChecking=no"

echo "==> 1/4 sync server script to pod"
$SCP runpod/triposr_server.py "root@$RUNPOD_POD_IP:/workspace/triposr_server.py"

echo "==> 2/4 ensure TripoSR repo + venv on pod"
$SSH bash -lc '
  set -e
  cd /workspace
  if [ ! -d TripoSR ]; then
    git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR
  fi
  cd TripoSR
  if [ ! -d venv_3d ]; then
    python3 -m venv venv_3d
  fi
  source venv_3d/bin/activate
  pip install -q --upgrade pip
  pip install -q -r requirements.txt
  pip install -q fastapi uvicorn[standard] python-multipart pydantic rembg onnxruntime-gpu pillow
  echo "TripoSR venv ready"
' 2>&1 | tail -8

echo "==> 3/4 (re)start triposr_server on :8020"
$SSH bash -lc '
  pkill -f triposr_server.py || true
  sleep 1
  cd /workspace/TripoSR
  source venv_3d/bin/activate
  nohup python3 /workspace/triposr_server.py </dev/null >/workspace/triposr.log 2>&1 &
  disown
  sleep 4
  pgrep -af triposr_server || echo "WARNING: not running yet, check /workspace/triposr.log"
'

echo "==> 4/4 wait for /health (up to 90s for cold model load)"
for i in $(seq 1 30); do
  if curl -fs --max-time 3 "http://$RUNPOD_POD_IP:8020/health" >/dev/null 2>&1; then
    echo "alive after ${i}*3s"
    curl -s "http://$RUNPOD_POD_IP:8020/health" | python3 -m json.tool || true
    exit 0
  fi
  sleep 3
done

echo "TIMED OUT — fetching log tail:"
$SSH "tail -40 /workspace/triposr.log"
exit 1
