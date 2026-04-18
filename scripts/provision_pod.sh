#!/usr/bin/env bash
# Provision a fresh RunPod GPU pod to serve Wav2Lip for EMPIRE.
#
# Does in one shot:
#   1. scp deploy script + server + substrate videos to /workspace/
#   2. ssh in and run deploy_wav2lip.sh (installs deps, downloads weights)
#   3. leaves you SSHed in at a prompt so you can launch the server
#
# Usage:
#   scripts/provision_pod.sh <POD_IP> <SSH_PORT> <SSH_KEY>
#
# Example:
#   scripts/provision_pod.sh 123.45.67.89 37291 ~/.ssh/id_ed25519
#
# Assumes the target is RunPod's PyTorch 2.1 template (Ubuntu, root).

set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <POD_IP> <SSH_PORT> <SSH_KEY>"
  exit 1
fi

POD_IP="$1"
SSH_PORT="$2"
SSH_KEY="$3"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SSH_COMMON=(-i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
            -o ServerAliveInterval=20 -o ServerAliveCountMax=10 -o ConnectTimeout=20)

echo "=== 1/4 Uploading deploy script + server to pod ==="
scp -P "$SSH_PORT" "${SSH_COMMON[@]}" \
  phase0/runpod/deploy_wav2lip.sh \
  phase0/runpod/wav2lip_server_v2.py \
  "root@$POD_IP:/workspace/"

echo
echo "=== 2/4 Uploading default substrate video ==="
scp -P "$SSH_PORT" "${SSH_COMMON[@]}" \
  phase0/assets/states/state_pitching_pose_speaking_1080p.mp4 \
  "root@$POD_IP:/workspace/"

echo
echo "=== 3/4 Uploading per-idle speaking variants ==="
ssh -p "$SSH_PORT" "${SSH_COMMON[@]}" "root@$POD_IP" "mkdir -p /workspace/idle_speaking"
scp -P "$SSH_PORT" "${SSH_COMMON[@]}" \
  phase0/assets/states/idle/idle_calm_speaking.mp4 \
  phase0/assets/states/idle/idle_reading_comments_speaking.mp4 \
  phase0/assets/states/idle/idle_thinking_speaking.mp4 \
  phase0/assets/states/idle/misc_glance_aside_speaking.mp4 \
  phase0/assets/states/idle/misc_hair_touch_speaking.mp4 \
  "root@$POD_IP:/workspace/idle_speaking/"

echo
echo "=== 4/4 Starting deploy_wav2lip.sh in background on pod ==="
# Kick off deploy detached so SSH drops don't kill it. Log to /workspace/deploy.log.
# Previous attempt died when RunPod cut idle SSH mid-PyTorch-install.
ssh -p "$SSH_PORT" "${SSH_COMMON[@]}" "root@$POD_IP" \
  "nohup bash /workspace/deploy_wav2lip.sh > /workspace/deploy.log 2>&1 & disown; echo PID=\$!"

echo
echo "Tailing deploy.log — safe to Ctrl-C at any time, deploy keeps running."
echo "Resume tailing later with:"
echo "  ssh -p $SSH_PORT -i $SSH_KEY root@$POD_IP 'tail -f /workspace/deploy.log'"
echo
ssh -p "$SSH_PORT" "${SSH_COMMON[@]}" "root@$POD_IP" \
  "tail -F /workspace/deploy.log 2>/dev/null & TP=\$!; while ! grep -q 'All deps installed' /workspace/deploy.log 2>/dev/null; do sleep 3; if ! pgrep -f deploy_wav2lip.sh >/dev/null; then break; fi; done; sleep 2; kill \$TP 2>/dev/null"

cat <<EOF

=====================================================
Deploy complete. Last steps — these are manual:

1. SSH into the pod and start the Wav2Lip server:
     ssh -p $SSH_PORT -i $SSH_KEY root@$POD_IP
     cd /workspace/Wav2Lip && source venv/bin/activate
     python /workspace/wav2lip_server_v2.py
   Leave it running (tmux or screen recommended).

2. In another terminal on your laptop, open the SSH tunnel:
     export RUNPOD_POD_IP=$POD_IP
     export RUNPOD_SSH_PORT=$SSH_PORT
     export RUNPOD_SSH_KEY=$SSH_KEY
     bash phase0/scripts/open_tunnel.sh

3. Verify:
     curl http://127.0.0.1:8010/health
   Expect JSON with gpu info — not "connection refused".

4. Add the three RUNPOD_* vars to .env so open_tunnel.sh keeps working:
     RUNPOD_POD_IP=$POD_IP
     RUNPOD_SSH_PORT=$SSH_PORT
     RUNPOD_SSH_KEY=$SSH_KEY

5. Restart the backend, hold the mic, watch the avatar speak.
=====================================================
EOF
