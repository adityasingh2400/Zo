#!/usr/bin/env bash
# Background retry loop for idle_attentive when Veo is rate-limited.
# Tries every 2 minutes until success.
set -u
cd /Users/aditya/Desktop/ychackathon
source backend/venv/bin/activate
attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "[$(date +%H:%M:%S)] attempt $attempt"
  python -u phase0/scripts/veo_idle_library.py idle_attentive > /tmp/veo_attentive_attempt_${attempt}.log 2>&1
  if grep -q "OK   (1)" /tmp/veo_attentive_attempt_${attempt}.log 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] SUCCESS on attempt $attempt"
    break
  fi
  echo "[$(date +%H:%M:%S)] failed, sleeping 120s..."
  sleep 120
  if [ "$attempt" -ge 20 ]; then
    echo "[$(date +%H:%M:%S)] gave up after 20 attempts"
    break
  fi
done
