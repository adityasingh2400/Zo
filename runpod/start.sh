#!/bin/bash
set -e

echo "=== Starting LiveTalking ==="
cd /app/livetalking
python3 app.py --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1 --port 8010 &

echo "=== Starting TripoSR API ==="
cd /app/triposr
python3 run.py --port 8020 &

echo "=== All services running ==="
wait
