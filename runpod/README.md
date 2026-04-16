# RunPod Setup for EMPIRE

## Quick Start

### Option A: Use RunPod Template (fastest)

1. Go to https://www.runpod.io/gpu-cloud
2. Select A100 80GB ($1.19/hr) or A40 ($0.79/hr)
3. Use template: `nvidia/cuda:12.4.0-runtime-ubuntu22.04`
4. SSH in and run:

```bash
# Install LiveTalking
git clone https://github.com/lipku/LiveTalking.git
cd LiveTalking
pip install -r requirements.txt

# Download Wav2Lip model
mkdir -p checkpoints
# Follow LiveTalking docs for model downloads

# Start
python app.py --transport webrtc --model wav2lip --port 8010
```

### Option B: Use our Dockerfile

```bash
cd runpod
docker build -t empire-gpu .
docker run --gpus all -p 8010:8010 -p 8020:8020 empire-gpu
```

### Testing

Once running, open `http://<POD_IP>:8010/webrtcapi.html` in your browser.
Type text, the avatar should speak with lip sync.

### Connecting to Backend

Set in your `.env`:
```
RUNPOD_POD_IP=<your_pod_ip>
RUNPOD_LIVETALKING_PORT=8010
RUNPOD_TRIPOSR_PORT=8020
```
