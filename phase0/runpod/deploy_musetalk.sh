#!/bin/bash
# Plan B: MuseTalk v1.5 deployment on RunPod 5090 (Blackwell, sm_120).
# Significantly more complex than Wav2Lip. Only run this if Wav2Lip quality fails.
#
# Takes ~30-60 min on a fresh pod. Requires NVIDIA Driver 575.x+.

set -e

echo "=== System deps ==="
apt-get update -qq
apt-get install -y -qq python3.10 python3.10-venv python3-pip git wget ffmpeg curl \
    libgl1 libglib2.0-0 build-essential ninja-build

cd /workspace

echo "=== Clone MuseTalk ==="
if [ ! -d "MuseTalk" ]; then
  git clone https://github.com/TMElyralab/MuseTalk.git
fi
cd MuseTalk

echo "=== Python 3.10 venv ==="
python3.10 -m venv venv
source venv/bin/activate
pip install -q --upgrade pip wheel

echo "=== PyTorch 2.10 + CUDA 12.8 (Blackwell 5090) ==="
pip install -q torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "=== Core deps (pin to avoid breakage) ==="
pip install -q numpy==1.26.4 opencv-python-headless librosa numba tqdm scipy
pip install -q transformers diffusers accelerate omegaconf einops
pip install -q mediapipe face_alignment  # replaces mmpose on py3.10+ where mmcv is painful
pip install -q fastapi "uvicorn[standard]" python-multipart httpx pydantic

echo "=== MMCV 1.7.2 (the compat version) ==="
pip install -q -U openmim || true
mim install "mmcv-full==1.7.2" || pip install "mmcv==1.7.2"

echo "=== Download MuseTalk weights (~5 GB) ==="
mkdir -p models
if [ ! -f "models/musetalkV15/unet.pth" ]; then
  python -c "
from huggingface_hub import snapshot_download
snapshot_download('TMElyralab/MuseTalk', local_dir='models', allow_patterns=['musetalkV15/*','sd-vae/*','whisper/*','syncnet/*'])
" || echo "Weight download failed — manual step required"
fi

echo "=== Done. Start server with:  python /workspace/musetalk_server.py ==="
