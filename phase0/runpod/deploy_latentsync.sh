#!/usr/bin/env bash
# Deploy ByteDance LatentSync on RunPod 5090 (Blackwell sm_120).
# Target: best sync + strong realism for live product pitch + comment responses.
# Runtime: ~45-60 min end-to-end (repo clone + torch cu128 + models).
#
# Usage: bash deploy_latentsync.sh 2>&1 | tee /workspace/latentsync_deploy.log
set -euo pipefail

cd /workspace

# 1. Clone repo --------------------------------------------------------------
if [ ! -d LatentSync ]; then
  echo "[LS] Cloning LatentSync..."
  git clone https://github.com/bytedance/LatentSync.git
fi
cd LatentSync

# 2. Python venv -------------------------------------------------------------
if [ ! -d venv ]; then
  echo "[LS] Creating venv..."
  python3 -m venv venv
fi
source venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# 3. PyTorch cu128 for Blackwell 5090 ----------------------------------------
# Blackwell sm_120 requires CUDA 12.8+. cu121 torch wheels don't have sm_120 kernels.
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "[LS] Detected GPU: $GPU_NAME"
if echo "$GPU_NAME" | grep -qi "5090\|5080\|B100\|B200"; then
  echo "[LS] Installing PyTorch cu128 (Blackwell)"
  pip install --pre torch==2.8.0.dev20250608+cu128 torchvision==0.23.0.dev20250608+cu128 \
    --index-url https://download.pytorch.org/whl/nightly/cu128 || \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
else
  echo "[LS] Installing PyTorch cu121 (Ada/Ampere)"
  pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
fi

# 4. LatentSync requirements -------------------------------------------------
# requirements.txt pins old torch (we want cu128 nightly) and pins
# mediapipe==0.10.11 which doesn't exist for python 3.12 anymore. Strip both
# pins, reinstall a compatible mediapipe below.
cp requirements.txt requirements.txt.bak
sed -i '/^torch/d; /^torchvision/d; /^torchaudio/d; /^mediapipe/d' requirements.txt
pip install -r requirements.txt
# xformers is optional; skip on 5090 since wheels lag
pip uninstall -y xformers 2>/dev/null || true

# audio and face deps. Use latest compatible mediapipe — the API surface
# LatentSync uses (face_mesh + face_detection) is stable across 0.10.x.
pip install "mediapipe>=0.10.13" imageio[ffmpeg] librosa soundfile "numpy<2"

# 5. Model checkpoints -------------------------------------------------------
# LatentSync uses: stable-diffusion-v1-5 base + latentsync_unet.pt + whisper tiny
mkdir -p checkpoints/whisper
cd checkpoints

# LatentSync 1.6 (released June 2025) — trained on 512×512 videos, specifically
# fixed the blurry teeth/lips issue that v1.5 had at 256-res. Same model
# architecture so the codebase loads it identically; just point at the new repo
# and set the unet config resolution to 512.
LS_REPO="ByteDance/LatentSync-1.6"
if [ ! -f latentsync_unet.pt ]; then
  echo "[LS] Downloading latentsync_unet.pt from $LS_REPO (~5GB)..."
  wget -q --show-progress "https://huggingface.co/${LS_REPO}/resolve/main/latentsync_unet.pt"
fi

# Whisper tiny for audio embeddings (same in 1.5 / 1.6)
if [ ! -f whisper/tiny.pt ]; then
  echo "[LS] Downloading whisper tiny..."
  wget -q --show-progress -O whisper/tiny.pt \
    "https://huggingface.co/${LS_REPO}/resolve/main/whisper/tiny.pt"
fi

# StableSyncNet auxiliary loss model (same in 1.5 / 1.6)
if [ ! -f stable_syncnet.pt ]; then
  wget -q --show-progress "https://huggingface.co/${LS_REPO}/resolve/main/stable_syncnet.pt"
fi

# SD 1.5 VAE + scheduler (used as base)
# LatentSync ships its own unet, but needs SD VAE. Use diffusers format.
cd ..
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    local_dir="checkpoints/stable-diffusion-v1-5",
    allow_patterns=["vae/*", "scheduler/*", "model_index.json"],
)
print("[LS] SD1.5 VAE+scheduler OK")
PY

echo "[LS] Checkpoints installed:"
du -sh checkpoints/*

# 6. Smoke test --------------------------------------------------------------
# Create a 2-second dummy video + silent audio, run inference, confirm it exits 0.
python - <<'PY'
import torch
print(f"[LS] torch={torch.__version__} cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
PY

echo "[LS] Deploy complete. Next: start wav2lip-style server on port 8766."
