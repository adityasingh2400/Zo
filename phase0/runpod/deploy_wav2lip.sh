#!/bin/bash
# Deploy a lip-sync API on a RunPod pod.
# Runs over SSH. Takes ~10-15 min on a fresh pod.
#
# Plan A: Wav2Lip (MIT license, older but rock-solid, fastest deploy)
#   - PyTorch 2.1+ CUDA 12.1
#   - ~1-3s per 10s clip on 5090
#   - Quality: 6.5/10 (teeth area slightly blurry)
#
# Usage (from laptop):
#   scp phase0/runpod/* root@<IP>:/workspace/
#   ssh root@<IP> 'bash /workspace/deploy_wav2lip.sh'

set -e

echo "=== Installing system deps ==="
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv git wget ffmpeg curl libgl1 libglib2.0-0

cd /workspace

echo "=== Cloning Wav2Lip (patched fork for newer PyTorch) ==="
if [ ! -d "Wav2Lip" ]; then
  # rudrabha/Wav2Lip is the original. justinjohn0306/Wav2Lip has compatibility fixes.
  git clone https://github.com/justinjohn0306/Wav2Lip.git
fi
cd Wav2Lip

echo "=== Python venv ==="
python3 -m venv venv
source venv/bin/activate

echo "=== PyTorch cu121 (works on most Ampere+ pods, fallback to cu128 for Blackwell 5090) ==="
# Detect GPU — 5090 is compute capability 12.0
GPU_CAP=$(python3 -c "import subprocess; print(subprocess.check_output(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader,nounits']).decode().strip().split('\n')[0])")
echo "Detected GPU compute capability: $GPU_CAP"

if [ "$GPU_CAP" = "12.0" ] || [ "$GPU_CAP" = "8.9" ]; then
  echo "Installing PyTorch cu128 for Blackwell 5090..."
  pip install -q --upgrade pip
  pip install -q torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
else
  echo "Installing PyTorch cu121 for Ampere/Ada..."
  pip install -q --upgrade pip
  pip install -q torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
fi

echo "=== Wav2Lip deps ==="
pip install -q numpy==1.26.4 opencv-python-headless librosa numba tqdm scipy
pip install -q face_alignment==1.3.5 basicsr || true  # basicsr may fail, acceptable
pip install -q fastapi uvicorn[standard] python-multipart httpx

# RetinaFace via batch-face (used by wav2lip_server_v2.py for fast face detection).
pip install -q batch-face

echo "=== Downloading Wav2Lip weights ==="
mkdir -p checkpoints
if [ ! -s "checkpoints/wav2lip_gan.pth" ]; then
  # Working mirror (Sharepoint and the original camenduru drop both 404 in 2026).
  wget -q --show-progress --timeout=60 -O checkpoints/wav2lip_gan.pth \
    "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth?download=true"
fi

echo "=== Downloading face detection weights ==="
mkdir -p face_detection/detection/sfd
if [ ! -s "face_detection/detection/sfd/s3fd.pth" ]; then
  # adrianbulat.com is unreliable; HF camenduru mirror is fast.
  wget -q --timeout=60 -O face_detection/detection/sfd/s3fd.pth \
    "https://huggingface.co/camenduru/facexlib/resolve/main/s3fd-619a316812.pth?download=true"
fi
if [ ! -s "checkpoints/mobilenet.pth" ]; then
  # batch-face's RetinaFace network=mobilenet expects this file.
  wget -q --timeout=60 -O checkpoints/mobilenet.pth \
    "https://huggingface.co/py-feat/retinaface/resolve/main/mobilenet0.25_Final.pth?download=true"
fi

ls -la checkpoints/ face_detection/detection/sfd/

echo "=== All deps installed. ==="
echo "To start:  cd /workspace/Wav2Lip && source venv/bin/activate && python /workspace/wav2lip_server_v2.py"
