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

echo "=== Downloading Wav2Lip weights ==="
mkdir -p checkpoints
if [ ! -f "checkpoints/wav2lip_gan.pth" ]; then
  # Mirror from the original HuggingFace model distribution
  wget -q -O checkpoints/wav2lip_gan.pth "https://huggingface.co/camenduru/Wav2Lip/resolve/main/wav2lip_gan.pth" || {
    echo "HF mirror failed, trying alternate..."
    wget -q -O checkpoints/wav2lip_gan.pth "https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA"
  }
fi
ls -la checkpoints/

echo "=== Downloading face detection weights ==="
if [ ! -f "face_detection/detection/sfd/s3fd.pth" ]; then
  mkdir -p face_detection/detection/sfd
  wget -q -O face_detection/detection/sfd/s3fd.pth "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" || \
  wget -q -O face_detection/detection/sfd/s3fd.pth "https://github.com/1adrianb/face-alignment/releases/download/v1.0.1/s3fd-619a316812.pth"
fi

echo "=== All deps installed. Next: copy server.py and start it. ==="
echo "To start:  cd /workspace/Wav2Lip && source venv/bin/activate && python /workspace/wav2lip_server.py"
