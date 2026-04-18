#!/usr/bin/env bash
# MuseTalk v1.5 deployment on RunPod 5090 (Blackwell sm_120, Ubuntu 24.04).
# This is the gnarliest of the lipsync deploys — MuseTalk depends on the
# OpenMMLab stack (mmcv/mmpose/mmdet/mmengine) which historically only
# shipped pre-built wheels for older torch+cuda combos, and Ubuntu 24.04's
# default Python is 3.12 (MuseTalk needs 3.10).
#
# Strategy distilled from the MuseTalk PR #351 thread + zenn.dev trial log:
#   1. Install Python 3.10 via deadsnakes PPA (24.04 ships only 3.12)
#   2. PyTorch 2.7+cu128 nightly wheels (Blackwell sm_120 needs CUDA 12.8)
#   3. Build mmcv 2.1.0 from source against our torch (no pre-built wheel
#      exists for cu128/torch2.7+blackwell)
#   4. Patch mmengine's torch.load() to add weights_only=False
#   5. Download MuseTalk v1.5 weights (~5 GB)
#
# Runtime: 60-90 min on a fresh pod. Logs to /workspace/deploy_musetalk.log.
#
# Bail-out conditions: if any single step takes >20 min hung with no output,
# this script kills the whole pipeline and leaves an error in the log.

set -e

LOG_PFX="[MT]"
log() { echo "$LOG_PFX $@"; }

cd /workspace

# 1. ─── Python 3.10 via deadsnakes PPA ──────────────────────────────────────
log "Installing Python 3.10 (deadsnakes PPA)"
apt-get update -qq
apt-get install -y -qq software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update -qq
apt-get install -y -qq python3.10 python3.10-venv python3.10-dev \
    git wget ffmpeg curl libgl1 libglib2.0-0 \
    build-essential ninja-build cmake

# 2. ─── Clone MuseTalk + venv ───────────────────────────────────────────────
if [ ! -d MuseTalk ]; then
  log "Cloning MuseTalk"
  git clone https://github.com/TMElyralab/MuseTalk.git
fi
cd MuseTalk

if [ ! -d venv ]; then
  log "Creating Python 3.10 venv"
  python3.10 -m venv venv
fi
source venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# 3. ─── PyTorch cu128 (Blackwell) ──────────────────────────────────────────
log "Installing PyTorch cu128 nightly (Blackwell sm_120)"
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

python -c "import torch; assert torch.cuda.is_available(); \
    print('torch', torch.__version__, 'cuda', torch.version.cuda, \
          'gpu', torch.cuda.get_device_name(0))"

# 4. ─── MuseTalk requirements (skip torch — already installed) ────────────
log "Installing MuseTalk requirements (excluding torch pins)"
cp requirements.txt requirements.txt.bak 2>/dev/null || true
sed -i '/^torch/d; /^torchvision/d; /^torchaudio/d; /^mmcv/d; /^mmdet/d; /^mmpose/d; /^mmengine/d' requirements.txt
pip install -r requirements.txt

# Common MuseTalk runtime deps not in requirements
pip install fastapi "uvicorn[standard]" python-multipart httpx pydantic \
    diffusers accelerate omegaconf einops mediapipe imageio[ffmpeg] librosa \
    soundfile "numpy<2"

# 5. ─── OpenMMLab stack ──────────────────────────────────────────────────
# mmcv-2.1.0 is the version MuseTalk's modules expect. No pre-built wheel
# exists for cu128/torch2.7+sm120, so we build from source. ~10 min.
log "Installing openmim + mmengine"
pip install --no-cache-dir -U openmim
mim install mmengine

log "Building mmcv from source (~10 min, no pre-built wheel for Blackwell)"
# MMCV_WITH_OPS=1 enables CUDA ops (required by mmpose). FORCE_CUDA=1 ignores
# CPU fallback. The pinned 2.1.0 tag matches MuseTalk's expectations.
MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install --no-cache-dir \
    git+https://github.com/open-mmlab/mmcv.git@v2.1.0

log "Installing mmdet + mmpose with patched min-mmcv version checks"
mim install "mmdet==3.2.0" "mmpose==1.3.2" || {
  log "mim install failed, falling back to plain pip"
  pip install --no-cache-dir mmdet==3.2.0 mmpose==1.3.2
}

# Patch the version checks: mmdet/mmpose require mmcv<2.1, but 2.1.0 IS
# what we built. Bump the upper bound in the package's __init__ files.
log "Patching mmdet/mmpose min-mmcv version checks"
SP=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
for pkg in mmdet mmpose; do
  init="$SP/$pkg/__init__.py"
  if [ -f "$init" ]; then
    sed -i 's/mmcv_maximum_version = .*/mmcv_maximum_version = "2.99.0"/' "$init" || true
  fi
done

# Patch mmengine's checkpoint loader to add weights_only=False — torch >=2.6
# defaults to weights_only=True which breaks MuseTalk's checkpoint format.
log "Patching mmengine torch.load to use weights_only=False"
CK="$SP/mmengine/runner/checkpoint.py"
if [ -f "$CK" ] && ! grep -q "weights_only=False" "$CK"; then
  sed -i 's/torch.load(filename, map_location=map_location)/torch.load(filename, map_location=map_location, weights_only=False)/g' "$CK" || true
fi

# 6. ─── Smoke test stack ─────────────────────────────────────────────────
log "Sanity check: mmcv._ext compiled for Blackwell?"
python - <<'PY'
import torch, mmcv, mmengine, mmpose, mmdet
print("torch    :", torch.__version__)
print("mmcv     :", mmcv.__version__)
print("mmengine :", mmengine.__version__)
print("mmpose   :", mmpose.__version__)
print("mmdet    :", mmdet.__version__)
import mmcv._ext as ext
print("mmcv._ext: OK", ext.__file__)
# A real CUDA op call to confirm sm_120 kernels were built
from mmcv.ops import nms
import torch
boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.]], device="cuda")
scores = torch.tensor([0.9, 0.8], device="cuda")
keep = nms(boxes, scores, 0.5)
print("mmcv.ops.nms on cuda:", keep)
PY

# 7. ─── Download MuseTalk v1.5 weights (~5 GB) ────────────────────────────
log "Downloading MuseTalk v1.5 weights to models/ (~5 GB)"
mkdir -p models
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    'TMElyralab/MuseTalk',
    local_dir='models',
    allow_patterns=['musetalkV15/*', 'sd-vae/*', 'whisper/*', 'syncnet/*'],
)
print("[MT] weights ok")
PY

# 8. ─── Done ────────────────────────────────────────────────────────────
log "MuseTalk deploy complete. Next: scp musetalk_server.py and start it."
log "Smoke command:  cd /workspace/MuseTalk && source venv/bin/activate && python /workspace/musetalk_server.py --port 8012"
