#!/bin/bash
# Deploy GFPGAN face restoration on the same RunPod pod that runs Wav2Lip.
# Adds ~1-1.5s per render but visibly sharpens teeth, eyes, and skin texture
# in the predicted mouth area. Runs in the same venv as Wav2Lip so it shares
# torch/cuda — no second python process.
#
# Pre-req: deploy_wav2lip.sh has already run. /workspace/Wav2Lip exists with
# its venv activated and torch installed.
#
# Usage (from laptop):
#   scp phase0/runpod/deploy_gfpgan.sh root@<IP>:/workspace/
#   ssh root@<IP> 'bash /workspace/deploy_gfpgan.sh'
#
# After install, restart the wav2lip server:
#   pkill -f wav2lip_server_v2 ; cd /workspace/Wav2Lip && source venv/bin/activate \
#     && GFPGAN_ENABLED=1 nohup python /workspace/wav2lip_server_v2.py \
#        > /workspace/w2l.log 2>&1 &

set -e

cd /workspace/Wav2Lip
source venv/bin/activate

echo "=== GFPGAN runtime deps ==="
# basicsr is the backbone GFPGAN uses for the restorer arch.
# realesrgan is its background upsampler (we disable it; we just want the face).
# facexlib provides the face detection + alignment GFPGAN uses internally
# when has_aligned=False; we use has_aligned=True so this is a courtesy install.
# Pin gfpgan==1.3.8 — last release before they broke the API in unmerged commits.
pip install -q --upgrade pip
pip install -q gfpgan==1.3.8 basicsr==1.4.2 facexlib==0.3.0 realesrgan==0.3.0 || true

# basicsr 1.4.2 uses torchvision.transforms.functional_tensor which was removed
# in torchvision >=0.17. Patch the one import via raw filesystem lookup —
# we can't `import basicsr` here because that very import is what's broken.
SITE_PACKAGES=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
BASICSR_DEG="$SITE_PACKAGES/basicsr/data/degradations.py"
if [ ! -f "$BASICSR_DEG" ]; then
  # fallback: hunt for it
  BASICSR_DEG=$(find "$SITE_PACKAGES" -path "*/basicsr/data/degradations.py" -print -quit)
fi
echo "Patching basicsr import: $BASICSR_DEG"
sed -i 's|from torchvision.transforms.functional_tensor import rgb_to_grayscale|from torchvision.transforms.functional import rgb_to_grayscale|' "$BASICSR_DEG"

# Verify the import works now
python -c "from basicsr.data.degradations import circular_lowpass_kernel; print('basicsr import ok')"

echo "=== Downloading GFPGAN v1.4 weights (~340 MB) ==="
mkdir -p checkpoints
if [ ! -s "checkpoints/GFPGANv1.4.pth" ]; then
  wget -q --show-progress --timeout=120 -O checkpoints/GFPGANv1.4.pth \
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
fi

# GFPGAN's facexlib will try to lazy-download these on first call; pre-stage
# them so the first request doesn't pay a 5-10s download tax.
echo "=== Pre-staging facexlib weights ==="
mkdir -p gfpgan/weights
for url in \
  "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth" \
  "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth" ; do
  fname=$(basename "$url")
  if [ ! -s "gfpgan/weights/$fname" ]; then
    wget -q --timeout=60 -O "gfpgan/weights/$fname" "$url"
  fi
done

echo "=== Smoke test: load GFPGANer in-process ==="
python - <<'PY'
import time, os
os.chdir("/workspace/Wav2Lip")
t0 = time.perf_counter()
from gfpgan import GFPGANer
print(f"  import: {time.perf_counter()-t0:.2f}s")

t0 = time.perf_counter()
ENH = GFPGANer(
    model_path="checkpoints/GFPGANv1.4.pth",
    upscale=1,                  # we resize back ourselves
    arch="clean",
    channel_multiplier=2,
    bg_upsampler=None,          # face only — saves the 1.5GB realesrgan load
)
print(f"  load:   {time.perf_counter()-t0:.2f}s")

# Real restore on a synthetic 96x96 face-shaped patch (zeros are fine for timing).
import numpy as np, cv2
patch = np.full((96, 96, 3), 128, dtype=np.uint8)
t0 = time.perf_counter()
_, restored, _ = ENH.enhance(patch, has_aligned=True, only_center_face=True, paste_back=False)
print(f"  enhance(96x96 -> 512x512): {(time.perf_counter()-t0)*1000:.1f}ms  out={restored[0].shape if restored else 'None'}")
PY

echo
echo "=== GFPGAN ready. ==="
echo "Restart server with GFPGAN enabled:"
echo "  pkill -f wav2lip_server_v2 || true"
echo "  cd /workspace/Wav2Lip && source venv/bin/activate \\"
echo "    && GFPGAN_ENABLED=1 nohup python /workspace/wav2lip_server_v2.py \\"
echo "       > /workspace/w2l.log 2>&1 &"
