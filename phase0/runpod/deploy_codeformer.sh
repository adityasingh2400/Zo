#!/bin/bash
# Install CodeFormer in the same Wav2Lip venv as GFPGAN.
# Uses the basicsr-direct path (the codeformer-pip wrapper hangs trying to
# download things on import). We just need the CodeFormer arch (already
# registered by basicsr) + the .pth checkpoint. ~360 MB.
#
# Pre-req: deploy_wav2lip.sh + deploy_gfpgan.sh have already run.
set -e

cd /workspace/Wav2Lip
source venv/bin/activate

echo "=== Downloading CodeFormer weights (~360 MB) ==="
mkdir -p checkpoints
if [ ! -s "checkpoints/codeformer.pth" ]; then
  wget -q --show-progress --timeout=120 -O checkpoints/codeformer.pth \
    "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
fi

echo "=== Smoke test (basicsr-direct path) ==="
python - <<'PY'
import time, os, torch, numpy as np
os.chdir("/workspace/Wav2Lip")

# CodeFormer arch lives in basicsr's registry (basicsr ships the arch file
# upstream). If absent, fall back to git-installing just the arch file.
try:
    from basicsr.utils.registry import ARCH_REGISTRY
    NET_CLS = ARCH_REGISTRY.get("CodeFormer")
    print(f"  arch registered via basicsr: {NET_CLS}")
except KeyError:
    print("  CodeFormer arch not in basicsr registry, installing arch file...")
    import subprocess
    subprocess.run(["pip", "install", "-q",
                    "git+https://github.com/sczhou/CodeFormer.git@master"],
                   check=True)
    # Re-import
    import importlib, basicsr.utils.registry as _r
    importlib.reload(_r)
    from basicsr.utils.registry import ARCH_REGISTRY
    NET_CLS = ARCH_REGISTRY.get("CodeFormer")
    print(f"  arch loaded: {NET_CLS}")

t0 = time.perf_counter()
net = NET_CLS(
    dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
    connect_list=["32", "64", "128", "256"],
).to("cuda")
sd = torch.load("checkpoints/codeformer.pth", map_location="cuda", weights_only=False)
net.load_state_dict(sd["params_ema"])
net.eval()
print(f"  load: {time.perf_counter()-t0:.2f}s")

# Cold + warm timing
x = torch.zeros(1, 3, 512, 512, device="cuda")
with torch.no_grad():
    out = net(x, w=0.7, adain=True)[0]      # cold
torch.cuda.synchronize()
t1 = time.perf_counter()
with torch.no_grad():
    for _ in range(5):
        out = net(x, w=0.7, adain=True)[0]
torch.cuda.synchronize()
print(f"  warm forward (avg over 5): {(time.perf_counter()-t1)*1000/5:.1f}ms  out={tuple(out.shape)}")
PY

echo "=== CodeFormer ready. Pass ENHANCER_TYPE=codeformer when starting wav2lip server ==="
