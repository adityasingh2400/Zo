"""TripoSR server — single image → 3D GLB mesh.

Drop-in FastAPI server intended to run on a CUDA pod (port 8020).
Loads TripoSR once at startup, holds it in VRAM, serves /generate.

Hardware: tested target = RTX 5090 (~32GB VRAM, ~1s/inference).
        : A100 40GB also fine. Anything <12GB will OOM.

Endpoints:
    GET  /health          → {status, model, device, vram_gb}
    POST /generate        → JSON {image_b64, format} → {glb_b64, ms}
    POST /generate_file   → multipart image upload → GLB binary

Memory plan if co-resident with LatentSync on the same GPU:
    LatentSync: ~14GB VRAM (idle), ~22GB peak
    TripoSR:    ~5GB VRAM (idle), ~7GB peak
    → safe on 32GB if no concurrent renders. Use a single asyncio Lock
      across both servers if running on shared GPU.

This file is self-contained: no project imports, just stdlib + torch +
TripoSR + PIL. Deploy by `scp` and `python triposr_server.py`.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import sys
import tempfile
import time
import threading
import uuid
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("triposr")

# ── Config (env-overridable) ─────────────────────────────────────────────────
TRIPOSR_ROOT = Path(os.getenv("TRIPOSR_ROOT", "/workspace/TripoSR"))
WORK_ROOT = Path(os.getenv("TRIPOSR_WORK", "/workspace/work_3d"))
WORK_ROOT.mkdir(parents=True, exist_ok=True)
PORT = int(os.getenv("TRIPOSR_PORT", "8020"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16

# ── Load TripoSR once ────────────────────────────────────────────────────────
sys.path.insert(0, str(TRIPOSR_ROOT))

log.info("loading TripoSR from %s on %s (%s)...", TRIPOSR_ROOT, DEVICE, DTYPE)
t0 = time.perf_counter()

try:
    from tsr.system import TSR
    from tsr.utils import remove_background as tsr_remove_bg, resize_foreground

    MODEL = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    MODEL.renderer.set_chunk_size(8192)
    MODEL.to(DEVICE)
    if DEVICE == "cuda":
        MODEL = MODEL.to(dtype=DTYPE)
    log.info("TripoSR loaded in %.2fs", time.perf_counter() - t0)
except Exception as e:
    log.error("TripoSR load failed: %s", e)
    log.error("install steps: cd /workspace && git clone https://github.com/VAST-AI-Research/TripoSR && cd TripoSR && pip install -r requirements.txt")
    raise

# Optional: rembg for background removal (tsr_remove_bg uses rembg internally)
try:
    from rembg import new_session
    REMBG_SESSION = new_session("u2net")
    log.info("rembg session ready")
except Exception as e:
    log.warning("rembg unavailable, expecting clean foreground: %s", e)
    REMBG_SESSION = None

# Single-render lock — TripoSR is fast but VRAM-bursty
RENDER_LOCK = threading.Lock()

# ── Inference ────────────────────────────────────────────────────────────────


def _decode_b64_image(b64: str) -> Image.Image:
    raw = base64.b64decode(b64.split(",")[-1])  # tolerate data: prefix
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _preprocess(img: Image.Image, do_remove_bg: bool = True) -> Image.Image:
    if do_remove_bg and REMBG_SESSION is not None:
        img = tsr_remove_bg(img, REMBG_SESSION)
        img = resize_foreground(img, 0.85)
    return img


def _run_inference(image: Image.Image, mc_resolution: int = 256) -> bytes:
    """Run image → mesh → GLB bytes. Returns the GLB file contents."""
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=DEVICE == "cuda"):
        scene_codes = MODEL([image], device=DEVICE)
        meshes = MODEL.extract_mesh(scene_codes, resolution=mc_resolution)
    mesh = meshes[0]
    out_path = WORK_ROOT / f"mesh_{uuid.uuid4().hex[:8]}.glb"
    mesh.export(str(out_path), file_type="glb")
    data = out_path.read_bytes()
    out_path.unlink(missing_ok=True)
    return data


# ── API ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="EMPIRE TripoSR")


class GenRequest(BaseModel):
    image_b64: str
    format: str = "glb"
    remove_bg: bool = True
    mc_resolution: int = 256  # 256 = standard, 320 = high-detail (slower)


class GenResponse(BaseModel):
    glb_b64: str
    ms: int
    bytes: int


@app.get("/health")
def health():
    free = total = None
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
    return {
        "status": "ok",
        "model": "TripoSR",
        "device": DEVICE,
        "torch": torch.__version__,
        "vram_free_gb": round(free / 1e9, 2) if free else None,
        "vram_total_gb": round(total / 1e9, 2) if total else None,
        "rembg": REMBG_SESSION is not None,
    }


@app.post("/generate", response_model=GenResponse)
async def generate(req: GenRequest):
    t0 = time.perf_counter()
    try:
        img = _decode_b64_image(req.image_b64)
        img = _preprocess(img, do_remove_bg=req.remove_bg)
        with RENDER_LOCK:
            glb = _run_inference(img, mc_resolution=req.mc_resolution)
    except Exception as e:
        log.exception("generate failed")
        raise HTTPException(500, str(e))

    ms = int((time.perf_counter() - t0) * 1000)
    log.info("generate ok: %dms, %dKB", ms, len(glb) // 1024)
    return GenResponse(glb_b64=base64.b64encode(glb).decode(), ms=ms, bytes=len(glb))


@app.post("/generate_file")
async def generate_file(
    image: UploadFile = File(...),
    remove_bg: bool = Form(True),
    mc_resolution: int = Form(256),
):
    t0 = time.perf_counter()
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = _preprocess(img, do_remove_bg=remove_bg)
        with RENDER_LOCK:
            glb = _run_inference(img, mc_resolution=mc_resolution)
    except Exception as e:
        log.exception("generate_file failed")
        raise HTTPException(500, str(e))

    out_path = WORK_ROOT / f"resp_{uuid.uuid4().hex[:8]}.glb"
    out_path.write_bytes(glb)
    ms = int((time.perf_counter() - t0) * 1000)
    return FileResponse(
        path=out_path,
        media_type="model/gltf-binary",
        filename="model.glb",
        headers={"X-Total-Ms": str(ms), "X-Bytes": str(len(glb))},
        background=None,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
