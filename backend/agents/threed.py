"""3D-ish product view generation for EMPIRE.

Strategy:
  Tier 1 (no GPU, runs locally): extract N angle frames from a product video,
    rembg-clean each, save as static files in renders/. Dashboard plays them
    as a smooth rotating carousel — looks like a 3D model spin, real photos.
  Tier 2 (RunPod GPU): single image -> GLB mesh via TripoSR/Hunyuan3D.
    Same `View3D` contract; dashboard chooses viewer by `kind`.

Both produce the same shape:
    {
      "kind":   "frame_carousel" | "glb",
      "frames": ["/renders/spin/abc/0.png", ...]   # carousel only
      "url":    "/renders/abc.glb"                  # glb only
      "ms":     int,
      "source": "video" | "triposr" | "hunyuan3d",
    }
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("empire.threed")

RENDER_DIR = Path(__file__).resolve().parent.parent / "renders"
SPIN_DIR = RENDER_DIR / "spin"
SPIN_DIR.mkdir(parents=True, exist_ok=True)


# ── Tier 1: video → angle carousel ───────────────────────────────────────────


async def carousel_from_video(
    video_path: str,
    *,
    n_frames: int = 12,
    out_size: int = 512,
    clean_bg: bool = True,
) -> dict[str, Any]:
    """Extract N angle frames spread across the video, optionally rembg them,
    save to /renders/spin/<hash>/. Returns the View3D dict."""
    t_all = time.perf_counter()

    duration = await _video_duration(video_path)
    if duration <= 0:
        return {"kind": "frame_carousel", "frames": [], "ms": 0, "source": "video", "error": "no_duration"}

    # Slug = sha256(video bytes)[:10] so re-runs on same video hit the cache
    slug = _video_slug(video_path)
    out_dir = SPIN_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    # If we already have a full set cached, reuse
    cached = sorted(out_dir.glob("*.png"))
    if len(cached) >= n_frames:
        urls = [f"/renders/spin/{slug}/{p.name}" for p in cached[:n_frames]]
        ms = int((time.perf_counter() - t_all) * 1000)
        logger.info("carousel cache hit: %s (%d frames, %dms)", slug, len(urls), ms)
        return {"kind": "frame_carousel", "frames": urls, "ms": ms, "source": "video", "slug": slug, "cached": True}

    # 1. Extract candidate frames at 3x density via ffmpeg, then pick sharpest per slot
    candidates_per_slot = 3
    target_frames = n_frames * candidates_per_slot
    fps = max(target_frames / duration, 0.5)

    t0 = time.perf_counter()
    raw = await _ffmpeg_extract_jpegs(video_path, fps=fps)
    extract_ms = int((time.perf_counter() - t0) * 1000)
    if not raw:
        return {"kind": "frame_carousel", "frames": [], "ms": 0, "source": "video", "error": "extract_failed"}

    # 2. Pick sharpest frame per timeline slot (good motion handling)
    t0 = time.perf_counter()
    picks = _pick_sharpest_per_slot(raw, n_slots=n_frames)
    pick_ms = int((time.perf_counter() - t0) * 1000)

    # 3. Resize + (optional) rembg + save
    t0 = time.perf_counter()
    saved: list[str] = []

    if clean_bg:
        try:
            from rembg import new_session, remove
            session = new_session("u2netp")  # ~5MB model, fast
        except Exception as e:
            logger.warning("rembg unavailable, saving raw: %s", e)
            session = None
    else:
        session = None

    for i, jpeg in enumerate(picks):
        out_path = out_dir / f"{i:02d}.png"
        try:
            img = Image.open(io.BytesIO(jpeg)).convert("RGB")
            img = _square_resize(img, out_size)
            if session is not None:
                img = remove(img, session=session)  # RGBA
            img.save(out_path, format="PNG", optimize=False)
            saved.append(f"/renders/spin/{slug}/{out_path.name}")
        except Exception as e:
            logger.warning("frame %d failed: %s", i, e)

    process_ms = int((time.perf_counter() - t0) * 1000)
    total_ms = int((time.perf_counter() - t_all) * 1000)

    logger.info(
        "carousel built: %s, %d frames, %dms (extract=%d pick=%d process=%d, clean_bg=%s)",
        slug, len(saved), total_ms, extract_ms, pick_ms, process_ms, clean_bg,
    )

    return {
        "kind": "frame_carousel",
        "frames": saved,
        "ms": total_ms,
        "source": "video",
        "slug": slug,
        "cached": False,
        "timings": {
            "extract_ms": extract_ms,
            "pick_ms": pick_ms,
            "process_ms": process_ms,
        },
    }


# ── Tier 2: image → GLB (RunPod) ─────────────────────────────────────────────


async def glb_from_image(image_b64: str) -> dict[str, Any]:
    """Send single product image to remote 3D server (TripoSR / Hunyuan3D),
    get back GLB. Cached by image hash."""
    import base64
    import httpx

    from config import RUNPOD_POD_IP, RUNPOD_TRIPOSR_PORT

    if not RUNPOD_POD_IP:
        return {"kind": "glb", "url": None, "error": "no_pod", "source": "triposr"}

    img_bytes = base64.b64decode(image_b64)
    slug = hashlib.sha256(img_bytes).hexdigest()[:10]
    out_path = RENDER_DIR / f"model_{slug}.glb"

    if out_path.exists():
        return {"kind": "glb", "url": f"/renders/{out_path.name}", "ms": 0, "source": "triposr", "cached": True}

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"http://{RUNPOD_POD_IP}:{RUNPOD_TRIPOSR_PORT}/generate",
                json={"image_b64": image_b64, "format": "glb"},
            )
            resp.raise_for_status()
            data = resp.json()
            glb_b64 = data.get("glb_b64")
            if not glb_b64:
                return {"kind": "glb", "url": None, "error": "no_glb_in_response", "source": "triposr"}
            out_path.write_bytes(base64.b64decode(glb_b64))
    except Exception as e:
        logger.warning("triposr remote failed: %s", e)
        return {"kind": "glb", "url": None, "error": str(e), "source": "triposr"}

    ms = int((time.perf_counter() - t0) * 1000)
    return {"kind": "glb", "url": f"/renders/{out_path.name}", "ms": ms, "source": "triposr"}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _video_slug(video_path: str) -> str:
    h = hashlib.sha256()
    with open(video_path, "rb") as f:
        while chunk := f.read(1 << 16):
            h.update(chunk)
    return h.hexdigest()[:10]


async def _video_duration(video_path: str) -> float:
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    try:
        return float(stdout.decode().strip())
    except Exception:
        return 0.0


async def _ffmpeg_extract_jpegs(video_path: str, fps: float) -> list[bytes]:
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps:.3f}",
        "-f", "image2pipe", "-vcodec", "mjpeg",
        "-q:v", "2", "-loglevel", "error", "-",
    ]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, _ = await proc.communicate()
    if proc.returncode != 0:
        return []

    frames: list[bytes] = []
    SOI, EOI = b"\xff\xd8", b"\xff\xd9"
    buf = bytearray(stdout)
    while True:
        s = buf.find(SOI)
        if s == -1:
            break
        e = buf.find(EOI, s + 2)
        if e == -1:
            break
        frames.append(bytes(buf[s : e + 2]))
        buf = buf[e + 2 :]
    return frames


def _sharpness(jpeg: bytes) -> float:
    arr = np.frombuffer(jpeg, np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _pick_sharpest_per_slot(frames: list[bytes], n_slots: int) -> list[bytes]:
    if len(frames) <= n_slots:
        return frames
    seg = len(frames) / n_slots
    picks = []
    for i in range(n_slots):
        start = int(i * seg)
        end = int((i + 1) * seg) if i < n_slots - 1 else len(frames)
        chunk = frames[start:end]
        if not chunk:
            continue
        picks.append(max(chunk, key=_sharpness))
    return picks


def _square_resize(img: Image.Image, size: int) -> Image.Image:
    """Center-crop to square then resize."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((size, size), Image.LANCZOS)
