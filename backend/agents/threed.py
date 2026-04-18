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
    n_frames: int = 24,
    out_size: int = 640,
    clean_bg: bool = True,
    rembg_model: str = "u2net",
    stabilize: bool = True,
    drop_blurriest_pct: float = 0.15,
    min_coverage: float = 0.01,
) -> dict[str, Any]:
    """Extract N angle frames spread across the video and produce a
    polished spin-ready set under /renders/spin/<hash>/.

    Quality steps (in order):
      1. ffmpeg extracts 3x-density candidates so we have headroom to drop bad ones.
      2. _pick_sharpest_per_slot picks the sharpest candidate per timeline slot.
      3. rembg cuts the background; we use 'u2net' by default (~170MB, much
         cleaner edges than u2netp) when available, falling back gracefully.
      4. STABILIZATION: every frame's alpha mask gives us the product bbox.
         We pick a *global* bbox (the union of all per-frame bboxes, padded
         and squared) and crop every frame to it. Result: product stays at
         constant size, centered, no breathing.
      5. Frames whose product coverage is below `min_coverage` (% of pixels
         with alpha > 0) are dropped — those are usually motion-blur shots
         where rembg returned mostly-empty.

    Args:
      n_frames: target frame count after dropping bad ones (default 24).
      out_size: output square edge in px (default 640 — looks crisp at the
                ~280px dashboard render size and on retina).
      rembg_model: "u2net" (sharper, ~170MB) or "u2netp" (smaller, faster).
      stabilize: enable bbox stabilization. Disable to debug raw rembg.
      min_sharpness: Laplacian variance threshold; frames below get dropped.
      min_coverage: alpha coverage threshold (0..1); frames below get dropped.
    """
    t_all = time.perf_counter()

    duration = await _video_duration(video_path)
    if duration <= 0:
        return {"kind": "frame_carousel", "frames": [], "ms": 0, "source": "video", "error": "no_duration"}

    slug = _video_slug(video_path)
    out_dir = SPIN_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cache hit: every output frame already exists for this video hash.
    cached = sorted(out_dir.glob("*.png"))
    if len(cached) >= n_frames:
        urls = [f"/renders/spin/{slug}/{p.name}" for p in cached[:n_frames]]
        ms = int((time.perf_counter() - t_all) * 1000)
        logger.info("carousel cache hit: %s (%d frames, %dms)", slug, len(urls), ms)
        return {
            "kind": "frame_carousel", "frames": urls, "ms": ms,
            "source": "video", "slug": slug, "cached": True,
        }

    # 1. Extract 3x candidates so we can drop blurry / motion-corrupt shots.
    candidates_per_slot = 3
    target_frames = n_frames * candidates_per_slot
    fps = max(target_frames / duration, 0.5)

    t0 = time.perf_counter()
    raw = await _ffmpeg_extract_jpegs(video_path, fps=fps)
    extract_ms = int((time.perf_counter() - t0) * 1000)
    if not raw:
        return {"kind": "frame_carousel", "frames": [], "ms": 0, "source": "video", "error": "extract_failed"}

    # 2. Pick sharpest candidate per timeline slot.
    t0 = time.perf_counter()
    picks = _pick_sharpest_per_slot(raw, n_slots=n_frames)
    pick_ms = int((time.perf_counter() - t0) * 1000)

    # 3. rembg setup. u2net = sharper edges than u2netp; if it isn't downloaded
    #    yet rembg will fetch it on first init (one-time ~170MB).
    session = None
    if clean_bg:
        try:
            from rembg import new_session, remove
            session = new_session(rembg_model)
        except Exception as e:
            logger.warning("rembg %s unavailable, trying u2netp: %s", rembg_model, e)
            try:
                from rembg import new_session, remove
                session = new_session("u2netp")
            except Exception as e2:
                logger.warning("rembg unavailable entirely: %s", e2)
                session = None

    # 4. First pass: rembg every frame, compute bbox + sharpness + coverage.
    t0 = time.perf_counter()
    frame_records: list[dict[str, Any]] = []
    for i, jpeg in enumerate(picks):
        try:
            img = Image.open(io.BytesIO(jpeg)).convert("RGB")
            sharp = _img_sharpness(img)
            if sharp < min_sharpness:
                logger.debug("frame %d dropped (sharpness=%.1f < %.1f)", i, sharp, min_sharpness)
                continue
            if session is not None:
                rgba = remove(img, session=session)  # RGBA
                bbox, coverage = _alpha_bbox(rgba)
            else:
                rgba = img.convert("RGBA")
                bbox, coverage = (0, 0, img.width, img.height), 1.0
            if coverage < min_coverage:
                logger.debug("frame %d dropped (coverage=%.3f < %.3f)", i, coverage, min_coverage)
                continue
            frame_records.append({
                "idx": i, "rgba": rgba, "bbox": bbox,
                "coverage": coverage, "sharpness": sharp,
            })
        except Exception as e:
            logger.warning("frame %d rembg failed: %s", i, e)

    if not frame_records:
        return {"kind": "frame_carousel", "frames": [], "ms": 0,
                "source": "video", "error": "all_frames_dropped"}

    # 5. Compute global bbox = union of every kept frame's bbox, padded.
    if stabilize:
        global_bbox = _global_bbox([r["bbox"] for r in frame_records],
                                   pad_pct=0.10,
                                   square=True,
                                   img_w=frame_records[0]["rgba"].width,
                                   img_h=frame_records[0]["rgba"].height)
    else:
        global_bbox = None

    # 6. Crop to global bbox, resize, save. Sequence-renumbered so dashboard
    #    sees a clean 0..N-1 set even if some inputs were dropped.
    saved: list[str] = []
    for j, rec in enumerate(frame_records):
        out_path = out_dir / f"{j:02d}.png"
        rgba = rec["rgba"]
        try:
            if global_bbox is not None:
                rgba = rgba.crop(global_bbox)
            rgba = _square_resize_rgba(rgba, out_size)
            rgba.save(out_path, format="PNG", optimize=False)
            saved.append(f"/renders/spin/{slug}/{out_path.name}")
        except Exception as e:
            logger.warning("frame %d save failed: %s", j, e)

    process_ms = int((time.perf_counter() - t0) * 1000)
    total_ms = int((time.perf_counter() - t_all) * 1000)

    logger.info(
        "carousel built: %s, %d frames (kept %d/%d), %dms "
        "(extract=%d pick=%d process=%d, model=%s, stabilize=%s)",
        slug, len(saved), len(frame_records), len(picks), total_ms,
        extract_ms, pick_ms, process_ms, rembg_model, stabilize,
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
        "stats": {
            "candidates": len(picks),
            "kept": len(frame_records),
            "dropped": len(picks) - len(frame_records),
            "rembg_model": rembg_model if session is not None else None,
            "stabilized": stabilize,
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
    """Center-crop RGB to square then resize."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((size, size), Image.LANCZOS)


def _square_resize_rgba(img: Image.Image, size: int) -> Image.Image:
    """Pad RGBA to square (preserving aspect ratio), then resize.
    Uses a transparent canvas so the product floats cleanly on the dashboard
    without a distorted aspect ratio."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((size, size), Image.LANCZOS)


def _img_sharpness(img: Image.Image) -> float:
    """Laplacian variance on a grayscale PIL image (PIL → numpy → cv2)."""
    arr = np.asarray(img.convert("L"))
    return float(cv2.Laplacian(arr, cv2.CV_64F).var())


def _alpha_bbox(rgba: Image.Image,
                alpha_threshold: int = 16) -> tuple[tuple[int, int, int, int], float]:
    """For a rembg RGBA image, return (bbox, coverage):
      - bbox = tight (l, t, r, b) of pixels with alpha > threshold
      - coverage = fraction of total pixels that exceed threshold
    Both are 0-based; bbox is exclusive on r/b (PIL convention)."""
    if rgba.mode != "RGBA":
        rgba = rgba.convert("RGBA")
    a = np.asarray(rgba.split()[-1])
    mask = a > alpha_threshold
    coverage = float(mask.mean()) if mask.size else 0.0
    if not mask.any():
        return (0, 0, rgba.width, rgba.height), 0.0
    ys, xs = np.where(mask)
    return (int(xs.min()), int(ys.min()),
            int(xs.max()) + 1, int(ys.max()) + 1), coverage


def _global_bbox(bboxes: list[tuple[int, int, int, int]],
                 pad_pct: float,
                 square: bool,
                 img_w: int,
                 img_h: int) -> tuple[int, int, int, int]:
    """Union of bboxes (so the product never clips off-frame across the spin),
    padded by `pad_pct` of the bbox's longer edge, optionally squared, and
    clipped to image bounds. Squared = same crop applied to every frame so
    the product can't drift off-center."""
    ls = min(b[0] for b in bboxes)
    ts = min(b[1] for b in bboxes)
    rs = max(b[2] for b in bboxes)
    bs = max(b[3] for b in bboxes)
    w, h = rs - ls, bs - ts
    pad = int(max(w, h) * pad_pct)
    ls -= pad; ts -= pad; rs += pad; bs += pad
    if square:
        cw, ch = rs - ls, bs - ts
        side = max(cw, ch)
        cx, cy = (ls + rs) // 2, (ts + bs) // 2
        ls = cx - side // 2
        rs = ls + side
        ts = cy - side // 2
        bs = ts + side
    # Clamp to image bounds. If clamping makes it non-square, accept the
    # slight asymmetry; better than going out of frame.
    ls = max(0, ls); ts = max(0, ts)
    rs = min(img_w, rs); bs = min(img_h, bs)
    return (ls, ts, rs, bs)
