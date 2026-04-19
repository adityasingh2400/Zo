"""CREATOR — content factory for product photos + promo video.

Scope (this build):
  - generate_product_photos: 3 marketplace-style photos from one input frame
    (rembg bg-strip, white-bg composite, branded composite with name + price)
  - generate_promo_video: 15s 9:16 MP4 slideshow stitched from the 3 photos
    via ffmpeg (5s per photo, no audio)
  - build_all: orchestrator that runs the above + the existing 3D model call,
    writes outputs under backend/renders/creator/<request_id>/, returns URLs

What's NOT here yet (roadmap):
  - Vertex AI Imagen / Stability AI generative variants — needs API keys we
    don't have wired today. Once available, drop in a fourth generator that
    creates a "lifestyle scene" with the product composited into a generated
    environment.
  - World Labs Marble 3D world — no integration plan, deferred.
  - Short-form clip variants (TikTok/Reels/Shorts cuts).
  - Per-Q/A response photo (e.g., a close-up of the leather grain to pair
    with the "is it real leather" answer).
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont
from rembg import remove

from config import RUNPOD_POD_IP, RUNPOD_TRIPOSR_PORT

logger = logging.getLogger("empire.creator")

# Outputs go to backend/renders/creator/<request_id>/ — served via the
# existing /renders mount in main.py (line ~1121).
RENDERS_ROOT = Path(__file__).resolve().parent.parent / "renders" / "creator"
RENDERS_ROOT.mkdir(parents=True, exist_ok=True)

# 9:16 vertical canvas for TikTok/Reels/Shorts. 1080×1920 is the standard.
CANVAS_W = 1080
CANVAS_H = 1920


# ── Existing helpers (kept from the original 34-LOC stub) ───────────────────


async def remove_background(frame_b64: str) -> str:
    """Remove background from product photo, return clean PNG as base64."""
    img_bytes = base64.b64decode(frame_b64)
    img = Image.open(io.BytesIO(img_bytes))
    clean = remove(img)
    buf = io.BytesIO()
    clean.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


async def generate_3d_model(frame_b64: str) -> dict | None:
    """Send product image to TripoSR on RunPod, get back GLB mesh URL.
    Returns None if pod isn't configured, or {"error": ...} on failure."""
    if not RUNPOD_POD_IP:
        return None
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"http://{RUNPOD_POD_IP}:{RUNPOD_TRIPOSR_PORT}/generate",
                json={"image": frame_b64, "output_format": "glb"},
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        return {"error": str(e)}


# ── Cross-platform font loader ──────────────────────────────────────────────


def _load_font(size: int):
    """Load a TrueType font, falling through macOS → Linux → PIL default so
    code runs whether deployed locally on macOS or on a Linux pod."""
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",                       # macOS
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",      # Linux
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    """Width of `text` in pixels under `font`. Uses textbbox on modern Pillow,
    falls back to textsize on legacy."""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]
    except AttributeError:
        return draw.textsize(text, font=font)[0]


# ── Photo generation (3 variants from one input frame) ──────────────────────


def _decode_frame(frame_b64: str) -> Image.Image:
    """Decode a base64 PNG/JPEG into an RGBA PIL Image."""
    return Image.open(io.BytesIO(base64.b64decode(frame_b64))).convert("RGBA")


def _fit_canvas(img: Image.Image, bg_color: tuple[int, int, int, int]) -> Image.Image:
    """Resize img to fit inside CANVAS_W×CANVAS_H preserving aspect, then
    pad with bg_color so the output is exactly canvas-sized."""
    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), bg_color)
    fit = img.copy()
    fit.thumbnail((CANVAS_W, CANVAS_H), Image.LANCZOS)
    x = (CANVAS_W - fit.width) // 2
    y = (CANVAS_H - fit.height) // 2
    canvas.paste(fit, (x, y), fit if fit.mode == "RGBA" else None)
    return canvas


def _gen_clean_photo(frame: Image.Image) -> Image.Image:
    """Variant 1: bg-stripped, transparent background. Marketplace-classic."""
    buf = io.BytesIO()
    frame.save(buf, format="PNG")
    bg_stripped = remove(buf.getvalue())
    img = Image.open(io.BytesIO(bg_stripped)).convert("RGBA")
    return _fit_canvas(img, (0, 0, 0, 0))


def _gen_white_bg_photo(clean: Image.Image) -> Image.Image:
    """Variant 2: bg-stripped composited onto solid white. Standard listing
    photo for Amazon/Shopify/marketplace catalogs."""
    bg = Image.new("RGBA", (CANVAS_W, CANVAS_H), (255, 255, 255, 255))
    bg.paste(clean, (0, 0), clean)
    return bg


def _gen_branded_photo(frame: Image.Image, product_data: dict) -> Image.Image:
    """Variant 3: original framed on a dark canvas with product name + price
    baked in as large white-stroke text. The "lifestyle hero" framing — drops
    cleanly into an Instagram post or hero card."""
    name = (product_data.get("name") or "").strip()
    price = (product_data.get("price") or "").strip()

    bg = Image.new("RGBA", (CANVAS_W, CANVAS_H), (15, 15, 18, 255))
    fitted = frame.copy()
    fitted.thumbnail((CANVAS_W - 80, CANVAS_H - 600), Image.LANCZOS)
    x = (CANVAS_W - fitted.width) // 2
    y = (CANVAS_H - fitted.height) // 2 - 140
    bg.paste(fitted, (x, y), fitted if fitted.mode == "RGBA" else None)

    draw = ImageDraw.Draw(bg)
    if name:
        font_name = _load_font(88)
        text_w = _measure_text(draw, name, font_name)
        draw.text(((CANVAS_W - text_w) // 2, CANVAS_H - 320),
                  name, font=font_name, fill="white",
                  stroke_width=5, stroke_fill="black")
    if price:
        font_price = _load_font(140)
        text_w = _measure_text(draw, price, font_price)
        draw.text(((CANVAS_W - text_w) // 2, CANVAS_H - 200),
                  price, font=font_price, fill="white",
                  stroke_width=6, stroke_fill="black")
    return bg


def generate_product_photos(frame_b64: str, product_data: dict,
                            out_dir: Path) -> list[Path]:
    """Generate 3 product photos from one input frame. Writes PNGs to
    out_dir, returns paths in [clean, white_bg, branded] order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = _decode_frame(frame_b64)

    clean = _gen_clean_photo(frame)
    white_bg = _gen_white_bg_photo(clean)
    branded = _gen_branded_photo(frame, product_data)

    paths = []
    for name, img in (("photo_clean.png", clean),
                      ("photo_white_bg.png", white_bg),
                      ("photo_branded.png", branded)):
        p = out_dir / name
        img.save(p, format="PNG")
        paths.append(p)
    return paths


# ── Promo video (ffmpeg slideshow) ──────────────────────────────────────────


async def generate_promo_video(photo_paths: list[Path], out_path: Path,
                                seconds_per_photo: float = 5.0) -> Path:
    """15-second 9:16 MP4 slideshow from the 3 generated photos. No audio,
    no avatar — just the product images sliding through. Drop into TikTok /
    Reels / Shorts as-is.

    Implementation: one ffmpeg invocation that loops each PNG for N seconds,
    pads to canvas, then concats. Crossfade between slides intentionally
    omitted — adds filter-graph complexity for marginal aesthetic gain."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH — required for promo video")
    if not photo_paths:
        raise ValueError("no photos provided")

    inputs: list[str] = []
    for p in photo_paths:
        inputs += ["-loop", "1", "-t", str(seconds_per_photo), "-i", str(p)]

    parts = []
    for i in range(len(photo_paths)):
        parts.append(
            f"[{i}:v]scale={CANVAS_W}:{CANVAS_H}:force_original_aspect_ratio=decrease,"
            f"pad={CANVAS_W}:{CANVAS_H}:(ow-iw)/2:(oh-ih)/2:white,"
            f"setsar=1[v{i}]"
        )
    concat_chain = "".join(f"[v{i}]" for i in range(len(photo_paths)))
    parts.append(f"{concat_chain}concat=n={len(photo_paths)}:v=1:a=0[outv]")
    filter_complex = ";".join(parts)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-r", "25", "-movflags", "+faststart",
        str(out_path),
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        msg = (stderr or b"").decode(errors="replace")[:400]
        raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}): {msg}")
    return out_path


# ── Orchestrator ────────────────────────────────────────────────────────────


async def build_all(frame_b64: str, product_data: dict,
                    include_3d: bool = True) -> dict:
    """Run the full CREATOR pipeline: 3 photos + 1 promo video + (optional)
    3D model. Outputs go to backend/renders/creator/<request_id>/, returns
    a dict of /renders-relative URLs ready for the dashboard.

    If 3D fails (TripoSR pod unreachable / timeout), the error is surfaced
    in `model_3d.error` but the rest of the build still succeeds. Photos
    + promo are the demo-critical pieces."""
    request_id = uuid.uuid4().hex[:12]
    out_dir = RENDERS_ROOT / request_id

    t0 = time.perf_counter()

    # Photo generation is CPU-bound (rembg + PIL); offload from the event loop.
    photo_paths = await asyncio.to_thread(
        generate_product_photos, frame_b64, product_data, out_dir
    )
    t_photos = time.perf_counter() - t0

    promo_path = out_dir / "promo.mp4"
    await generate_promo_video(photo_paths, promo_path)
    t_promo = time.perf_counter() - t0 - t_photos

    model_3d = None
    if include_3d:
        model_3d = await generate_3d_model(frame_b64)
    t_total = time.perf_counter() - t0

    logger.info(
        "[creator] build_all id=%s photos=%.1fs promo=%.1fs total=%.1fs",
        request_id, t_photos, t_promo, t_total,
    )

    base_url = f"/renders/creator/{request_id}"
    return {
        "request_id": request_id,
        "photos": [f"{base_url}/{p.name}" for p in photo_paths],
        "promo_video": f"{base_url}/{promo_path.name}",
        "model_3d": model_3d,
        "timing_ms": {
            "photos": int(t_photos * 1000),
            "promo": int(t_promo * 1000),
            "total": int(t_total * 1000),
        },
    }
