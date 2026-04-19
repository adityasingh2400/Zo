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
import concurrent.futures
import hashlib
import io
import logging
import subprocess
import threading
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


# ── Parallel rembg infrastructure ────────────────────────────────────────────
# rembg uses onnxruntime under the hood; sessions are NOT thread-safe to share
# across concurrent calls. Solution: one thread pool, one session per thread
# (thread-local), reused across requests so the ~170MB ONNX load amortizes.
#
# 4 workers is the sweet spot: ~700MB total RAM for u2net sessions, 4x faster
# than serial without saturating an M-series CPU. Adjust REMBG_MAX_WORKERS if
# you're memory-constrained or have a beefier machine.

REMBG_MAX_WORKERS = 4

# Detect best ONNX provider once at import time. CoreML on Apple Silicon
# gives ~5-10x speedup for u2net inference vs CPU. Falls back to CPU if not
# available (Linux, Intel Macs, missing onnxruntime-coreml package).
def _detect_rembg_providers() -> list[str]:
    try:
        import onnxruntime as ort
        avail = set(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]
    preferred = []
    if "CoreMLExecutionProvider" in avail:
        preferred.append("CoreMLExecutionProvider")
    if "CUDAExecutionProvider" in avail:
        preferred.append("CUDAExecutionProvider")
    preferred.append("CPUExecutionProvider")
    return preferred


REMBG_PROVIDERS = _detect_rembg_providers()
# CoreML on Apple Silicon is GPU-bound (shared resource); spawning many
# workers thrashes the GPU. CPU-only benefits from full parallelism.
REMBG_WORKERS_EFFECTIVE = 2 if "CoreMLExecutionProvider" in REMBG_PROVIDERS else REMBG_MAX_WORKERS

_REMBG_POOL: concurrent.futures.ThreadPoolExecutor | None = None
_REMBG_POOL_LOCK = threading.Lock()
_REMBG_THREAD_LOCAL = threading.local()


def _get_rembg_pool() -> concurrent.futures.ThreadPoolExecutor:
    """Module-level pool. Lazy init so import-time cost stays zero."""
    global _REMBG_POOL
    if _REMBG_POOL is None:
        with _REMBG_POOL_LOCK:
            if _REMBG_POOL is None:
                _REMBG_POOL = concurrent.futures.ThreadPoolExecutor(
                    max_workers=REMBG_WORKERS_EFFECTIVE,
                    thread_name_prefix="rembg",
                )
                logger.info(
                    "[rembg pool] initialized: %d workers, providers=%s",
                    REMBG_WORKERS_EFFECTIVE, REMBG_PROVIDERS,
                )
    return _REMBG_POOL


async def prewarm_rembg(model_name: str = "u2net") -> dict[str, Any]:
    """Pre-warm the rembg pool by running a tiny dummy image through every
    worker. Pays the CoreML kernel compile cost ONCE at startup so the
    first real user upload doesn't eat 30+ seconds.

    Call once from main.py at app startup. Idempotent — calling again is
    a no-op-ish ~50ms tap to confirm warmth.
    """
    t0 = time.perf_counter()
    pool = _get_rembg_pool()
    loop = asyncio.get_running_loop()
    # Tiny synthetic image: a gray rectangle on a black bg, large enough that
    # rembg actually runs (some models reject under-128px input).
    dummy = Image.new("RGB", (256, 256), (32, 32, 32))
    for x in range(80, 176):
        for y in range(80, 176):
            dummy.putpixel((x, y), (200, 200, 200))
    # Submit one warm call per worker so each thread compiles its kernel.
    await asyncio.gather(*[
        loop.run_in_executor(pool, _rembg_worker, dummy.copy(), model_name)
        for _ in range(REMBG_WORKERS_EFFECTIVE)
    ])
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    logger.info("[rembg pool] prewarmed %d workers in %dms (model=%s)",
                REMBG_WORKERS_EFFECTIVE, elapsed_ms, model_name)
    return {"workers": REMBG_WORKERS_EFFECTIVE, "providers": REMBG_PROVIDERS,
            "model": model_name, "ms": elapsed_ms}


def _keep_central_component(
    rgba: Image.Image,
    alpha_threshold: int = 16,
    min_size_pct: float = 0.005,
) -> Image.Image:
    """Keep only the most product-likely connected component in the alpha mask.

    Use case: when you shoot a product sitting on a notebook / lazy susan /
    stand, rembg captures BOTH the product and the prop as foreground. The
    prop pollutes the bbox, smearing the per-frame center across the spin.
    This pass scores every connected component in the alpha by:
      - centrality (closer to image center = higher score; user-framed product)
      - compactness (closer to square bbox = higher score; props are elongated)
      - size penalty (giant blobs lose points; notebooks / backgrounds are huge)

    Returns an RGBA with only the winning component's alpha; everything else
    becomes transparent. Use this when your shoot has the product on a stand
    or held by hand — toggle off when you've got a clean shoot with the
    product alone in frame.

    Conservative — if no component scores well (e.g. fragmented mask), returns
    the original RGBA untouched.
    """
    arr = np.asarray(rgba)
    if arr.shape[2] < 4:
        return rgba
    alpha = arr[:, :, 3]
    binary = (alpha > alpha_threshold).astype(np.uint8)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8,
    )
    if n_labels <= 1:
        return rgba  # nothing foreground; skip

    h, w = alpha.shape
    img_cx, img_cy = w / 2.0, h / 2.0
    img_area = float(w * h)
    max_dist = float((img_cx ** 2 + img_cy ** 2) ** 0.5)
    min_pixels = int(img_area * min_size_pct)

    best_label = -1
    best_score = -float("inf")
    for lbl in range(1, n_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_pixels:
            continue
        cx, cy = float(centroids[lbl, 0]), float(centroids[lbl, 1])
        dist = ((cx - img_cx) ** 2 + (cy - img_cy) ** 2) ** 0.5
        centrality = 1.0 - (dist / max_dist) if max_dist > 0 else 0.0
        bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        compactness = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0.0
        size_norm = area / img_area
        # Penalize blobs that take up >30% of frame — likely notebook / wall.
        size_penalty = 0.0 if size_norm < 0.30 else (size_norm - 0.30) * 3.0
        score = centrality * 2.5 + compactness * 1.0 - size_penalty
        if score > best_score:
            best_score = score
            best_label = lbl

    if best_label < 0:
        return rgba  # nothing scored; safest to leave alone

    keep = (labels == best_label)
    new_alpha = np.where(keep, alpha, 0).astype(np.uint8)
    out = arr.copy()
    out[:, :, 3] = new_alpha
    return Image.fromarray(out, mode="RGBA")


def _subtract_skin_from_alpha(rgba: Image.Image) -> Image.Image:
    """Find skin-tone pixels in an RGBA image and zero them out of alpha.

    Use case: when shooting a held product (watch in hand), rembg's u2net
    treats hand+product as one foreground blob. This pass detects skin
    pixels via HSV thresholding and removes them, leaving only the product.

    Limitations:
      - HSV skin ranges cover most fair-to-medium tones reliably; very dark
        or very pale skin may not match. The user can disable this and use
        a glove or stand instead.
      - Tan leather watch bands overlap with skin tones — DO NOT enable for
        leather-band products.
      - Yellow/warm room lighting shifts skin into a wider range; we use
        a permissive threshold and clean up with morphology.
    """
    arr = np.asarray(rgba).copy()
    rgb_bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # Skin range: low H (red-orange-yellow), moderate S, mid-high V.
    # OpenCV's H is 0..179.
    skin = ((h <= 25) | (h >= 165)) & (s >= 30) & (s <= 175) & (v >= 60)
    # Cleanup: open (erode then dilate) kills isolated skin-detection noise.
    skin_u8 = (skin.astype(np.uint8)) * 255
    kernel = np.ones((3, 3), np.uint8)
    skin_clean = cv2.morphologyEx(skin_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    # Erode the alpha by a few pixels at the skin boundary so we don't leave
    # skin-tinted halos around the kept product. Then zero alpha where skin.
    alpha = arr[:, :, 3]
    alpha = np.where(skin_clean > 0, 0, alpha).astype(np.uint8)
    arr[:, :, 3] = alpha
    return Image.fromarray(arr, mode="RGBA")


def _rembg_worker(
    img: Image.Image,
    model_name: str,
    rembg_max_edge: int = 1024,
    remove_skin: bool = False,
    keep_central: bool = True,
) -> tuple[Image.Image, tuple[int, int, int, int], float]:
    """Thread-pool worker: rembg one frame, return (rgba_full_res, bbox, coverage).

    Optimization: rembg's u2net model resizes input to 320x320 internally, so
    sending 4K source data is pure preprocessing waste. We:
      1. Downscale the source to `rembg_max_edge` (default 1024) for the call.
      2. Run rembg on the small image to get a small alpha mask.
      3. Upscale ONLY the alpha mask back to source resolution and apply to
         the original full-res RGB. The output is full-res with a smooth alpha.

    Net effect on 4K phone video: ~4x faster rembg, no visible quality loss
    (alpha edges are intrinsically smooth, upscaling them is fine).
    """
    from rembg import new_session, remove

    if not hasattr(_REMBG_THREAD_LOCAL, "sessions"):
        _REMBG_THREAD_LOCAL.sessions = {}
    sessions = _REMBG_THREAD_LOCAL.sessions
    if model_name not in sessions:
        try:
            sessions[model_name] = new_session(model_name, providers=REMBG_PROVIDERS)
        except Exception as e:
            logger.warning("[rembg pool] model %s unavailable, falling back: %s", model_name, e)
            sessions[model_name] = new_session("u2netp", providers=REMBG_PROVIDERS)
    sess = sessions[model_name]

    full_w, full_h = img.size
    if max(full_w, full_h) > rembg_max_edge:
        scale = rembg_max_edge / max(full_w, full_h)
        small = img.resize(
            (max(1, int(full_w * scale)), max(1, int(full_h * scale))),
            Image.LANCZOS,
        )
        small_rgba = remove(small, session=sess)
        # Extract alpha, upscale to full res, apply to original full-res RGB.
        small_alpha = small_rgba.split()[-1]
        full_alpha = small_alpha.resize((full_w, full_h), Image.LANCZOS)
        rgba = img.convert("RGBA")
        rgba.putalpha(full_alpha)
    else:
        rgba = remove(img, session=sess)

    # Optional skin-tone subtraction — kills hand pixels when product is held.
    if remove_skin:
        rgba = _subtract_skin_from_alpha(rgba)

    # Connected-component filter — keeps only the most product-likely blob.
    # Drops notebook / lazy-susan / stand props that rembg captured alongside.
    # Conservative: returns input unchanged if it can't confidently pick.
    if keep_central:
        rgba = _keep_central_component(rgba)

    bbox, coverage = _alpha_bbox(rgba)
    return rgba, bbox, coverage


# ── Tier 1: video → angle carousel ───────────────────────────────────────────


async def carousel_from_video(
    video_path: str,
    *,
    n_frames: int = 48,
    out_size: int = 1024,
    clean_bg: bool = True,
    rembg_model: str = "u2net",
    stabilize: bool = True,
    drop_blurriest_pct: float = 0.20,
    min_coverage: float = 0.01,
    min_sharpness: float = 0.0,
    smooth_window: int = 7,
    remove_skin: bool = False,
    keep_central: bool = True,
    n_heroes: int = 4,
    hero_size: int = 1536,
) -> dict[str, Any]:
    """Extract N angle frames spread across the video and produce a
    polished spin-ready set under /renders/spin/<hash>/.

    Quality steps (in order):
      1. ffmpeg extracts 3x-density candidates so we have headroom to drop bad ones.
      2. _pick_sharpest_per_slot picks the sharpest candidate per timeline slot.
      3. rembg cuts the background; we use 'u2net' by default (~170MB, much
         cleaner edges than u2netp) when available, falling back gracefully.
      4. STABILIZATION (per-frame center, median size):
         every frame's alpha mask gives us the product center + size. We
         pick the median size across the whole sequence and crop each frame
         centered on its OWN product, using that shared size. Out-of-bounds
         is filled with transparent (no garbage edges). Result:
           - camera shake doesn't shift the product within the crop
           - distance variation is normalized to one apparent size
           - one outlier frame can't poison the whole carousel (median is robust)
         Optimal for handheld camera-orbit videos.
      5. Frames whose product coverage is below `min_coverage` (% of pixels
         with alpha > 0) or sharpness is below `min_sharpness` get dropped.

    Args:
      n_frames: target frame count after dropping bad ones (default 24).
      out_size: output square edge in px (default 1024 — texture cap on
                most GPUs, looks pristine when upscaled by the WebGL shader).
      rembg_model: "u2net" (sharper, ~170MB) or "u2netp" (smaller, faster).
      stabilize: enable per-frame-center crop. Disable to debug raw rembg.
      min_sharpness: ABSOLUTE Laplacian-variance floor. Default 0 = disabled.
                     Sharpness is video-relative: 4K HEVC handheld phone
                     video has values of 2-30, while a tripod DSLR shoots
                     200-1000+. An absolute threshold over-rejects on the
                     former and under-rejects on the latter. Use
                     drop_blurriest_pct instead — it adapts per-upload.
      drop_blurriest_pct: drop the bottom N% of picks by sharpness (0.0-0.5).
                          Defaults to 0.20 (drop the worst 20%). Robust to
                          phone-vs-DSLR variation; never starves the carousel.
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
    cached = sorted(p for p in out_dir.glob("[0-9]*.png"))
    cached_heroes = sorted(p.name for p in out_dir.glob("hero_*.png"))
    if len(cached) >= n_frames:
        urls = [f"/renders/spin/{slug}/{p.name}" for p in cached[:n_frames]]
        hero_urls = [f"/renders/spin/{slug}/{name}" for name in cached_heroes]
        ms = int((time.perf_counter() - t_all) * 1000)
        logger.info("carousel cache hit: %s (%d frames, %d heroes, %dms)",
                    slug, len(urls), len(hero_urls), ms)
        return {
            "kind": "frame_carousel", "frames": urls, "heroes": hero_urls,
            "ms": ms, "source": "video", "slug": slug, "cached": True,
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
            from rembg import new_session
            session = new_session(rembg_model)
        except Exception as e:
            logger.warning("rembg %s unavailable, trying u2netp: %s", rembg_model, e)
            try:
                from rembg import new_session
                session = new_session("u2netp")
            except Exception as e2:
                logger.warning("rembg unavailable entirely: %s", e2)
                session = None

    # 4. Pre-compute sharpness for every pick so we can do percentile-based
    #    rejection that adapts to whatever camera the user shot with. Absolute
    #    thresholds over-reject on phone HEVC and under-reject on DSLR raw.
    t0 = time.perf_counter()
    pick_imgs: list[tuple[int, Image.Image, float]] = []
    for i, jpeg in enumerate(picks):
        try:
            img = Image.open(io.BytesIO(jpeg)).convert("RGB")
            pick_imgs.append((i, img, _img_sharpness(img)))
        except Exception as e:
            logger.warning("frame %d decode failed: %s", i, e)

    if not pick_imgs:
        return {"kind": "frame_carousel", "frames": [], "ms": 0,
                "source": "video", "error": "all_decodes_failed"}

    # Percentile cutoff = drop bottom drop_blurriest_pct of picks. Apply the
    # absolute floor on top (defaults to 0 = disabled). Always keep at least
    # n_frames to avoid starving the carousel on a uniformly-noisy clip.
    sharps = sorted(s for _, _, s in pick_imgs)
    pct_cutoff = sharps[max(0, int(len(sharps) * drop_blurriest_pct) - 1)] if sharps else 0.0
    cutoff = max(pct_cutoff, min_sharpness)
    survivors = [p for p in pick_imgs if p[2] > cutoff]
    if len(survivors) < n_frames:
        # Not enough survivors — fall back to the top-N by sharpness so we
        # always emit a full carousel even if the shoot was uniformly soft.
        survivors = sorted(pick_imgs, key=lambda p: -p[2])[:n_frames]
    sharpness_dropped = len(pick_imgs) - len(survivors)

    # 5. rembg + bbox on the survivors. PARALLEL across `REMBG_MAX_WORKERS`
    #    threads so we can render twice the frame count in roughly the same
    #    wall-clock time as the old serial loop.
    frame_records: list[dict[str, Any]] = []
    coverage_dropped = 0
    if session is None:
        # No rembg available — fast path, no parallelism needed.
        for i, img, sharp in survivors:
            rgba = img.convert("RGBA")
            frame_records.append({
                "idx": i, "rgba": rgba, "bbox": (0, 0, img.width, img.height),
                "coverage": 1.0, "sharpness": sharp,
            })
    else:
        loop = asyncio.get_running_loop()
        pool = _get_rembg_pool()

        async def _process_one(idx, img, sharp):
            try:
                # Use functools.partial since run_in_executor only accepts
                # positional args — we need to pass keyword args to the worker.
                from functools import partial
                worker = partial(
                    _rembg_worker, img, rembg_model,
                    rembg_max_edge=1024,
                    remove_skin=remove_skin,
                    keep_central=keep_central,
                )
                rgba, bbox, coverage = await loop.run_in_executor(pool, worker)
                return {"ok": True, "idx": idx, "rgba": rgba, "bbox": bbox,
                        "coverage": coverage, "sharpness": sharp}
            except Exception as e:
                logger.warning("frame %d rembg failed: %s", idx, e)
                return {"ok": False, "idx": idx, "error": str(e)}

        results = await asyncio.gather(*[
            _process_one(i, img, sharp) for i, img, sharp in survivors
        ])
        # Preserve original order so the carousel rotates monotonically.
        results.sort(key=lambda r: r["idx"])
        for r in results:
            if not r.get("ok"):
                continue
            if r["coverage"] < min_coverage:
                logger.debug("frame %d dropped (coverage=%.3f < %.3f)",
                             r["idx"], r["coverage"], min_coverage)
                coverage_dropped += 1
                continue
            frame_records.append({
                "idx": r["idx"], "rgba": r["rgba"], "bbox": r["bbox"],
                "coverage": r["coverage"], "sharpness": r["sharpness"],
            })

    if not frame_records:
        return {"kind": "frame_carousel", "frames": [], "ms": 0,
                "source": "video", "error": "all_frames_dropped",
                "stats": {"sharpness_dropped": sharpness_dropped,
                          "coverage_dropped": coverage_dropped,
                          "candidates": len(picks),
                          "sharpness_cutoff": round(cutoff, 2)}}

    # 5. Build per-frame crop boxes. Each frame is centered on ITS OWN product
    #    using a shared crop size (median of per-frame sizes, padded). This is
    #    robust to handheld camera-orbit shake and distance drift.
    img_w = frame_records[0]["rgba"].width
    img_h = frame_records[0]["rgba"].height
    if stabilize:
        crops, canvas_side, orbit_stats = _build_centered_crops(
            [r["bbox"] for r in frame_records],
            pad_pct=0.12,
            img_w=img_w,
            img_h=img_h,
            smooth_window=smooth_window,
        )
    else:
        crops = [None] * len(frame_records)
        canvas_side = 0
        orbit_stats = {"center_stddev_px": 0.0, "size_stddev_px": 0.0,
                       "median_side_px": 0, "clipped_frames": 0,
                       "raw_center_stddev_px": 0.0, "smooth_window": 0}

    # 6. Crop to per-frame center, resize, save. Sequence-renumbered so the
    #    dashboard sees a clean 0..N-1 set even if some inputs were dropped.
    saved: list[str] = []
    clipped_frames = 0
    for j, (rec, crop_box) in enumerate(zip(frame_records, crops)):
        out_path = out_dir / f"{j:02d}.png"
        rgba = rec["rgba"]
        try:
            if crop_box is not None:
                if rgba.mode != "RGBA":
                    rgba = rgba.convert("RGBA")
                rgba, did_clip = _safe_crop_rgba(rgba, crop_box)
                if did_clip:
                    clipped_frames += 1
            rgba = _square_resize_rgba(rgba, out_size)
            rgba.save(out_path, format="PNG", optimize=False)
            saved.append(f"/renders/spin/{slug}/{out_path.name}")
        except Exception as e:
            logger.warning("frame %d save failed: %s", j, e)
    orbit_stats["clipped_frames"] = clipped_frames

    process_ms = int((time.perf_counter() - t0) * 1000)

    # ── Hero pass ──────────────────────────────────────────────────────────
    # Pick N angle-diverse, high-scoring frames and re-save them at higher
    # resolution. These are the "magazine shots" — meant to be displayed in a
    # gallery alongside the carousel. Reuses existing full-res RGBA so cost
    # is just PIL crop + resize per hero (~300ms total for 4).
    t_hero = time.perf_counter()
    hero_urls: list[str] = []
    hero_meta: list[dict[str, Any]] = []
    if n_heroes > 0 and frame_records:
        try:
            from PIL import ImageFilter
            hero_indexes = _pick_diverse_heroes(frame_records, n_heroes=n_heroes)
            for j, idx in enumerate(hero_indexes):
                rec = frame_records[idx]
                # Use the same per-frame crop the carousel used at this index.
                crop_box = crops[idx] if (stabilize and idx < len(crops)) else None
                rgba = rec["rgba"]
                if rgba.mode != "RGBA":
                    rgba = rgba.convert("RGBA")
                if crop_box is not None:
                    rgba, _ = _safe_crop_rgba(rgba, crop_box)
                # Hero output = larger square. Lanczos upscale + unsharp mask
                # for that "magazine print" crispness. Kept transparent so the
                # frontend can render its own white card / drop shadow CSS.
                hero_img = _square_resize_rgba(rgba, hero_size)
                hero_img = hero_img.filter(
                    ImageFilter.UnsharpMask(radius=1.4, percent=120, threshold=2)
                )
                out_path = out_dir / f"hero_{j:02d}.png"
                hero_img.save(out_path, format="PNG", optimize=False)
                hero_urls.append(f"/renders/spin/{slug}/{out_path.name}")
                hero_meta.append({
                    "index": int(idx),
                    "angle_deg": round(360.0 * idx / max(1, len(frame_records)), 1),
                    "sharpness": round(rec["sharpness"], 2),
                    "coverage": round(rec["coverage"], 4),
                })
        except Exception as e:
            logger.warning("hero render failed: %s", e)
    hero_ms = int((time.perf_counter() - t_hero) * 1000)
    total_ms = int((time.perf_counter() - t_all) * 1000)

    logger.info(
        "carousel built: %s, %d frames + %d heroes (kept %d/%d, dropped "
        "sharp=%d cov=%d, cutoff=%.1f), %dms (extract=%d pick=%d "
        "process=%d hero=%d, model=%s, stabilize=%s, median_side=%dpx, "
        "center_drift=%.1fpx, size_drift=%.1fpx, clipped=%d)",
        slug, len(saved), len(hero_urls), len(frame_records), len(picks),
        sharpness_dropped, coverage_dropped, cutoff,
        total_ms, extract_ms, pick_ms, process_ms, hero_ms,
        rembg_model, stabilize, orbit_stats["median_side_px"],
        orbit_stats["center_stddev_px"], orbit_stats["size_stddev_px"],
        orbit_stats["clipped_frames"],
    )

    return {
        "kind": "frame_carousel",
        "frames": saved,
        "heroes": hero_urls,
        "hero_meta": hero_meta,
        "ms": total_ms,
        "source": "video",
        "slug": slug,
        "cached": False,
        "timings": {
            "extract_ms": extract_ms,
            "pick_ms": pick_ms,
            "process_ms": process_ms,
            "hero_ms": hero_ms,
        },
        "stats": {
            "candidates": len(picks),
            "kept": len(frame_records),
            "dropped": len(picks) - len(frame_records),
            "sharpness_dropped": sharpness_dropped,
            "coverage_dropped": coverage_dropped,
            "sharpness_cutoff": round(cutoff, 2),
            "rembg_model": rembg_model if session is not None else None,
            "stabilized": stabilize,
            # Orbit/shoot quality signals — useful for diagnosing bad shoots.
            #   center_stddev_px : how much the product center moved frame-to-frame.
            #     Low (<20px) = item was stationary OR camera orbited steadily.
            #     High (>80px) = handheld jitter; per-frame centering hides it.
            #   size_stddev_px : variation in product apparent size across frames.
            #     Low = consistent camera distance.
            #     High = you walked closer/farther; median-size crop normalizes.
            #   median_side_px : the crop edge applied to every frame.
            #   clipped_frames : how many frames had the crop extend outside the
            #     image (filled with transparent). >25% suggests the product is
            #     too close to the frame edge — pull camera back.
            "orbit": orbit_stats,
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


def _pick_diverse_heroes(
    frame_records: list[dict[str, Any]],
    n_heroes: int = 4,
) -> list[int]:
    """Pick N hero-worthy frames spread across the rotation.

    Divides the sequence into N equal arc quadrants. Within each quadrant,
    picks the frame with the highest composite score:
      - sharpness (clearer edges read as more "professional")
      - compactness of the alpha bbox (closer to square = product faces
        camera; elongated = side/profile angle which usually looks worse)
      - coverage (more of the product visible = better hero shot)

    Returns a list of indexes into frame_records, one per quadrant. If
    fewer than n_heroes frames available, returns what we have.
    """
    n_total = len(frame_records)
    if n_total == 0:
        return []
    n = min(n_heroes, n_total)

    def score(rec: dict[str, Any]) -> float:
        bbox = rec["bbox"]
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        compactness = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0.0
        # Normalize sharpness to roughly 0-1 range (we've seen 5-30 in HEVC).
        sharp_norm = min(1.0, rec["sharpness"] / 25.0)
        return sharp_norm * 1.0 + compactness * 1.5 + rec["coverage"] * 0.5

    chunk = n_total / n
    chosen = []
    for q in range(n):
        lo = int(q * chunk)
        hi = int((q + 1) * chunk) if q < n - 1 else n_total
        if lo >= hi:
            continue
        # Find the best-scoring record in this quadrant by index.
        local = max(range(lo, hi), key=lambda i: score(frame_records[i]))
        chosen.append(local)
    return chosen


def _sliding_median(values: list[float], window: int) -> list[float]:
    """1D sliding-median filter. Window of N takes the median of N consecutive
    samples centered on each output. Robust to single-frame outliers — one
    bad rembg mask can't yank the apparent product center."""
    n = len(values)
    if n == 0 or window < 2:
        return list(values)
    half = window // 2
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = sorted(values[lo:hi])
        out.append(chunk[len(chunk) // 2])
    return out


def _build_centered_crops(
    bboxes: list[tuple[int, int, int, int]],
    pad_pct: float,
    img_w: int,
    img_h: int,
    smooth_window: int = 5,
) -> tuple[list[tuple[int, int, int, int]], int, dict[str, Any]]:
    """Per-frame-center crops with a shared trimmed-mean size and temporally
    smoothed centers.

    Pipeline:
      1. Compute raw bbox center + size for each frame.
      2. SLIDING-MEDIAN smooth the centers (window=5 default). This is the
         critical step for orbit videos: rembg's per-frame confidence
         varies, so the raw center jitters by 50-100px between adjacent
         frames especially at edge-on angles. The median window kills
         that without lagging.
      3. Pick one crop SIZE = trimmed-mean of per-frame longer-edge,
         padded. Robust to one-frame outliers blowing up the canvas.
      4. CLAMP each smoothed center so the crop never extends past the
         source image — guarantees no transparent edges and no clipped
         frames in the output.

    Returns (per_frame_boxes, canvas_side, stats).
    """
    empty_stats = {"center_stddev_px": 0.0, "size_stddev_px": 0.0,
                   "median_side_px": 0, "clipped_frames": 0,
                   "raw_center_stddev_px": 0.0, "smooth_window": smooth_window}
    if not bboxes:
        return [], 0, empty_stats

    ws = [b[2] - b[0] for b in bboxes]
    hs = [b[3] - b[1] for b in bboxes]
    cxs_raw = [(b[0] + b[2]) / 2 for b in bboxes]
    cys_raw = [(b[1] + b[3]) / 2 for b in bboxes]

    # 1. Compute crop side from per-frame max-edge.
    # Pure MEDIAN (not trimmed mean) is the robust choice when a few frames
    # have polluted bboxes from background props (notebook, hand, etc).
    # If 60% of frames have just-the-product, median = product size.
    sides = sorted(max(w, h) for w, h in zip(ws, hs))
    n = len(sides)
    base_side = sides[n // 2] if n > 0 else 0
    side = int(round(base_side * (1.0 + 2.0 * pad_pct)))

    # Cap to MIN(img_w, img_h). Going beyond either dimension means the crop
    # always extends past the source — wasted transparent pixels and clipped
    # output. min() is the right cap; the previous max() was a bug.
    side = max(16, min(side, min(img_w, img_h)))
    half = side / 2

    # 2. Temporal smoothing of centers — the actual fix for the jumpy spin.
    cxs = _sliding_median(cxs_raw, smooth_window) if smooth_window > 1 else list(cxs_raw)
    cys = _sliding_median(cys_raw, smooth_window) if smooth_window > 1 else list(cys_raw)

    # 3. Clamp each smoothed center so the crop stays inside source bounds.
    crops: list[tuple[int, int, int, int]] = []
    for cx, cy in zip(cxs, cys):
        cx_c = max(half, min(img_w - half, cx))
        cy_c = max(half, min(img_h - half, cy))
        l = int(round(cx_c - half))
        t = int(round(cy_c - half))
        crops.append((l, t, l + side, t + side))

    # Stats — diagnostics for shoot quality.
    def _stddev(xs):
        if len(xs) < 2:
            return 0.0
        m = sum(xs) / len(xs)
        return float((sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5)

    raw_drift = round((_stddev(cxs_raw) ** 2 + _stddev(cys_raw) ** 2) ** 0.5, 1)
    smoothed_drift = round((_stddev(cxs) ** 2 + _stddev(cys) ** 2) ** 0.5, 1)
    stats = {
        "center_stddev_px": smoothed_drift,
        "raw_center_stddev_px": raw_drift,
        "smooth_window": smooth_window,
        "size_stddev_px": round((_stddev(ws) ** 2 + _stddev(hs) ** 2) ** 0.5, 1),
        "median_side_px": side,
        "clipped_frames": 0,  # filled in by caller after cropping
    }
    return crops, side, stats


def _safe_crop_rgba(
    src: Image.Image,
    box: tuple[int, int, int, int],
) -> tuple[Image.Image, bool]:
    """Crop `src` to `box`, padding with transparent for any out-of-bounds
    region. Returns (cropped, did_clip).

    PIL's native crop() with out-of-bounds coords has format-dependent fill
    behavior we don't trust. This builds a transparent canvas of the exact
    requested size and pastes the in-bounds intersection, so the result is
    always exactly box-sized with clean transparent edges.
    """
    l, t, r, b = box
    side_w = r - l
    side_h = b - t
    canvas = Image.new("RGBA", (side_w, side_h), (0, 0, 0, 0))

    src_l = max(0, l)
    src_t = max(0, t)
    src_r = min(src.width, r)
    src_b = min(src.height, b)
    if src_l >= src_r or src_t >= src_b:
        return canvas, True  # entirely outside source — all transparent

    region = src.crop((src_l, src_t, src_r, src_b))
    paste_x = src_l - l  # 0 if box is in-bounds on left, positive if l<0
    paste_y = src_t - t
    if region.mode != "RGBA":
        region = region.convert("RGBA")
    canvas.paste(region, (paste_x, paste_y), region)

    did_clip = (l < 0) or (t < 0) or (r > src.width) or (b > src.height)
    return canvas, did_clip


def _global_bbox(bboxes: list[tuple[int, int, int, int]],
                 pad_pct: float,
                 square: bool,
                 img_w: int,
                 img_h: int) -> tuple[int, int, int, int]:
    """Legacy union-bbox stabilizer. Kept for backward compat / debugging.
    The active pipeline uses _build_centered_crops instead, which is more
    robust to handheld camera-orbit shake.

    Union of bboxes (so the product never clips off-frame across the spin),
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
