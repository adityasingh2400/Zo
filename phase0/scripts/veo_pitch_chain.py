#!/usr/bin/env python
"""Generate a multi-segment pitch substrate by chaining Veo 3.1 renders
through their own last neutral frame.

Veo 3.1's hard cap on image_to_video is 8 s per render. To get a clean
~24 s pitch substrate we render N segments and chain each to the previous
through frame conditioning:

  1. Render segment 1 (8 s) using the canonical portrait as the first
     frame. Prompt is the OPENING beat of the pitch arc.

  2. Extract the last ~1 second of frames from segment 1 (typically 24
     frames at 24 fps or 30 at 30 fps). For each, score how similar it
     is to the canonical portrait via mean-squared-error in resized
     grayscale — lower MSE = more "neutral resting pose."

  3. Pick the most-neutral frame, truncate segment 1 at that point, and
     hand THAT frame to Veo as the first frame of segment 2 (the MIDDLE
     beat). Because Veo's image_to_video uses the input image as frame 0,
     segment 2 literally starts from the same pixel content segment 1
     ended on. The concat seam is pixel-identical, no jump.

  4. Repeat for segment 3 (the CLOSING beat).

  5. ffmpeg-concat the truncated segments → one ~22-26 s pitch MP4.

Output: phase0/assets/bridges/pitch/pitch_chained.mp4

Why this beats anchor-pose-matching alone: identical frame at the seam
guarantees zero pose discontinuity. Anchor-pose prompts give Veo a strong
hint but don't guarantee pixel-perfect end-frames.

Usage:
  python phase0/scripts/veo_pitch_chain.py            # render all 3
  python phase0/scripts/veo_pitch_chain.py --dry-run  # show plan only
  python phase0/scripts/veo_pitch_chain.py --keep-segments
                                                      # don't delete
                                                      # intermediate clips

Cost: 3 × Veo 3.1 8s renders ≈ ~$9-15. Wall time ~5-8 min sequential.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
ENV = dotenv_values(ROOT.parent / ".env")
for _k, _v in ENV.items():
    if _v is not None:
        os.environ.setdefault(_k, _v)

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

GCP_PROJECT = os.environ.get("GCP_PROJECT_ID")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
if not GCP_PROJECT:
    print("FATAL: GCP_PROJECT_ID missing from .env", file=sys.stderr)
    sys.exit(1)

PORTRAIT = ROOT / "assets" / "portraits" / "portrait.png"
OUT_DIR = ROOT / "assets" / "bridges" / "pitch"
OUT_FILE = OUT_DIR / "pitch_chained.mp4"
# Persistent work dir for chained segments — kept on disk between
# invocations so the user can review each segment before the next one
# renders. Names start with `_chain_` so the dev/clips gallery skips them.
CHAIN_DIR = OUT_DIR / "_chain"

MODEL = "veo-3.1-generate-001"
DURATION_S = 8
RESOLUTION = "1080p"
ASPECT = "9:16"

# ─────────────────────────────────────────────────────────────────────────────
# The 3-beat pitch arc. Each beat is a separate Veo render. Anchor pose at
# start AND end of every beat — but the chain mechanism means we don't rely
# on the anchor-pose prompt alone for seamless handoff (we pick the actual
# most-neutral frame from the tail of segment N and feed it to N+1).
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """\
The same woman from the reference image stands centered in her warm cozy
bedroom, with the soft string-lights backdrop and ivy garlands identical to
the reference frame. Locked-off camera on a tripod, no zoom, no pan.

CONTINUITY CONSTRAINT (critical — must be obeyed exactly):
  This is a SINGLE CONTINUOUS TAKE — locked-off security-camera-style
  footage of one uninterrupted moment. Do NOT insert any of the following:
    - scene cuts, jump cuts, hard cuts
    - wipe transitions, sliding black bars, sliding color bars
    - dissolves, fades, fade-to-black
    - frame composition changes, crops, zooms, pans
    - flashes, bloom transitions
  The camera position and frame composition stay 100% identical from
  frame 0 to the final frame. The ENTIRE 8-second clip is one
  uninterrupted continuous take of the woman in front of the camera in
  her bedroom — like a webcam recording with no editing, not a
  produced video with cuts.

OPENING POSE (frame 1 — taken from the reference image):
  - Hands relaxed at her waist, fingers gently open
  - Lips gently parted in a soft conversational smile
  - Calm attentive expression, direct eye contact with camera
  - Neutral upright posture

MID-CLIP MOTION (8 second window):
  {mid}

CLOSING POSE (final frame):
  Returns to the EXACT same opening pose — hands at waist, lips gently
  parted in soft conversational smile, calm expression, direct eye contact.
  Hands settle back to waist within the last ~1.5 seconds so the final
  ~30 frames hold the neutral resting pose. This is critical — the next
  segment will be conditioned on a frame from this resting tail.

SPEAKING / MOUTH MOTION:
  Throughout the clip the woman is mid-conversation — speaking softly with
  natural conversational phonemes (mouth opening and closing at a calm
  rhythm, jaw and cheeks have natural speech micro-motion, occasional
  small teeth visibility). NO specific words, NO exaggerated mouth shapes.
  Just natural conversational chat motion.

ENVIRONMENT (locked):
  Identical bedroom set, lighting, wardrobe (cream knit sweater), hair,
  framing. Background ivy garlands and string lights stay in fixed
  positions throughout — no parallax, no relighting.
"""

BEATS = [
    # 1. OPEN — relaxed conversational gesture (avoid commercial-style
    # "presentation" language so Veo doesn't insert a product-cut wipe)
    ("open",
     "Both hands rise smoothly from waist to chest, cup gently as if "
     "cradling something small in front of her, warm engaged smile, "
     "slight relaxed forward lean toward camera. Then hands open outward "
     "in a small flowing gesture before settling back to waist."),

    # 2. MID — descriptive hand motion
    ("mid",
     "Right hand moves in flowing descriptive gestures at chest height — "
     "counts 2-3 fingers, then draws a small shape in the air, with an "
     "occasional open-palm 'and another' motion. Left hand stays mostly "
     "at waist. Confident warm smile. Hands settle back to waist."),

    # 3. CLOSE — warm inviting gesture (avoid sales call-to-action
    # phrasing that might trigger "end-of-spot" cut behaviour)
    ("close",
     "Both hands open outward at chest level in a warm inviting gesture, "
     "slight forward lean with bright welcoming smile, brief small "
     "friendly wave of right hand toward camera (relaxed, not pointing), "
     "then hands settle smoothly back to waist with a closing soft smile."),
]

# Time window inside each non-final segment to scan for the most-neutral
# frame. We pick from the ~6 s mark (NOT the last second of the 8 s clip)
# because Veo's prompted "hands return to waist + hold" leaves the final
# 2 s essentially static — that's wasted footage. Cutting at ~6 s drops
# 2 s of dead air per chained segment from the final stitched pitch.
#
# At 24 fps a 0.4 s window is ~10 frame candidates — enough variation to
# find a settled neutral pose, not so few that we're forced to pick badly.
NEUTRAL_WINDOW_START_S = 5.8
NEUTRAL_WINDOW_END_S = 6.2


def build_prompt(mid: str) -> str:
    return PROMPT_TEMPLATE.format(mid=mid.strip())


def submit_with_retry(client, *, label: str, prompt: str,
                      image_bytes: bytes, max_retries: int = 5):
    """Same retry pattern as veo_bridges_batch.py — backoff on 429/503."""
    delay = 30.0
    for attempt in range(1, max_retries + 1):
        try:
            return client.models.generate_videos(
                model=MODEL, prompt=prompt,
                image=types.Image(image_bytes=image_bytes,
                                  mime_type="image/png"),
                config=types.GenerateVideosConfig(
                    aspect_ratio=ASPECT, duration_seconds=DURATION_S,
                    resolution=RESOLUTION,
                ),
            )
        except Exception as e:
            msg = str(e)
            transient = ("429" in msg or "RESOURCE_EXHAUSTED" in msg
                         or "503" in msg or "currently unavailable" in msg.lower()
                         or "INTERNAL" in msg or "code': 14" in msg
                         or "code': 13" in msg)
            if not transient or attempt == max_retries:
                raise
            print(f"[{label}] transient — retry {attempt}/{max_retries-1} "
                  f"in {int(delay)}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 300.0)


def render_segment(client, label: str, prompt: str, image_bytes: bytes,
                   out_path: Path) -> None:
    """Render one Veo segment, save to out_path. Polls until done."""
    print(f"[{label}] submit Veo {RESOLUTION} {ASPECT} {DURATION_S}s",
          flush=True)
    t0 = time.perf_counter()
    op = submit_with_retry(client, label=label, prompt=prompt,
                           image_bytes=image_bytes)
    print(f"[{label}] submitted ({time.perf_counter()-t0:.1f}s)", flush=True)

    poll = 0
    while not op.done:
        poll += 1
        time.sleep(8)
        op = client.operations.get(op)
        elapsed = time.perf_counter() - t0
        if poll % 3 == 0:
            print(f"[{label}] poll {poll} elapsed={int(elapsed)}s done={op.done}",
                  flush=True)
        if elapsed > 600:
            raise TimeoutError(f"{label} timed out after 10 min")

    if op.error:
        raise RuntimeError(f"{label} Veo error: {op.error}")

    resp = op.response
    videos = (getattr(resp, "generated_videos", None)
              or getattr(resp, "videos", None))
    if not videos and hasattr(resp, "model_dump"):
        raw = resp.model_dump()
        videos = raw.get("generated_videos") or raw.get("videos")
    if not videos:
        raise RuntimeError(f"{label} no videos in response: {resp}")

    v = videos[0]
    video_file = (v.video if hasattr(v, "video")
                  else (v.get("video") if isinstance(v, dict) else None))

    if video_file and hasattr(video_file, "save"):
        try:
            client.files.download(file=video_file)
        except Exception as e:
            print(f"  [{label}] download warning: {e}")
        video_file.save(str(out_path))
    elif isinstance(video_file, dict) and ("uri" in video_file
                                            or "gcsUri" in video_file):
        from google.cloud import storage
        gcs = video_file.get("gcsUri") or video_file["uri"]
        if not gcs.startswith("gs://"):
            raise RuntimeError(f"{label} non-gs URI: {gcs}")
        bn, blob_name = gcs[5:].split("/", 1)
        sclient = storage.Client(project=GCP_PROJECT)
        sclient.bucket(bn).blob(blob_name).download_to_filename(str(out_path))
    else:
        raise RuntimeError(f"{label} unknown video payload: {type(video_file)}")

    total = time.perf_counter() - t0
    size_kb = out_path.stat().st_size // 1024
    print(f"[{label}] DONE {total:.0f}s -> {out_path.name} ({size_kb} KB)",
          flush=True)


def video_meta(path: Path) -> dict:
    """ffprobe wrapper — duration in seconds + frame count + fps."""
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames,r_frame_rate,duration",
        "-count_frames",
        "-of", "json", str(path),
    ])
    import json
    info = json.loads(out)["streams"][0]
    num, den = info["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    nframes = int(info.get("nb_read_frames") or 0)
    dur = float(info.get("duration") or (nframes / fps if fps else 0))
    return {"fps": fps, "frames": nframes, "duration": dur}


def extract_window_frames(video: Path, start_s: float, end_s: float,
                          work_dir: Path) -> list[tuple[Path, float]]:
    """Pull all frames in the [start_s, end_s] window as lossless PNGs.
    Returns list of (frame_path, time_in_video_seconds) tuples in time
    order. Used to find the most-neutral frame inside a target time
    window (e.g. ~6 s mark), not the literal end of the clip."""
    meta = video_meta(video)
    dur = meta["duration"]
    end_s = min(end_s, dur)
    start_s = max(0.0, min(start_s, end_s))
    work_dir.mkdir(parents=True, exist_ok=True)

    pattern = work_dir / "frame_%04d.png"
    # Frame-accurate window: -ss + -to clip the input to [start, end] and
    # ffmpeg writes one PNG per frame in that range. PNG is lossless so
    # the frame we eventually feed back to Veo carries no extra
    # generation loss beyond the original H.264 encode.
    subprocess.check_call([
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start_s:.4f}",
        "-to", f"{end_s:.4f}",
        "-i", str(video),
        "-fps_mode", "passthrough",
        str(pattern),
    ])

    frames: list[tuple[Path, float]] = []
    paths = sorted(work_dir.glob("frame_*.png"))
    n = len(paths)
    if n == 0:
        return []
    # Distribute frame timestamps evenly across the [start, end] window.
    span = max(end_s - start_s, 1e-6)
    for i, p in enumerate(paths):
        t = start_s + (i / max(n - 1, 1)) * span if n > 1 else start_s
        frames.append((p, t))
    return frames


def neutral_score(frame_path: Path, anchor_path: Path) -> float:
    """How "neutral" is this frame? Lower MSE vs the canonical portrait
    in resized grayscale = more neutral. Returns negative MSE so larger
    is better (handy with max())."""
    from PIL import Image
    import numpy as np
    SIZE = (192, 336)  # portrait is 768x1344, this is /4 — captures pose
    f = np.asarray(Image.open(frame_path).convert("L").resize(SIZE),
                   dtype=np.float32)
    a = np.asarray(Image.open(anchor_path).convert("L").resize(SIZE),
                   dtype=np.float32)
    return -float(np.mean((f - a) ** 2))


def trim_video(src: Path, dst: Path, end_s: float) -> None:
    """Truncate src to [0, end_s] without re-encoding. -c copy preserves
    the original quality (no generation loss across segments)."""
    subprocess.check_call([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-t", f"{end_s:.4f}",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(dst),
    ])


def concat_videos(parts: list[Path], dst: Path) -> None:
    """ffmpeg concat demuxer — bit-exact stitching, no re-encode if all
    parts share codec/resolution/fps (which they will, all from Veo)."""
    listfile = dst.parent / "_concat_list.txt"
    listfile.write_text(
        "\n".join(f"file '{p.resolve()}'" for p in parts) + "\n"
    )
    try:
        subprocess.check_call([
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(listfile),
            "-c", "copy",
            str(dst),
        ])
    finally:
        listfile.unlink(missing_ok=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan and the prompts, no API calls.")
    ap.add_argument("--max-segments", type=int, default=len(BEATS),
                    help="Stop after rendering segment N (1, 2, or 3). "
                         f"Default {len(BEATS)} (all). Use --max-segments 1 "
                         "to render just the first beat for review, then "
                         "--max-segments 2 to add the next, etc. "
                         "Idempotent — already-rendered segments on disk "
                         "are skipped.")
    ap.add_argument("--from-scratch", action="store_true",
                    help="Wipe the persistent _chain/ work dir before "
                         "rendering anything. Use this to start over with "
                         "a new prompt template.")
    ap.add_argument("--concat-only", action="store_true",
                    help="Skip rendering; just stitch existing trimmed "
                         "segments from _chain/ into the final pitch_chained.mp4.")
    return ap.parse_args()


def _seg_paths(idx: int, label: str) -> tuple[Path, Path, Path]:
    """For segment idx (1-indexed) with label, return:
      (raw_path, trimmed_path, tail_dir)
    Persisted under CHAIN_DIR so they survive between invocations.
    """
    raw = CHAIN_DIR / f"_chain_seg_{idx}_{label}_raw.mp4"
    trimmed = CHAIN_DIR / f"_chain_seg_{idx}_{label}_trimmed.mp4"
    tail = CHAIN_DIR / f"_chain_seg_{idx}_{label}_tail"
    return raw, trimmed, tail


def main():
    args = parse_args()

    if not PORTRAIT.exists():
        print(f"FATAL: portrait missing at {PORTRAIT}", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.from_scratch and CHAIN_DIR.exists():
        print(f"[from-scratch] wiping {CHAIN_DIR.relative_to(ROOT.parent)}")
        shutil.rmtree(CHAIN_DIR)
    CHAIN_DIR.mkdir(parents=True, exist_ok=True)

    n = max(1, min(args.max_segments, len(BEATS)))
    print(f"Plan: {n} of {len(BEATS)} chained segments, "
          f"{DURATION_S}s each")
    print(f"Work dir: {CHAIN_DIR.relative_to(ROOT.parent)}")
    print(f"Final concat: {OUT_FILE.relative_to(ROOT.parent)}")

    if args.dry_run:
        for i, (label, mid) in enumerate(BEATS[:n], start=1):
            print(f"\n──── segment {i}: {label} ────\n"
                  f"{build_prompt(mid).rstrip()}")
        print(f"\n[dry-run] {n} prompt(s) ready.")
        return

    if args.concat_only:
        trimmed_paths: list[Path] = []
        for i, (label, _mid) in enumerate(BEATS, start=1):
            _, trimmed, _ = _seg_paths(i, label)
            if trimmed.exists():
                trimmed_paths.append(trimmed)
        if not trimmed_paths:
            print("FATAL: no trimmed segments on disk; render some first")
            sys.exit(1)
        print(f"[concat] stitching {len(trimmed_paths)} segments → "
              f"{OUT_FILE.name}")
        concat_videos(trimmed_paths, OUT_FILE)
        meta = video_meta(OUT_FILE)
        print(f"[concat] DONE: {OUT_FILE.relative_to(ROOT.parent)} "
              f"({meta['duration']:.1f}s, {meta['frames']} frames @ "
              f"{meta['fps']:.1f}fps, {OUT_FILE.stat().st_size // 1024} KB)")
        return

    client = genai.Client(vertexai=True, project=GCP_PROJECT,
                          location=GCP_LOCATION)

    # First segment uses the canonical portrait. Every subsequent segment
    # uses the most-neutral tail frame of the previous segment, persisted
    # at CHAIN_DIR/_chain_seg_<i>_<label>_neutral.png.
    next_image_bytes: bytes | None = None
    for i, (label, mid) in enumerate(BEATS[:n], start=1):
        seg_label = f"segment_{i}_{label}"
        raw, trimmed, tail_dir = _seg_paths(i, label)

        # Resolve the input image for this segment
        if i == 1:
            next_image_bytes = PORTRAIT.read_bytes()
        else:
            # Load the previous segment's chosen neutral frame
            prev_label = BEATS[i - 2][0]
            _, _, prev_tail = _seg_paths(i - 1, prev_label)
            picked = prev_tail / "_picked_neutral.png"
            if not picked.exists():
                print(f"FATAL: previous segment's neutral frame missing at "
                      f"{picked}. Re-render segment {i-1} with "
                      f"--max-segments {i-1}.", file=sys.stderr)
                sys.exit(1)
            next_image_bytes = picked.read_bytes()

        # Render unless the raw file is already on disk (idempotent)
        if raw.exists():
            print(f"[{seg_label}] SKIP — already on disk "
                  f"({raw.stat().st_size // 1024} KB)", flush=True)
        else:
            render_segment(client, seg_label, build_prompt(mid),
                           next_image_bytes, raw)

        # For non-final segments, pick the most-neutral frame inside the
        # NEUTRAL_WINDOW_START_S..NEUTRAL_WINDOW_END_S window (~6 s mark)
        # and truncate the raw clip to end at that frame's time. Cutting
        # at ~6 s rather than ~8 s strips dead air the prompt asks Veo to
        # paint (final 1-2 s holding the resting pose), tightening the
        # final stitched pitch by ~6 s across all 3 segments.
        if i < len(BEATS):
            if not trimmed.exists() or not (tail_dir / "_picked_neutral.png").exists():
                if tail_dir.exists():
                    shutil.rmtree(tail_dir)
                window_frames = extract_window_frames(
                    raw, NEUTRAL_WINDOW_START_S, NEUTRAL_WINDOW_END_S,
                    tail_dir,
                )
                if not window_frames:
                    print(f"[{seg_label}] WARN — no frames extracted from "
                          f"[{NEUTRAL_WINDOW_START_S}, "
                          f"{NEUTRAL_WINDOW_END_S}]s window")
                    sys.exit(1)
                scored = [(p, t, neutral_score(p, PORTRAIT))
                          for p, t in window_frames]
                scored.sort(key=lambda x: x[2], reverse=True)
                best_path, best_t, best_score = scored[0]
                print(f"[{seg_label}] picked neutral frame at t={best_t:.2f}s "
                      f"(score={best_score:.0f}; {len(window_frames)} "
                      f"candidates in [{NEUTRAL_WINDOW_START_S:.1f}, "
                      f"{NEUTRAL_WINDOW_END_S:.1f}]s)",
                      flush=True)
                # Persist the picked frame so the next invocation can use it
                shutil.copy2(best_path, tail_dir / "_picked_neutral.png")
                trim_video(raw, trimmed, best_t)
            else:
                print(f"[{seg_label}] SKIP trim — trimmed + neutral on disk")
        else:
            # Final segment — full 8s, just copy raw → trimmed
            if not trimmed.exists():
                shutil.copy2(raw, trimmed)

    # Auto-concat only if we just finished the LAST segment in the BEATS
    # list. Partial runs (--max-segments < N) leave OUT_FILE alone.
    if n == len(BEATS):
        all_trimmed = []
        for i, (label, _mid) in enumerate(BEATS, start=1):
            _, trimmed, _ = _seg_paths(i, label)
            if trimmed.exists():
                all_trimmed.append(trimmed)
        if len(all_trimmed) == len(BEATS):
            print(f"\n[concat] stitching {len(all_trimmed)} segments → "
                  f"{OUT_FILE.name}")
            concat_videos(all_trimmed, OUT_FILE)
            meta = video_meta(OUT_FILE)
            print(f"[concat] DONE: {OUT_FILE.relative_to(ROOT.parent)} "
                  f"({meta['duration']:.1f}s, {meta['frames']} frames @ "
                  f"{meta['fps']:.1f}fps, "
                  f"{OUT_FILE.stat().st_size // 1024} KB)")

    # Tell the user how to view the segments they just rendered
    print(f"\n=== Review URLs (raw 8s clips, untrimmed) ===")
    for i, (label, _mid) in enumerate(BEATS[:n], start=1):
        raw, _, _ = _seg_paths(i, label)
        if raw.exists():
            url = f"http://127.0.0.1:8002/bridges/pitch/_chain/{raw.name}"
            print(f"  segment {i} ({label}): {url}")


if __name__ == "__main__":
    main()
