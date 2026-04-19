#!/usr/bin/env python
"""Re-render the Tier 0 / Tier 1 idle clips that shipped as TALKING
substrates so they're silent body-language clips instead.

The bridge library was rendered with talking-substrate prompts (mouth
moving naturally so Wav2Lip can overlay cleanly at runtime). But two
clips that cycle during pure-idle periods — Tier 0 `idle_thinking` and
Tier 1 `misc_glance_aside` — never get a Wav2Lip overlay or audio. The
avatar mouthing words to nobody during idle reads as broken UX.

This script regenerates those two with the SILENT prompt template the
original `state_*_silent_*.mp4` library uses (mouth closed in soft
natural smile, no speech motion). Outputs land next to the existing
clips with `_silent.mp4` suffix; the runtime config (Director +
simulator) gets updated to point at the new files in a follow-up edit.

Usage:
  python phase0/scripts/veo_silent_idle_renders.py            # render both
  python phase0/scripts/veo_silent_idle_renders.py --dry-run  # show prompts
  python phase0/scripts/veo_silent_idle_renders.py --only idle_thinking
"""
from __future__ import annotations
import argparse
import concurrent.futures
import os
import sys
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
IDLE_DIR = ROOT / "assets" / "states" / "idle"

MODEL = "veo-3.1-generate-001"
DURATION_S = 8
RESOLUTION = "1080p"
ASPECT = "9:16"

PROMPT_TEMPLATE = """\
The same woman from the reference image stands centered in her warm cozy
bedroom, with the soft string-lights backdrop and ivy garlands identical to
the reference frame. Locked-off camera on a tripod, no zoom, no pan.

CONTINUITY CONSTRAINT (critical):
  This is a SINGLE CONTINUOUS TAKE — locked-off security-camera-style
  footage of one uninterrupted moment. NO scene cuts, NO wipe transitions,
  NO sliding bars, NO dissolves, NO fades, NO frame composition changes.
  The camera and frame composition stay 100% identical from frame 0 to the
  final frame. The entire 8-second clip is one uninterrupted take.

OPENING POSE (frame 1 — taken from the reference image):
  - Hands relaxed at her waist, fingers gently open
  - Lips closed in a soft natural smile (mouth completely shut, no teeth)
  - Calm attentive expression, direct eye contact with camera
  - Neutral upright posture

MID-CLIP MOTION (8 second window):
  {mid}

CLOSING POSE (final frame):
  Returns to the EXACT same opening pose described above — hands at waist,
  lips closed in soft natural smile, calm expression, direct eye contact.
  Hands settle back to waist within the last ~1 second so the final still
  frame is the neutral resting pose.

LIPS / MOUTH:
  Lips remain CLOSED in a soft natural smile throughout the entire clip.
  ABSOLUTELY NO speech motion, NO mouth opening, NO phonemes, NO teeth
  showing, NO talking. The avatar is silent, breathing naturally, with a
  closed-mouth resting expression. This is a quiet idle moment — she is
  NOT speaking and her mouth never opens for words.

ENVIRONMENT (locked):
  Identical bedroom set, lighting, wardrobe (cream knit sweater), hair,
  framing. Background ivy garlands and string lights stay in fixed
  positions — no parallax, no relighting.
"""

# Each: (label, output filename, mid-clip prompt)
JOBS: list[tuple[str, str, str]] = [
    ("idle_thinking",
     "idle_thinking_silent.mp4",
     "Subtle thoughtful expression — eyes briefly look up and to the "
     "left as if pondering something for half a second, then return to "
     "direct camera contact. Soft natural blinks. Slight head tilt to "
     "the right ~5 degrees and back. Hands stay completely at waist; "
     "fingers may flex very gently once. Mouth stays closed in a soft "
     "smile throughout — she is thinking, not speaking."),
    ("misc_glance_aside",
     "misc_glance_aside_silent.mp4",
     "A brief glance to the side — eyes shift to the right (camera-left "
     "from viewer's perspective) for about 1 second as if something off-"
     "screen briefly caught her attention, then return to direct camera "
     "contact. Subtle head turn ~10 degrees to follow the eyes, then "
     "back. Hands stay completely at waist throughout. Mouth stays closed "
     "in a soft smile throughout — she is glancing, not speaking."),
]


def submit_with_retry(client, *, label: str, prompt: str,
                      image_bytes: bytes, max_retries: int = 5):
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


def render_one(label: str, out_name: str, mid: str, *, overwrite: bool) -> dict:
    out = IDLE_DIR / out_name
    if out.exists() and not overwrite:
        size_kb = out.stat().st_size // 1024
        print(f"[{label}] SKIP — already on disk ({size_kb} KB)", flush=True)
        return {"label": label, "path": str(out), "skipped": True}

    out.parent.mkdir(parents=True, exist_ok=True)
    image_bytes = PORTRAIT.read_bytes()
    prompt = PROMPT_TEMPLATE.format(mid=mid.strip())

    print(f"[{label}] submit Veo {RESOLUTION} {ASPECT} {DURATION_S}s",
          flush=True)
    t0 = time.perf_counter()
    client = genai.Client(vertexai=True, project=GCP_PROJECT,
                          location=GCP_LOCATION)
    op = submit_with_retry(client, label=label, prompt=prompt,
                           image_bytes=image_bytes)
    print(f"[{label}] submitted ({time.perf_counter() - t0:.1f}s)", flush=True)

    poll = 0
    while not op.done:
        poll += 1
        time.sleep(8)
        op = client.operations.get(op)
        elapsed = time.perf_counter() - t0
        if poll % 3 == 0:
            print(f"[{label}] poll {poll} elapsed={int(elapsed)}s "
                  f"done={op.done}", flush=True)
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
    vfile = (v.video if hasattr(v, "video")
             else (v.get("video") if isinstance(v, dict) else None))
    if vfile and hasattr(vfile, "save"):
        try:
            client.files.download(file=vfile)
        except Exception as e:
            print(f"  [{label}] download warning: {e}")
        vfile.save(str(out))
    elif isinstance(vfile, dict) and ("uri" in vfile or "gcsUri" in vfile):
        from google.cloud import storage
        gcs = vfile.get("gcsUri") or vfile["uri"]
        if not gcs.startswith("gs://"):
            raise RuntimeError(f"{label} non-gs URI: {gcs}")
        bn, blob_name = gcs[5:].split("/", 1)
        sclient = storage.Client(project=GCP_PROJECT)
        sclient.bucket(bn).blob(blob_name).download_to_filename(str(out))
    else:
        raise RuntimeError(f"{label} unknown video payload: {type(vfile)}")

    total = time.perf_counter() - t0
    size_kb = out.stat().st_size // 1024
    print(f"[{label}] DONE {total:.0f}s -> {out.name} ({size_kb} KB)",
          flush=True)
    return {"label": label, "path": str(out), "wall_sec": round(total, 1),
            "size_kb": size_kb, "skipped": False}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="",
                    help="Comma-separated labels to render. "
                         f"Default = all ({','.join(j[0] for j in JOBS)}).")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if not PORTRAIT.exists():
        print(f"FATAL: portrait missing at {PORTRAIT}", file=sys.stderr)
        sys.exit(1)

    targets = (
        [s.strip() for s in args.only.split(",") if s.strip()]
        if args.only else [j[0] for j in JOBS]
    )
    todo = [j for j in JOBS if j[0] in targets]
    unknown = set(targets) - {j[0] for j in JOBS}
    if unknown:
        print(f"unknown labels: {unknown}", file=sys.stderr)
        sys.exit(2)

    print(f"Plan: {len(todo)} silent re-render(s)")
    print(f"Output: {IDLE_DIR.relative_to(ROOT.parent)}")

    if args.dry_run:
        for label, out_name, mid in todo:
            print(f"\n──── {label} → {out_name} ────")
            print(PROMPT_TEMPLATE.format(mid=mid.strip()).rstrip())
        return

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(todo)) as ex:
        futs = {ex.submit(render_one, *j, overwrite=args.overwrite): j[0]
                for j in todo}
        for fut in concurrent.futures.as_completed(futs):
            label = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"[{label}] FAILED: {e}")
                results.append({"label": label, "error": str(e)})

    print("\n=== summary ===")
    for r in results:
        if r.get("error"):
            print(f"  {r['label']}: FAILED — {r['error']}")
        elif r.get("skipped"):
            print(f"  {r['label']}: skipped (on disk)")
        else:
            print(f"  {r['label']}: {r['wall_sec']}s, {r['size_kb']}KB")


if __name__ == "__main__":
    main()
