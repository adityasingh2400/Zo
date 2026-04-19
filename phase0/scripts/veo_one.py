#!/usr/bin/env python
"""Minimal Veo 3.1 render harness — one call, one clip, one output.

  python veo_one.py --prompt prompts/seg1.txt --image portrait.png --out seg1.mp4

`--prompt` accepts either a path to a .txt file or an inline string.
`--image` defaults to the canonical portrait.
`--out` is the output mp4 path.

Used for ad-hoc bridge clip iteration where a full chain script is overkill.
For multi-segment chains see veo_pitch_chain.py.
"""
from __future__ import annotations
import argparse
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

GCP_PROJECT = os.environ["GCP_PROJECT_ID"]
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
DEFAULT_PORTRAIT = ROOT / "assets" / "portraits" / "portrait.png"


def render(prompt: str, image_path: Path, out_path: Path,
           audio: bool = False, duration: int = 8) -> None:
    client = genai.Client(vertexai=True, project=GCP_PROJECT,
                          location=GCP_LOCATION)
    print(f"[veo] render → {out_path.name} ({duration}s, "
          f"audio={'on' if audio else 'off'})", flush=True)
    t0 = time.perf_counter()
    op = client.models.generate_videos(
        model="veo-3.1-generate-001", prompt=prompt,
        image=types.Image(image_bytes=image_path.read_bytes(),
                          mime_type="image/png"),
        config=types.GenerateVideosConfig(
            aspect_ratio="9:16", duration_seconds=duration,
            resolution="1080p", generate_audio=audio,
        ),
    )
    while not op.done:
        time.sleep(8)
        op = client.operations.get(op)
        elapsed = int(time.perf_counter() - t0)
        print(f"[veo] polling… {elapsed}s", flush=True)
        if elapsed > 600:
            raise TimeoutError("Veo timed out after 10 min")
    if op.error:
        raise RuntimeError(f"Veo error: {op.error}")

    v = op.response.generated_videos[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    v.video.save(str(out_path))
    print(f"[veo] DONE {int(time.perf_counter()-t0)}s → "
          f"{out_path} ({out_path.stat().st_size // 1024} KB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True,
                    help="Inline prompt string OR path to a .txt file.")
    ap.add_argument("--image", default=str(DEFAULT_PORTRAIT),
                    help="Seed image. Defaults to canonical portrait.")
    ap.add_argument("--out", required=True, help="Output mp4 path.")
    ap.add_argument("--audio", action="store_true",
                    help="Let Veo generate native audio (default: silent).")
    ap.add_argument("--duration", type=int, default=8,
                    help="Veo clip length in seconds (max 8). Default 8.")
    args = ap.parse_args()

    prompt = args.prompt
    if Path(prompt).is_file():
        prompt = Path(prompt).read_text()

    render(prompt, Path(args.image), Path(args.out),
           audio=args.audio, duration=args.duration)


if __name__ == "__main__":
    main()
