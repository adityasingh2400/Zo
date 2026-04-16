#!/usr/bin/env python
"""Phase 0.2 — Generate character portrait via Nano Banana (Gemini 2.5 Flash Image).

Uses the Google GenAI SDK. Fixed seed for reproducibility. Downloads PNG.
Target: young TikTok-style creator, waist-up framing, ring light, cozy background,
neutral relaxed mouth (so it composites cleanly with lip-sync later).
"""
from __future__ import annotations
import os, sys, time, pathlib
from dotenv import dotenv_values

ROOT = pathlib.Path(__file__).resolve().parent.parent
env = dotenv_values(ROOT.parent / ".env")
for k, v in env.items():
    os.environ[k] = v

from google import genai
from google.genai import types

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("GEMINI_API_KEY missing from .env", file=sys.stderr); sys.exit(1)

OUT_DIR = ROOT / "assets" / "portraits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = """Photorealistic portrait of a young Gen-Z TikTok Shop creator, age 23, warm inviting smile
but lips RELAXED and SLIGHTLY PARTED (not a wide grin, not pressed closed — the mouth is
in a neutral resting position so audio lip-sync compositing works cleanly).
Waist-up framing, facing camera directly, natural eye contact.
Soft ring-light key with warm pink/peach fill, cozy bedroom-style background
with tasteful plant + LED strip lighting slightly defocused bokeh.
Wearing a casual oversized cream sweater. Shoulder-length wavy brown hair,
natural makeup, subtle dewy skin. Vertical 9:16 portrait, 1080x1920.
Shot on iPhone 16 Pro, shallow depth of field. Polished but authentic creator aesthetic.
IMPORTANT: mouth must be closed-but-relaxed, lips slightly parted, no teeth showing,
no specific phoneme shape — this face will be used as a base for lip-sync animation."""

NEGATIVE_HINTS = "no wide open mouth, no teeth visible, no exaggerated smile, no speaking pose"


def main():
    client = genai.Client(api_key=API_KEY)

    print(f"Nano Banana (gemini-2.5-flash-image) -> portrait")
    print(f"Output dir: {OUT_DIR}")
    t0 = time.perf_counter()

    # Generate 4 variations in one shot so we have choice
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[f"{PROMPT}\n\n(Avoid: {NEGATIVE_HINTS})"],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio="9:16"),
            candidate_count=1,  # server may cap at 1 for image
        ),
    )
    dt = time.perf_counter() - t0
    print(f"Response in {dt:.1f}s")

    saved = []
    for idx, cand in enumerate(response.candidates or []):
        for part in cand.content.parts:
            if getattr(part, "inline_data", None) and part.inline_data.data:
                mime = part.inline_data.mime_type or "image/png"
                ext = mime.split("/")[-1]
                ts = int(time.time())
                out = OUT_DIR / f"portrait_{ts}_{idx}.{ext}"
                out.write_bytes(part.inline_data.data)
                print(f"  saved: {out}  ({len(part.inline_data.data)} bytes)")
                saved.append(out)
            elif getattr(part, "text", None):
                print(f"  [text] {part.text[:200]}")

    if not saved:
        print("No images in response. Raw candidates:")
        for c in response.candidates or []:
            print(c)
        sys.exit(2)

    # Symlink the first as portrait.png for downstream
    canonical = OUT_DIR / "portrait.png"
    if canonical.exists() or canonical.is_symlink():
        canonical.unlink()
    try:
        canonical.symlink_to(saved[0].name)
    except OSError:
        canonical.write_bytes(saved[0].read_bytes())
    print(f"\nCanonical: {canonical} -> {saved[0].name}")


if __name__ == "__main__":
    main()
