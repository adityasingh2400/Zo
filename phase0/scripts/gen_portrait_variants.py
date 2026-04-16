#!/usr/bin/env python
"""Phase 0.2b — Generate 2 alternate portrait variants in parallel.

Variant A: stronger "lips together, no teeth, relaxed closed mouth"
Variant B: alternate look (different hair, different angle) for diversity
"""
from __future__ import annotations
import os, sys, time, pathlib, concurrent.futures
from dotenv import dotenv_values

ROOT = pathlib.Path(__file__).resolve().parent.parent
env = dotenv_values(ROOT.parent / ".env")
for k, v in env.items():
    os.environ[k] = v

from google import genai
from google.genai import types

API_KEY = os.environ["GEMINI_API_KEY"]
OUT_DIR = ROOT / "assets" / "portraits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = {
    "A_closed": (
        "Photorealistic portrait of the SAME young Gen-Z TikTok Shop creator — age 23, "
        "warm wavy brown shoulder-length hair, cream oversized sweater. "
        "MOUTH IS FULLY CLOSED but relaxed (lips softly together, NO teeth visible, "
        "NO smile, lips in their natural resting position — this is for lip-sync animation).\n"
        "Waist-up, facing camera. Soft ring-light, cozy bedroom bokeh with fairy lights + plant. "
        "Vertical 9:16 portrait, 1080x1920. Shot on iPhone 16 Pro.\n"
        "IMPORTANT: lips together, closed mouth, NO teeth, NO open-mouth smile."
    ),
    "B_alt": (
        "Photorealistic portrait of a young Gen-Z TikTok Shop creator, age 25, "
        "friendly approachable vibe. Straight black hair, tan skin, minimal makeup, "
        "wearing a soft pastel hoodie. MOUTH CLOSED-BUT-RELAXED, lips slightly parted at most, "
        "absolutely NO open-mouth smile (this is a lip-sync source image).\n"
        "Waist-up, facing camera directly. Warm key light, cozy loft bedroom with neon accent "
        "lighting, slightly defocused bokeh. Vertical 9:16 portrait, 1080x1920."
    ),
}


def gen(label: str, prompt: str) -> pathlib.Path | None:
    client = genai.Client(api_key=API_KEY)
    t0 = time.perf_counter()
    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio="9:16"),
        ),
    )
    dt = time.perf_counter() - t0
    for cand in resp.candidates or []:
        for part in cand.content.parts:
            if getattr(part, "inline_data", None) and part.inline_data.data:
                ts = int(time.time())
                out = OUT_DIR / f"portrait_{label}_{ts}.png"
                out.write_bytes(part.inline_data.data)
                print(f"[{label}] {dt:.1f}s  -> {out.name}")
                return out
    print(f"[{label}] no image returned in {dt:.1f}s")
    return None


def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(VARIANTS)) as ex:
        futs = {ex.submit(gen, label, prompt): label for label, prompt in VARIANTS.items()}
        for f in concurrent.futures.as_completed(futs):
            f.result()

if __name__ == "__main__":
    main()
