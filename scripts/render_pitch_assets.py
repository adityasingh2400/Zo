#!/usr/bin/env python
"""Render pitch assets for the audio-first 30s pitch path.

For each product slug we cache:
  - <slug>_pitch.mp3              ElevenLabs TTS audio (Cartesia later)
  - <slug>_pitch_words.json       synthesized word timings [{word,start,end},...]
  - manifest.json                 slug → {audio_url, words_url, video_url, audio_ms, script}

The video_url points at an existing Veo "pitching pose, speaking motion"
clip already on disk (phase0/assets/states/state_pitching_pose_speaking_1080p.mp4
served via /states). The 8-second clip loops underneath the 30s audio
overlay; karaoke captions divert eye gaze from the mouth so the loop is
invisible to the audience (per design doc thesis 3).

Pitch scripts are HAND-AUTHORED per product so demo runs are deterministic
(judges + repeat takes shouldn't see different pitches each time). Add new
products to PITCH_SCRIPTS below; pass --generate to re-author one via
Bedrock if you want a fresh take.

Usage:
    python scripts/render_pitch_assets.py                    # render all
    python scripts/render_pitch_assets.py leather_wallet     # just one
    python scripts/render_pitch_assets.py --generate         # re-author + render
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pitch")

# Hand-authored pitches. ~85-95 words each = ~30 seconds at flash_v2_5's
# rate. Reference specific product details (materials, price, target) so
# the pitch reads as authored, not generic. End on a CTA so the natural
# audio tail bridges into the post-pitch idle layer cleanly.
PITCH_SCRIPTS: dict[str, str] = {
    "leather_wallet": (
        "Okay, this wallet right here. "
        "Full-grain vegetable-tanned leather — the real stuff that gets richer the longer you carry it. "
        "Hand-stitched, RFID blocking, and it holds eight cards plus cash without bulging in your pocket. "
        "Caramel, oxblood, or classic black. "
        "Forty-nine dollars, free US shipping, two-year warranty, thirty-day returns. "
        "If you've been carrying a beat-up bifold from college, this is your upgrade. "
        "Tap the BUY button before they sell out — caramel always goes first."
    ),
}

# Where the cached audio + words + manifest live. Served at /pitch_assets
# by backend/main.py (static mount added in the audio-first refactor).
PITCH_ASSETS_DIR = BACKEND / "pitch_assets"
PITCH_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = PITCH_ASSETS_DIR / "manifest.json"

# The Veo speaking-pose clip we reuse for every pitch. Already at 1080p,
# 9:16, ~8s — loops 4x underneath the 30s audio. Different products MAY
# have different visuals once we add more substrates; for now the wallet
# pitch (the only demo product) uses the universal speaking pose.
DEFAULT_PITCH_VIDEO_URL = "/states/state_pitching_pose_speaking_1080p.mp4"


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text())
    except Exception as e:
        logger.warning("manifest read failed: %s", e)
        return {}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


async def render_one(slug: str, *, generate: bool = False) -> dict:
    """Render audio + word timings for one product. Returns the manifest entry.

    `generate=True` ignores PITCH_SCRIPTS and re-authors via Bedrock using
    seller.generate_sales_script. Useful when the hand-authored copy
    feels stale; cache the new script in PITCH_SCRIPTS afterwards so the
    demo stays deterministic.
    """
    from agents.seller import (
        text_to_speech, _probe_audio_duration_ms, generate_sales_script,
    )

    # Pull the active product's data from products.json so the LLM (and the
    # log) have full context.
    products_path = BACKEND / "data" / "products.json"
    if not products_path.exists():
        raise FileNotFoundError(f"products.json missing at {products_path}")
    products = json.loads(products_path.read_text())
    product = products.get(slug)
    if not product:
        raise ValueError(f"product '{slug}' not found in products.json")

    if generate:
        logger.info("[%s] generating pitch via Bedrock...", slug)
        script = (await generate_sales_script(product, "sell this for $49")).strip()
    else:
        script = PITCH_SCRIPTS.get(slug)
        if not script:
            raise ValueError(
                f"No PITCH_SCRIPTS entry for '{slug}' and --generate not set. "
                f"Either add a hand-authored script or pass --generate."
            )

    logger.info("[%s] script (%d words):", slug, len(script.split()))
    logger.info("    %s", script)

    # TTS — request word timings; we'll cache them alongside the audio so
    # the dashboard can pick them up via the manifest at WS-message time.
    t0 = time.time()
    audio_bytes, word_timings = await text_to_speech(
        script, return_word_timings=True,
    )
    tts_ms = int((time.time() - t0) * 1000)
    if not audio_bytes:
        raise RuntimeError(f"TTS returned empty audio for '{slug}'")
    logger.info("[%s] TTS done in %dms (%d KB, %d words timed)",
                slug, tts_ms, len(audio_bytes) // 1024, len(word_timings))

    audio_path = PITCH_ASSETS_DIR / f"{slug}_pitch.mp3"
    words_path = PITCH_ASSETS_DIR / f"{slug}_pitch_words.json"
    audio_path.write_bytes(audio_bytes)
    words_path.write_text(json.dumps(word_timings, indent=2))

    audio_ms = _probe_audio_duration_ms(audio_bytes) or 0

    entry = {
        "slug": slug,
        "script": script,
        "audio_url": f"/pitch_assets/{audio_path.name}",
        "words_url": f"/pitch_assets/{words_path.name}",
        "audio_ms": audio_ms,
        "video_url": DEFAULT_PITCH_VIDEO_URL,
        # Word timings inlined so the runtime can return the manifest in
        # one round-trip; words_url stays for any external loader.
        "word_timings": word_timings,
        "rendered_at": int(time.time()),
        "tts_ms": tts_ms,
    }

    manifest = load_manifest()
    manifest[slug] = entry
    save_manifest(manifest)
    logger.info("[%s] OK — audio=%s words=%s manifest=%s",
                slug, audio_path.name, words_path.name, MANIFEST_PATH.name)
    return entry


async def render_all(slugs: list[str] | None = None, *, generate: bool = False) -> int:
    if not slugs:
        slugs = list(PITCH_SCRIPTS.keys())
    failed = 0
    for slug in slugs:
        try:
            await render_one(slug, generate=generate)
        except Exception as e:
            logger.exception("[%s] render failed: %s", slug, e)
            failed += 1
    logger.info("=" * 60)
    logger.info("done: rendered=%d failed=%d", len(slugs) - failed, failed)
    logger.info("manifest: %s", MANIFEST_PATH)
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slugs", nargs="*", help="product slug(s) to render; default: all")
    parser.add_argument("--generate", action="store_true",
                        help="re-author the pitch via Bedrock instead of using PITCH_SCRIPTS")
    args = parser.parse_args()

    failed = asyncio.run(render_all(args.slugs or None, generate=args.generate))
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
