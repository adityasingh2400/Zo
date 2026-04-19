"""Pre-render every qa_index entry in products.json to a Wav2Lip MP4.

For each entry:
  1. ElevenLabs TTS → audio bytes
  2. Wav2Lip /lipsync_fast on the pod → mp4 bytes
  3. Write to backend/local_answers/<slug>.mp4 matching the URL

Idempotent: skips entries whose output file already exists. Pass --force to
re-render all.

Requires the backend's .env (AWS, ElevenLabs, Wav2Lip tunnel) so the script
imports from backend.config + backend.agents.seller to reuse the live clients.
Run with the tunnel open (bash phase0/scripts/open_tunnel.sh) and the Cactus
venv active.

Usage:
  python scripts/render_local_answers.py
  python scripts/render_local_answers.py --force
  python scripts/render_local_answers.py --only wallet_real_leather,wallet_price
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

# Load .env before importing seller so ElevenLabs/RunPod clients pick it up.
from dotenv import load_dotenv  # type: ignore
load_dotenv(ROOT / ".env")

from agents.seller import (  # noqa: E402
    text_to_speech,
    render_comment_response_wav2lip,
    trim_audio_silence,
    pad_wav2lip_video_to_audio,
)
from config import POD_SPEAKING_1080P  # noqa: E402

PRODUCTS_PATH = BACKEND / "data" / "products.json"
OUT_DIR = BACKEND / "local_answers"


def url_to_path(url: str) -> Path:
    # "/local_answers/wallet_real_leather.mp4" -> backend/local_answers/wallet_real_leather.mp4
    name = url.rsplit("/", 1)[-1]
    return OUT_DIR / name


async def render_one(slug: str, text: str, out_path: Path, substrate: str) -> tuple[float, int]:
    """TTS → trim → Wav2Lip → pad → write. Returns (elapsed_s, bytes).

    Mirrors the alignment pipeline in scripts/render_generic_clips.py so
    every per-product Q&A clip gets the same drift fix the bridges got:

      - trim_audio_silence (BEFORE wav2lip): crops head/tail silence from
        the TTS output so wav2lip's mouth predictor only runs on speech
        frames. Avoids the random mouth-flap on silent edges.

      - pad_wav2lip_video_to_audio (AFTER wav2lip): re-mux locally with
        the FULL trimmed audio + video padded by holding the last frame
        for the structural ~120-180ms wav2lip mel-chunking shortfall.
        Without this, the pod's `-shortest` ffmpeg mux truncates the
        audio tail (the trailing word-end / breath gets cut).

    These clips are played from disk at runtime via the respond_locally
    path with their muxed audio. Drift here = audience-perceived drift,
    full stop. With the fix: ±50ms typical (well under the dashboard's
    250ms handshake and well below human-perceptible lipsync mismatch)."""
    t0 = time.perf_counter()
    audio_raw = await text_to_speech(text)
    if not audio_raw:
        raise RuntimeError("TTS returned empty audio (ElevenLabs not configured?)")
    t_tts = time.perf_counter() - t0

    t_trim_start = time.perf_counter()
    audio = trim_audio_silence(audio_raw)
    t_trim = time.perf_counter() - t_trim_start
    trim_pct = (1 - len(audio) / len(audio_raw)) * 100 if audio_raw else 0

    t1 = time.perf_counter()
    mp4_raw, _headers = await render_comment_response_wav2lip(
        audio_bytes=audio,
        source_path_on_pod=substrate,
        out_height=1920,
    )
    t_lip = time.perf_counter() - t1

    t_pad_start = time.perf_counter()
    mp4, pad_diag = pad_wav2lip_video_to_audio(mp4_raw, audio)
    t_pad = time.perf_counter() - t_pad_start

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(mp4)
    total = time.perf_counter() - t0
    drift_after = (
        pad_diag.get("audio_ms", 0) - pad_diag.get("video_ms_after", 0)
        if pad_diag.get("padded") else None
    )
    drift_str = f" drift={drift_after:+d}ms" if drift_after is not None else ""
    print(
        f"  ✓ {slug}: tts={t_tts:.1f}s trim={t_trim:.2f}s({trim_pct:.0f}%) "
        f"lip={t_lip:.1f}s pad={t_pad:.2f}s{drift_str} "
        f"total={total:.1f}s size={len(mp4) / 1024:.0f}KB → {out_path.relative_to(ROOT)}",
        flush=True,
    )
    return total, len(mp4)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-render existing files")
    ap.add_argument("--only", default="", help="comma-separated slug list to render")
    ap.add_argument(
        "--substrate",
        default=POD_SPEAKING_1080P,
        help="pod-side path to source speaking video",
    )
    args = ap.parse_args()

    only_set = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else None

    products = json.loads(PRODUCTS_PATH.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, str, Path]] = []
    for pid, product in products.items():
        qa = product.get("qa_index", {})
        for key, entry in qa.items():
            url = entry.get("url")
            if not url:
                continue
            out_path = url_to_path(url)
            slug = out_path.stem
            if only_set and slug not in only_set:
                continue
            if out_path.exists() and not args.force:
                print(f"  — skip {slug} (exists, pass --force to overwrite)")
                continue
            jobs.append((slug, entry["text"], out_path))

    if not jobs:
        print("Nothing to render.")
        return

    print(f"Rendering {len(jobs)} clips via substrate={args.substrate}")
    total_t = 0.0
    total_bytes = 0
    failures: list[tuple[str, str]] = []
    for slug, text, out_path in jobs:
        try:
            t, n = await render_one(slug, text, out_path, args.substrate)
            total_t += t
            total_bytes += n
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {slug}: {type(e).__name__}: {e}", flush=True)
            failures.append((slug, str(e)))

    print()
    print(
        f"Done. rendered={len(jobs) - len(failures)}/{len(jobs)} "
        f"wall={total_t:.1f}s total_size={total_bytes / 1024 / 1024:.1f}MB"
    )
    if failures:
        print("Failures:")
        for slug, err in failures:
            print(f"  {slug}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
