"""Pre-render the generic intro / bridge / acknowledgment clips used by the
stage demo's /api/go_live endpoint and the router's play_canned_clip path.

Mirrors scripts/render_local_answers.py — same TTS + Wav2Lip render loop,
just sourcing scripts from agents.bridge_clips.BRIDGE_SCRIPTS instead of
the per-product qa_index. Outputs to backend/local_answers/_generic/ so the
existing /local_answers static mount serves them with no extra wiring,
and writes a manifest the bridge_clips._load_generic_manifest() reader
discovers automatically (third-tier fallback in pick_bridge_clip).

Why Wav2Lip and not LatentSync (which bridge_clips.render_all uses):
  - Wav2Lip ~3-4s per clip on the warm pod vs LatentSync ~25-40s.
  - For a 24-hour hackathon build we want short feedback loops; the loss
    in quality on a 1.5-2.5s clip is invisible at stage-display distances.
  - Both renderers can coexist — pick_bridge_clip prefers the LatentSync
    manifest if present, falls through to this Wav2Lip set otherwise.

Idempotent: skips entries whose output file already exists. Pass --force to
re-render. Pass --only label1,label2 to render specific labels.

Requires the backend's .env (AWS, ElevenLabs, Wav2Lip tunnel) so the script
imports from backend.config + backend.agents.seller to reuse the live
clients. Run with the RunPod tunnel open and the Cactus venv active.

Usage:
  python scripts/render_generic_clips.py
  python scripts/render_generic_clips.py --force
  python scripts/render_generic_clips.py --only intro_arbitrary,bridge_arbitrary
  python scripts/render_generic_clips.py --substrate /workspace/state_pitching_pose_speaking_1080p.mp4
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

# Load .env before importing seller so the ElevenLabs/RunPod clients pick
# it up the same way render_local_answers.py does.
from dotenv import load_dotenv  # type: ignore
load_dotenv(ROOT / ".env")

from agents.seller import (  # noqa: E402
    text_to_speech,
    render_comment_response_wav2lip,
    trim_audio_silence,
    pad_wav2lip_video_to_audio,
)
from agents.bridge_clips import BRIDGE_SCRIPTS  # noqa: E402
from config import POD_SPEAKING_1080P  # noqa: E402

OUT_DIR = BACKEND / "local_answers" / "_generic"
MANIFEST_PATH = OUT_DIR / "manifest.json"


# Local thin wrapper that adapts the new (bytes, diag) return shape from
# pad_wav2lip_video_to_audio() back to the just-bytes contract render_one
# expects. The diag dict from seller.py is dropped here because the bridge
# render loop already prints its own per-clip stats; for the live path in
# main.py we DO consume the diag dict via trace.phase("video_padded", ...).
def pad_video_to_audio(mp4_bytes: bytes, audio_bytes: bytes) -> bytes:
    out, _diag = pad_wav2lip_video_to_audio(mp4_bytes, audio_bytes)
    return out


def _slug(text: str) -> str:
    """Stable 10-char hash → idempotent file naming. Matches bridge_clips._slug
    so the same script can't accidentally re-render under a different name."""
    return hashlib.sha256(text.encode()).hexdigest()[:10]


def load_manifest() -> dict[str, list[dict]]:
    """Load the existing _generic manifest, or {} on first run."""
    if not MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text())
    except Exception:
        return {}


def save_manifest(manifest: dict[str, list[dict]]) -> None:
    """Persist after each successful render so a mid-run abort doesn't
    lose progress. Sorted-key dump = clean diffs in version control if
    you ever check it in (it's gitignored by default — see .gitignore)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True))


async def render_one(
    label: str, text: str, out_path: Path, substrate: str,
    *, model_id: str = "eleven_v3",
) -> tuple[float, int]:
    """TTS → trim → Wav2Lip → pad → write. Returns (elapsed_s, bytes).
    Raises on failure so the caller can record + continue with the next clip.

    `model_id` defaults to eleven_v3 because BRIDGE_SCRIPTS is tagged with
    inline audio directives ([curious], [warmly], etc.) that flash would
    read aloud. v3 honours them. Render time per clip is ~6-8s (vs ~4-5s
    on flash) but bridges are pre-rendered once and re-used across every
    demo, so the extra render cost is paid in the past tense forever.

    Two post-processing steps wrap the wav2lip call:
      1. trim_audio_silence BEFORE wav2lip — strips head/tail silence so
         wav2lip's mouth predictor only runs on speech frames (kills the
         random mouth-flap on silent edges that v3's breathy intro/outro
         would otherwise produce).
      2. pad_video_to_audio AFTER wav2lip — wav2lip's mel-chunking always
         produces a video ~120-150ms shorter than the audio. The server's
         `-shortest` mux cuts the audio tail; we re-mux locally with the
         FULL trimmed audio + video padded by holding the last frame.
         End result: drift drops from -150ms to ±20ms, no audio cut."""
    t0 = time.perf_counter()
    audio_raw = await text_to_speech(text, model_id=model_id)
    if not audio_raw:
        raise RuntimeError("TTS returned empty audio (ElevenLabs not configured?)")
    t_tts = time.perf_counter() - t0

    # Trim head/tail silence from the v3 audio before sending to wav2lip.
    # Synchronous + fast (~80-150ms wall via ffmpeg) so we just call inline.
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

    # Pad video so durations align within ±20ms. Local ffmpeg re-mux
    # using the trimmed audio (not the wav2lip-cut version inside mp4_raw).
    t_pad_start = time.perf_counter()
    mp4 = pad_video_to_audio(mp4_raw, audio)
    t_pad = time.perf_counter() - t_pad_start

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(mp4)
    total = time.perf_counter() - t0
    print(
        f"  ✓ {label}/{out_path.stem} [{model_id}]: "
        f"tts={t_tts:.1f}s trim={t_trim:.2f}s({trim_pct:.0f}%) "
        f"lip={t_lip:.1f}s pad={t_pad:.2f}s "
        f"total={total:.1f}s size={len(mp4) / 1024:.0f}KB",
        flush=True,
    )
    return total, len(mp4)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-render existing files")
    ap.add_argument(
        "--only",
        default="",
        help="comma-separated label list (e.g. intro_arbitrary,bridge_arbitrary)",
    )
    ap.add_argument(
        "--substrate",
        default=POD_SPEAKING_1080P,
        help="pod-side path to source speaking video",
    )
    ap.add_argument(
        "--model",
        default="eleven_v3",
        help="ElevenLabs model id. Default eleven_v3 honours the audio "
             "tags in BRIDGE_SCRIPTS. Pass eleven_flash_v2_5 only after "
             "stripping tags from the scripts (otherwise flash reads them aloud).",
    )
    ap.add_argument(
        "--reset",
        action="store_true",
        help="wipe existing _generic dir + manifest before rendering. "
             "Use after BRIDGE_SCRIPTS text changes (sha256 slugs change → "
             "old MP4s become orphans the runtime picker still sees).",
    )
    args = ap.parse_args()

    # --reset: clear stale clips/manifest. Critical when BRIDGE_SCRIPTS
    # text changes — slugs are sha256-of-text, so any edit produces new
    # filenames and leaves the old ones behind. Without --reset, the
    # runtime random.choice() picker would mix new (tagged) clips with
    # stale (untagged) ones and the demo voice would feel inconsistent.
    if args.reset and OUT_DIR.exists():
        import shutil as _shutil
        for child in OUT_DIR.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                _shutil.rmtree(child)
        print(f"[reset] cleared {OUT_DIR.relative_to(ROOT)}/")

    only_set = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    # Plan: for each (label, text) in BRIDGE_SCRIPTS, decide whether to
    # render. Skip if the file already exists AND is registered in the
    # manifest, unless --force.
    jobs: list[tuple[str, str, Path]] = []
    for label, text in BRIDGE_SCRIPTS:
        if only_set and label not in only_set:
            continue
        slug = _slug(text)
        out_path = OUT_DIR / f"{label}_{slug}.mp4"
        already_in_manifest = any(
            e.get("file", "").endswith(out_path.name)
            for e in manifest.get(label, [])
        )
        if out_path.exists() and already_in_manifest and not args.force:
            print(f"  — skip {label}/{slug} (exists, pass --force to overwrite)")
            continue
        jobs.append((label, text, out_path))

    if not jobs:
        print("Nothing to render.")
        # Still print where the manifest is so the operator can verify.
        print(f"manifest: {MANIFEST_PATH.relative_to(ROOT)}")
        return

    print(f"Rendering {len(jobs)} generic clips via substrate={args.substrate}")
    print(f"  model={args.model}  output={OUT_DIR.relative_to(ROOT)}/")
    total_t = 0.0
    total_bytes = 0
    failures: list[tuple[str, str, str]] = []
    for label, text, out_path in jobs:
        try:
            t, n = await render_one(label, text, out_path, args.substrate,
                                    model_id=args.model)
            total_t += t
            total_bytes += n
            # Add to manifest under its label, dedup by file name so reruns
            # of --force don't keep accumulating duplicate entries.
            entries = manifest.setdefault(label, [])
            entries[:] = [e for e in entries if e.get("file", "").endswith(out_path.name) is False]
            entries.append({
                "script": text,
                "file": str(out_path.relative_to(BACKEND)),
                "url": f"/local_answers/_generic/{out_path.name}",
                "ms": None,  # would need ffprobe; bridge_clips falls back to
                             # a word-count estimate when ms is None.
            })
            save_manifest(manifest)
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {label}/{out_path.stem}: {type(e).__name__}: {e}", flush=True)
            failures.append((label, out_path.stem, str(e)))

    print()
    print(
        f"Done. rendered={len(jobs) - len(failures)}/{len(jobs)} "
        f"wall={total_t:.1f}s total_size={total_bytes / 1024 / 1024:.1f}MB"
    )
    print(f"manifest: {MANIFEST_PATH.relative_to(ROOT)}")
    if failures:
        print("Failures:")
        for label, slug, err in failures:
            print(f"  {label}/{slug}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
