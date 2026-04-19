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

from agents.seller import text_to_speech, render_comment_response_wav2lip  # noqa: E402
from agents.bridge_clips import BRIDGE_SCRIPTS  # noqa: E402
from config import POD_SPEAKING_1080P  # noqa: E402

OUT_DIR = BACKEND / "local_answers" / "_generic"
MANIFEST_PATH = OUT_DIR / "manifest.json"


# ── Audio + video alignment helpers ─────────────────────────────────────────
# Two structural problems showed up when we put v3 audio through Wav2Lip:
#
# 1. v3 adds ~50-100ms of "intake breath" silence at the start and ~100-300ms
#    of trailing silence/breath at the end of every clip. These are real
#    samples in the audio. Wav2Lip generates mouth shapes for ALL audio
#    frames, including silence — and on silent frames the mouth predictor
#    produces semi-random shapes (it wasn't trained for "person not speaking
#    while making no sound"). The result is a visible mouth-flap on what
#    should be a still frame at the start/end of every bridge.
#
# 2. Wav2Lip's mel-chunking algorithm computes video frame count as
#    `int(audio_seconds * fps)` truncated to the last full mel chunk window.
#    That always produces a video ~120-160ms shorter than the audio (a
#    structural artifact of MEL_STEP=16 windowing). When the wav2lip server
#    muxes with `-shortest`, that 150ms of audio gets cut at the tail —
#    which the user perceives as "the audio doesn't match the video"
#    because the trailing breath/word-end gets clipped.
#
# We fix both BEFORE re-rendering by:
# - trim_audio_silence(): crop head/tail silence from the v3 mp3 so wav2lip
#   only generates mouth shapes for actual speech frames. Internal pauses
#   (commas, ellipses) are preserved — only edge silence trims.
# - pad_video_to_audio(): re-mux the wav2lip output with the FULL trimmed
#   audio + video padded by holding the last frame for the gap. Output
#   length matches audio length within ±20ms; nothing gets cut.

def trim_audio_silence(
    audio_bytes: bytes,
    *,
    head_threshold_db: int = -40,
    head_min_silence: float = 0.03,
    tail_threshold_db: int = -35,
    tail_min_silence: float = 0.05,
) -> bytes:
    """Trim leading + trailing silence from MP3 bytes via ffmpeg's silenceremove.
    Internal pauses are preserved — the filter only triggers on a contiguous
    silent run at the start (and, after a reverse, at the end).

    Thresholds are deliberately conservative on the tail (-35dB, 50ms) so
    we keep faint trailing punctuation (the soft 't' at the end of 'about
    that') and only crop pure breath/silence."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fi:
        fi.write(audio_bytes)
        in_path = fi.name
    out_path = in_path.replace(".mp3", "_trimmed.mp3")
    af = (
        # Trim head silence
        f"silenceremove=start_periods=1:start_silence={head_min_silence}:"
        f"start_threshold={head_threshold_db}dB:detection=peak,"
        # Reverse, trim "head" (which is now the tail), reverse back
        f"areverse,"
        f"silenceremove=start_periods=1:start_silence={tail_min_silence}:"
        f"start_threshold={tail_threshold_db}dB:detection=peak,"
        f"areverse"
    )
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", in_path, "-af", af,
             "-loglevel", "error", out_path],
            check=True, timeout=15,
        )
        result = Path(out_path).read_bytes()
    finally:
        Path(in_path).unlink(missing_ok=True)
        Path(out_path).unlink(missing_ok=True)
    return result


def pad_video_to_audio(mp4_bytes: bytes, audio_bytes: bytes) -> bytes:
    """Re-mux mp4 with full audio + video padded by holding the last frame
    so durations match within ±20ms. Without this, wav2lip's `-shortest`
    mux truncates the audio tail (~150ms structural drift from mel chunking)
    and the user perceives the audio as "cut off"."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        mp4_in = td / "in.mp4"
        mp3_in = td / "audio.mp3"
        mp4_out = td / "out.mp4"
        mp4_in.write_bytes(mp4_bytes)
        mp3_in.write_bytes(audio_bytes)

        def _probe(p: Path, args: list[str]) -> float:
            return float(subprocess.check_output(
                ["ffprobe", "-v", "error", *args, "-show_entries",
                 "format=duration", "-of", "default=nw=1:nk=1", str(p)],
                timeout=10,
            ).strip())

        try:
            audio_dur = _probe(mp3_in, [])
            video_dur = float(subprocess.check_output(
                ["ffprobe", "-v", "error", "-select_streams", "v",
                 "-show_entries", "stream=duration", "-of",
                 "default=nw=1:nk=1", str(mp4_in)], timeout=10,
            ).strip())
        except Exception:
            # If probe fails, return the original bytes unmodified rather
            # than silently producing a broken file.
            return mp4_bytes

        pad = max(0.0, audio_dur - video_dur)
        if pad < 0.02:
            # Already aligned within 20ms; nothing to do.
            return mp4_bytes

        # tpad stop_mode=clone freezes the last frame for stop_duration sec.
        # Re-encoding video here is fine — bridges are 2-5s and the pod's
        # libx264 ultrafast preset chews through it in <300ms.
        subprocess.run([
            "ffmpeg", "-y", "-i", str(mp4_in), "-i", str(mp3_in),
            "-filter_complex",
            f"[0:v]tpad=stop_mode=clone:stop_duration={pad:.3f}[v]",
            "-map", "[v]", "-map", "1:a",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-loglevel", "error", str(mp4_out),
        ], check=True, timeout=30)
        return mp4_out.read_bytes()


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
