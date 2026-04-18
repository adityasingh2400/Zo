"""Video intake pipeline for EMPIRE — adapted from SwarmSell.

Records a 10-second product video with voiceover, then:
  1. Audio: ffmpeg extract → transcription
  2. Transcript → selling points extraction
  3. Video: ffmpeg key frames → OpenCV sharpness filter → best frames
  4. Best frames → product analysis + TripoSR 3D model
"""
from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
import base64
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("empire.intake")


@dataclass
class IntakeTimings:
    audio_extraction_sec: float = 0.0
    transcription_sec: float = 0.0
    frame_extraction_sec: float = 0.0
    filter_sec: float = 0.0
    total_sec: float = 0.0
    frame_count: int = 0
    filtered_frame_count: int = 0


# ── Audio Extraction ─────────────────────────────────────────────────────────

async def extract_audio(video_path: str) -> str:
    """Extract audio from video to WAV (16kHz mono). Returns path."""
    import tempfile
    output_path = tempfile.mktemp(suffix=".wav", prefix="empire_audio_")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-y", "-loglevel", "error", output_path,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise ValueError(f"Audio extraction failed: {stderr.decode(errors='replace')[:300]}")
    return output_path


async def transcribe_with_gemma(audio_path: str) -> str:
    """Transcribe audio. Uses Deepgram Nova-3 (fast, proven from SwarmSell).
    Falls back to Ollama if no Deepgram key."""
    import os as _os
    import httpx

    DEEPGRAM_API_KEY = _os.getenv("DEEPGRAM_API_KEY", "")

    # Primary: Deepgram Nova-3 (~1-2s for 10s audio, proven from SwarmSell)
    if DEEPGRAM_API_KEY:
        try:
            with open(audio_path, "rb") as f:
                audio_data = f.read()

            t0 = time.time()
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.deepgram.com/v1/listen",
                    params={"model": "nova-3", "smart_format": "true"},
                    headers={
                        "Authorization": f"Token {DEEPGRAM_API_KEY}",
                        "Content-Type": "audio/wav",
                    },
                    content=audio_data,
                )
                response.raise_for_status()

            transcript = response.json()["results"]["channels"][0]["alternatives"][0]["transcript"]
            ms = int((time.time() - t0) * 1000)
            logger.info("Deepgram transcription: %d chars in %dms", len(transcript), ms)
            return transcript
        except Exception as e:
            logger.warning("Deepgram failed: %s — falling back to Ollama", e)

    # Fallback: Ollama
    OLLAMA_URL = _os.getenv("OLLAMA_URL", "http://localhost:11434")
    GEMMA_MODEL = _os.getenv("GEMMA_MODEL", "gemma4:e4b")

    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": GEMMA_MODEL,
                    "messages": [{
                        "role": "user",
                        "content": "Transcribe this audio exactly. Return only the spoken words.",
                        "audios": [audio_b64],
                    }],
                    "stream": False,
                    "options": {"num_predict": 500},
                },
            )
            resp.raise_for_status()
            result = resp.json()
            transcript = result.get("message", {}).get("content", "")
            duration_ms = result.get("total_duration", 0) / 1_000_000
            logger.info("Ollama transcription: %d chars in %.0fms", len(transcript), duration_ms)
            return transcript
    except Exception as e:
        logger.warning("Transcription failed: %s", e)
        return ""


# ── Frame Extraction ─────────────────────────────────────────────────────────

async def extract_key_frames(
    video_path: str,
    target_frames: int = 10,
    jpeg_quality: int = 2,
) -> list[tuple[int, bytes]]:
    """Extract frames from video, return as (index, jpeg_bytes) tuples."""
    duration = await _get_video_duration(video_path)
    if duration <= 0:
        return []

    fps = target_frames / duration
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        "-f", "image2pipe", "-vcodec", "mjpeg",
        "-q:v", str(jpeg_quality),
        "-loglevel", "error", "-",
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        logger.warning("Frame extraction failed: %s", stderr.decode(errors='replace')[:200])
        return []

    frames = []
    SOI, EOI = b"\xff\xd8", b"\xff\xd9"
    buf = bytearray(stdout)
    idx = 0
    while True:
        soi_pos = buf.find(SOI)
        if soi_pos == -1:
            break
        eoi_pos = buf.find(EOI, soi_pos + 2)
        if eoi_pos == -1:
            break
        frames.append((idx, bytes(buf[soi_pos:eoi_pos + 2])))
        buf = buf[eoi_pos + 2:]
        idx += 1

    return frames


def filter_quality_frames(
    frames: list[tuple[int, bytes]],
    max_output: int = 4,
) -> list[tuple[int, bytes]]:
    """OpenCV Laplacian sharpness filter. Picks N frames spread across timeline for angle diversity."""
    if not frames:
        return []
    if len(frames) <= max_output:
        return frames

    # Split timeline into N equal segments, pick sharpest from each
    segment_size = len(frames) // max_output
    selected = []

    for i in range(max_output):
        start = i * segment_size
        end = start + segment_size if i < max_output - 1 else len(frames)
        segment = frames[start:end]

        best = max(segment, key=lambda f: _sharpness(f[1]))
        selected.append(best)

    logger.info("Quality filter: %d → %d frames (spread across timeline)", len(frames), len(selected))
    return selected


def _sharpness(jpeg_bytes: bytes) -> float:
    arr = np.frombuffer(jpeg_bytes, np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


async def _get_video_duration(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    try:
        return float(stdout.decode().strip())
    except (ValueError, TypeError):
        return 0.0


def frames_to_base64(frames: list[tuple[int, bytes]]) -> list[str]:
    """Convert frame jpeg bytes to base64 strings, resized to 720px wide for fast inference."""
    from PIL import Image
    from io import BytesIO

    result = []
    for _, data in frames:
        img = Image.open(BytesIO(data))
        if img.width > 512:
            ratio = 512 / img.width
            img = img.resize((512, int(img.height * ratio)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=70)
        result.append(base64.b64encode(buf.getvalue()).decode())
    return result


# ── Full Intake Pipeline ─────────────────────────────────────────────────────

async def process_video(video_path: str) -> dict:
    """Full video intake: extract audio + frames, transcribe, filter frames.

    Returns dict with transcript, selling_points_raw, and best frame base64 strings.
    """
    timings = IntakeTimings()
    t_total = time.perf_counter()

    # Run audio extraction and frame extraction in parallel
    t0 = time.perf_counter()
    audio_task = asyncio.create_task(extract_audio(video_path))
    frames_task = asyncio.create_task(extract_key_frames(video_path, target_frames=15))

    audio_path = await audio_task
    timings.audio_extraction_sec = time.perf_counter() - t0

    all_frames = await frames_task
    timings.frame_extraction_sec = time.perf_counter() - t0 - timings.audio_extraction_sec
    timings.frame_count = len(all_frames)

    # Transcribe audio with Gemma 4
    t1 = time.perf_counter()
    transcript = await transcribe_with_gemma(audio_path)
    timings.transcription_sec = time.perf_counter() - t1

    # Cleanup audio file
    Path(audio_path).unlink(missing_ok=True)

    # Filter for best frames. Transcript extraction runs as a background
    # task in main.py — it's slower than the rest of intake on CPU and we
    # don't want to gate the sell pipeline on it.
    t2 = time.perf_counter()
    best_frames = await asyncio.to_thread(filter_quality_frames, all_frames, max_output=4)
    timings.filter_sec = time.perf_counter() - t2
    timings.filtered_frame_count = len(best_frames)

    timings.total_sec = time.perf_counter() - t_total

    logger.info(
        "Intake complete: %.1fs total (audio=%.1f, transcribe=%.1f, frames=%.1f, filter=%.1f) "
        "%d→%d frames, %d chars transcript",
        timings.total_sec, timings.audio_extraction_sec, timings.transcription_sec,
        timings.frame_extraction_sec, timings.filter_sec,
        timings.frame_count, timings.filtered_frame_count,
        len(transcript),
    )

    return {
        "transcript": transcript,
        "best_frames": best_frames,
        "best_frames_b64": frames_to_base64(best_frames),
        "timings": {
            "audio_extraction_sec": round(timings.audio_extraction_sec, 2),
            "transcription_sec": round(timings.transcription_sec, 2),
            "frame_extraction_sec": round(timings.frame_extraction_sec, 2),
            "filter_sec": round(timings.filter_sec, 2),
            "total_sec": round(timings.total_sec, 2),
            "frame_count": timings.frame_count,
            "filtered_frame_count": timings.filtered_frame_count,
        },
    }
