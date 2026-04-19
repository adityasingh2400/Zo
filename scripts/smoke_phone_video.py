#!/usr/bin/env python
"""End-to-end smoke for the phone video upload path.

Connects to /ws/phone as the phone client, base64-encodes a real video
file, sends a `sell_video` message, then connects to /ws/dashboard as
the dashboard client and waits for the audio-first pitch to fire.

Verifies the full pipeline:
  phone (sell_video) → run_video_sell_pipeline → process_video → Claude
  → text_to_speech → Director.dispatch_audio_first_pitch → pitch_audio.

Usage:
    # 1) start backend
    cd backend && ./venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000

    # 2) run smoke
    python scripts/smoke_phone_video.py
    # or with a different video / port
    python scripts/smoke_phone_video.py --video path/to/clip.mp4 --port 8001
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from pathlib import Path

import websockets


async def _phone_uploader(uri: str, video_path: Path) -> str:
    """Connect as phone, send sell_video, return the pipeline session_id
    once we've seen pipeline_started."""
    print(f"phone: encoding {video_path.name} ({video_path.stat().st_size:,} bytes)…")
    video_b64 = base64.b64encode(video_path.read_bytes()).decode()

    async with websockets.connect(uri, max_size=2**26) as ws:
        await ws.send(json.dumps({
            "type": "sell_video",
            "video_b64": video_b64,
            "filename": video_path.name,
            "voice_text": "sell this premium watch for two hundred dollars",
        }))
        session_id = None
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=30)
                msg = json.loads(raw)
                if msg.get("type") != "phone_ack":
                    continue
                stage = msg.get("stage")
                print(f"phone: ack stage={stage} {msg.get('reason') or msg.get('bytes') or ''}")
                if stage == "received":
                    session_id = msg.get("session_id")
                if stage == "pipeline_started":
                    return session_id or msg.get("session_id") or ""
                if stage == "error":
                    return ""
        except asyncio.TimeoutError:
            print("phone: timed out waiting for pipeline_started ack")
            return ""


async def _dashboard_watcher(uri: str, *, deadline_sec: float) -> dict:
    """Connect as dashboard, drain state_sync, watch for pitch_audio
    until deadline. Returns the pitch_audio event dict or {} on timeout."""
    async with websockets.connect(uri, max_size=2**26) as ws:
        # Drain initial state_sync.
        try:
            await asyncio.wait_for(ws.recv(), timeout=3)
        except asyncio.TimeoutError:
            pass
        deadline = time.time() + deadline_sec
        pitch_evt: dict = {}
        seen_stages: list[str] = []
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=deadline - time.time())
            except asyncio.TimeoutError:
                break
            msg = json.loads(raw)
            t = msg.get("type")
            if t in (
                "phone_video_received", "transcript", "product_data",
                "sales_script", "pitch_audio", "pitch_video",
                "pitch_audio_end", "comment_response_video_failed",
            ):
                seen_stages.append(t)
                if t == "transcript":
                    print(f"  → transcript: {msg.get('text', '')[:80]!r}")
                elif t == "product_data":
                    name = (msg.get("data") or {}).get("name", "?")
                    print(f"  → product_data: {name}")
                elif t == "sales_script":
                    sc = msg.get("script", "")
                    print(f"  → sales_script ({len(sc.split())} words): {sc[:80]!r}…")
                elif t == "pitch_audio":
                    pitch_evt = msg
                    print(f"  → pitch_audio: url={msg.get('url')} "
                          f"words={len(msg.get('word_timings') or [])} "
                          f"dur={msg.get('expected_duration_ms')}ms")
                    break
                elif t == "pitch_video":
                    print(f"  → pitch_video: backend={msg.get('backend')}")
                else:
                    print(f"  → {t}")
        return pitch_evt or {"_seen": seen_stages}


async def main_async(host: str, port: int, video: Path, deadline: float) -> int:
    phone_uri = f"ws://{host}:{port}/ws/phone"
    dash_uri = f"ws://{host}:{port}/ws/dashboard"
    print(f"smoke target: phone={phone_uri} dashboard={dash_uri}")

    if not video.exists():
        print(f"FAIL — test video not found: {video}")
        return 1

    # Start the dashboard watcher first so it captures every event from
    # the moment the phone uploads. We don't await its task here; it runs
    # in parallel with the upload.
    dash_task = asyncio.create_task(
        _dashboard_watcher(dash_uri, deadline_sec=deadline),
    )
    # Brief pause so the watcher is connected before the phone uploads.
    await asyncio.sleep(0.5)

    session_id = await _phone_uploader(phone_uri, video)
    if not session_id:
        print("FAIL — phone never got pipeline_started ack")
        dash_task.cancel()
        return 1
    print(f"phone: pipeline_started ack received (session={session_id})")

    pitch_evt = await dash_task
    if pitch_evt and pitch_evt.get("type") == "pitch_audio":
        words = len(pitch_evt.get("word_timings") or [])
        dur = pitch_evt.get("expected_duration_ms")
        print(f"\nPASS — phone video → pitch_audio fired (words={words}, dur={dur}ms)")
        return 0
    seen = pitch_evt.get("_seen") if isinstance(pitch_evt, dict) else None
    print(f"\nFAIL — pitch_audio never fired within {deadline}s. "
          f"Saw stages: {seen}")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--video", type=Path,
                    default=Path("backend/test_fixtures/watch_demo.mov"))
    ap.add_argument("--deadline", type=float, default=60.0,
                    help="seconds to wait for pitch_audio after phone ack")
    args = ap.parse_args()
    sys.exit(asyncio.run(main_async(args.host, args.port, args.video, args.deadline)))


if __name__ == "__main__":
    main()
