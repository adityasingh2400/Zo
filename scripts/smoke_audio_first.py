#!/usr/bin/env python
"""End-to-end smoke for the avatar realism build (Phases 1-4).

Connects a websocket to a running backend and exercises three flows:
  1. Comment escalate → audio-first dispatch (comment_response_audio
     fires with audio_url + word_timings + duration).
  2. Pitch trigger ("sell this for ..." command) → router fires
     pitch_product → play_clip(pitch_veo, muted, looped) + pitch_audio.
  3. Mic press → play_clip(listening_attentive, muted, looped) +
     voice_state=transcribing.

Usage:
    # Start the backend on port 8000 first
    backend/venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000 &
    # Then run the smoke
    python scripts/smoke_audio_first.py
    # Or against a custom port:
    python scripts/smoke_audio_first.py --port 8001

Exits 0 if all three flows pass, 1 if any fail.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import Any

import websockets

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


async def _drain_state_sync(ws) -> None:
    """The dashboard ws sends state_sync as the first frame; consume it
    so the smoke checks see clean event streams from here on out."""
    await asyncio.wait_for(ws.recv(), timeout=3)


async def _collect(
    ws,
    *,
    timeout: float = 8.0,
    stop_when: set[str] | None = None,
) -> list[dict]:
    """Collect events until `timeout` total seconds elapse OR we see one
    of `stop_when` event types — whichever comes first. The stop_when
    early-exit is critical post commit 2c98beb because the Director's
    idle rotation + motivated-idle observer keep firing events
    indefinitely, so a "wait for N seconds of silence" termination
    never trips."""
    out: list[dict] = []
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
        except asyncio.TimeoutError:
            break
        msg = json.loads(raw)
        if msg.get("type") == "agent_log":
            continue
        out.append(msg)
        if stop_when and msg.get("type") in stop_when:
            # Drain a few more events for context (sometimes the duration
            # handshake info comes on the very next frame).
            try:
                while True:
                    raw2 = await asyncio.wait_for(ws.recv(), timeout=0.3)
                    m2 = json.loads(raw2)
                    if m2.get("type") != "agent_log":
                        out.append(m2)
            except asyncio.TimeoutError:
                pass
            break
    return out


def _has_event(events: list[dict], type_: str, **filters: Any) -> dict | None:
    for e in events:
        if e.get("type") != type_:
            continue
        if all(e.get(k) == v for k, v in filters.items()):
            return e
    return None


async def smoke_audio_first(uri: str) -> bool:
    print("\n=== AUDIO-FIRST: simulate_comment 'is it real leather' ===")
    async with websockets.connect(uri) as ws:
        await _drain_state_sync(ws)
        await ws.send(json.dumps({"type": "stage_ready"}))
        await ws.send(json.dumps({
            "type": "simulate_comment", "text": "is it real leather",
        }))
        # 30s upper bound — post commit 2c98beb the pipeline reorders
        # Wav2Lip in front of comment_response_audio (audio + video
        # dispatch together to keep lip-sync working with karaoke). On a
        # warm pod that's ~5-12s; on a cold/offline pod the
        # connection-refused fallback can take 5-30s before audio-only
        # fires. We stop early the moment we see the target event so the
        # idle-rotation noise doesn't swallow the script.
        events = await _collect(
            ws, timeout=30,
            stop_when={"comment_response_audio", "comment_response_video_failed"},
        )

    audio_evt = _has_event(events, "comment_response_audio")
    if audio_evt:
        url = audio_evt.get("url", "")
        dur = audio_evt.get("expected_duration_ms")
        words = len(audio_evt.get("word_timings") or [])
        print(f"  PASS — comment_response_audio fired: url={url} dur={dur}ms words={words}")
        return True
    print("  FAIL — no comment_response_audio event")
    print(f"  saw types: {[e.get('type') for e in events]}")
    return False


async def smoke_pitch(uri: str) -> bool:
    print("\n=== PITCH: simulate_comment 'sell this for forty dollars to gen z' ===")
    async with websockets.connect(uri) as ws:
        await _drain_state_sync(ws)
        await ws.send(json.dumps({"type": "stage_ready"}))
        await ws.send(json.dumps({
            "type": "simulate_comment",
            "text": "sell this for forty dollars to gen z",
        }))
        events = await _collect(
            ws, timeout=10, stop_when={"pitch_audio"},
        )

    decision = _has_event(events, "routing_decision", tool="pitch_product")
    pitch_audio = _has_event(events, "pitch_audio")
    pitch_clip = next(
        (e for e in events
         if e.get("type") == "play_clip" and e.get("intent") == "pitch_veo"),
        None,
    )
    if not decision:
        print("  FAIL — router didn't pick pitch_product")
        return False
    if not pitch_clip:
        print("  FAIL — no play_clip(pitch_veo) event")
        return False
    if not pitch_audio:
        print("  FAIL — no pitch_audio event")
        return False
    if not pitch_clip.get("loop") or not pitch_clip.get("muted"):
        print(f"  FAIL — pitch_veo clip should be loop=True muted=True, got "
              f"loop={pitch_clip.get('loop')} muted={pitch_clip.get('muted')}")
        return False
    words = len(pitch_audio.get("word_timings") or [])
    dur = pitch_audio.get("expected_duration_ms")
    print(f"  PASS — router=pitch_product, pitch_veo loop=True muted=True, "
          f"audio words={words} dur={dur}ms")
    return True


async def smoke_mic_press(uri: str) -> bool:
    print("\n=== MIC PRESS: mic_pressed → listening_attentive ===")
    async with websockets.connect(uri) as ws:
        await _drain_state_sync(ws)
        await ws.send(json.dumps({"type": "stage_ready"}))
        await ws.send(json.dumps({"type": "mic_pressed", "client_ts": 0}))
        # Wait for either the listening_attentive play_clip OR the
        # voice_state=transcribing — whichever lands first.
        def _is_target(m):
            return m.get("type") in {"play_clip", "voice_state"}
        out = []
        deadline = time.monotonic() + 5
        seen_listen = False
        seen_voice = False
        while time.monotonic() < deadline and not (seen_listen and seen_voice):
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=deadline - time.monotonic())
            except asyncio.TimeoutError:
                break
            m = json.loads(raw)
            if m.get("type") == "agent_log":
                continue
            out.append(m)
            if m.get("type") == "play_clip" and m.get("intent") == "listening_attentive":
                seen_listen = True
            if m.get("type") == "voice_state" and m.get("state") == "transcribing":
                seen_voice = True
        events = out

    listen = next(
        (e for e in events
         if e.get("type") == "play_clip" and e.get("intent") == "listening_attentive"),
        None,
    )
    voice = _has_event(events, "voice_state", state="transcribing")
    if not listen:
        print("  FAIL — no play_clip(listening_attentive) event")
        return False
    if not voice:
        print("  FAIL — no voice_state=transcribing event")
        return False
    if not listen.get("loop") or not listen.get("muted"):
        print(f"  FAIL — listening clip should be loop=True muted=True, got "
              f"loop={listen.get('loop')} muted={listen.get('muted')}")
        return False
    print(f"  PASS — listening_attentive loop=True muted=True + voice_state=transcribing")
    return True


async def main_async(host: str, port: int) -> int:
    uri = f"ws://{host}:{port}/ws/dashboard"
    print(f"smoke target: {uri}")
    failed = 0
    for fn in (smoke_audio_first, smoke_pitch, smoke_mic_press):
        try:
            ok = await fn(uri)
        except Exception as e:
            print(f"  FAIL — {fn.__name__} raised: {e}")
            ok = False
        if not ok:
            failed += 1
    print("\n" + "=" * 50)
    print(f"smoke: {3 - failed}/3 passed")
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()
    failed = asyncio.run(main_async(args.host, args.port))
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
