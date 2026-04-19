#!/usr/bin/env python
"""End-to-end smoke for the audience QR comment intake.

Mirrors the shape of scripts/smoke_audio_first.py + smoke_phone_video.py
and exercises the new /api/audience_comment endpoint that backs the
TikTokShopOverlay chat rail at /stage.

Verifies:
  1. Connect /ws/dashboard, drain initial state_sync.
  2. POST a comment to /api/audience_comment as if it came from a phone
     after scanning the QR.
  3. Within a generous deadline, observe both:
       a) `audience_comment` (drives the chat-rail bubble in the overlay)
       b) `routing_decision`  (proves run_routed_comment fired downstream
          — the same path typed comments use, so the cost ticker + avatar
          response chain is wired up identically for QR submissions)

The deadline gives the rule-based router and the on-device Gemma
classify call enough headroom on a cold demo Mac (~3-5s warm, up to
~12s cold for the first Cactus call). On a warm backend both events
arrive in <500ms.

Optional: --rate flag fires a second POST too fast to confirm the per-IP
rate limiter responds with HTTP 429 cleanly (no traceback, no broken WS).

Usage:
    backend/venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000 \\
      --ws-max-size 67108864 &
    python scripts/smoke_audience_qr.py
    # or against a different host/port
    python scripts/smoke_audience_qr.py --host 127.0.0.1 --port 8001

Exits 0 if the broadcast + downstream router events both fire, 1 otherwise.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.request
import urllib.error
from typing import Any

import websockets

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def _post_audience_comment(host: str, port: int, payload: dict) -> tuple[int, str]:
    """Plain stdlib POST to /api/audience_comment so the smoke has zero
    extra deps beyond the websockets package the other smokes already use.
    Returns (status_code, body) — does NOT raise on non-2xx so the caller
    can assert on 429 in the rate-limit branch."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://{host}:{port}/api/audience_comment",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.getcode(), resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, (e.read() or b"").decode("utf-8", "replace")


async def _drain_state_sync(ws) -> None:
    """The dashboard ws sends state_sync as the first frame; consume it
    so the smoke sees clean event streams from here on out."""
    await asyncio.wait_for(ws.recv(), timeout=3)


async def smoke_audience_qr(host: str, port: int) -> bool:
    """POST a QR-style audience comment and verify both the informational
    broadcast (audience_comment) AND the downstream router event
    (routing_decision) land on /ws/dashboard.

    The two events together prove the full wiring:
      audience_comment   = chat-rail bubble path (drives TikTokShopOverlay)
      routing_decision   = run_routed_comment path (drives CostTicker
                           + avatar response, identical to typed comments)
    """
    print("\n=== AUDIENCE QR: POST /api/audience_comment ===")
    uri = f"ws://{host}:{port}/ws/dashboard"
    payload = {
        "username": "smoke_qr",
        # Use the same canonical demo phrase smoke_audio_first.py routes —
        # 'is it real leather' is in products.json's qa_index for the demo
        # wallet so it picks up a respond_locally on a warm backend, but
        # it also exercises the cloud escalate fallback if local_answers/
        # is empty (the demo's fresh-clone state). Either way we get a
        # routing_decision broadcast — that's what we're asserting.
        "text": "is it real leather",
    }

    async with websockets.connect(uri) as ws:
        await _drain_state_sync(ws)

        status, body = _post_audience_comment(host, port, payload)
        if status != 200:
            print(f"  FAIL — POST returned {status}: {body[:200]}")
            return False
        print(f"  POST ok: {body[:120]}")

        # Collect events for up to 15s OR until we've seen both targets.
        # We don't early-return on the FIRST target because routing_decision
        # can land before the audience_comment broadcast on a very warm
        # box — order isn't guaranteed (both go through the same broadcast
        # bus, but the audience_comment is awaited first in the request
        # handler so it's typically first).
        deadline = time.monotonic() + 15.0
        seen_aud = None
        seen_route = None
        while time.monotonic() < deadline and not (seen_aud and seen_route):
            try:
                raw = await asyncio.wait_for(
                    ws.recv(),
                    timeout=max(0.05, deadline - time.monotonic()),
                )
            except asyncio.TimeoutError:
                break
            msg = json.loads(raw)
            mt = msg.get("type")
            if mt == "agent_log":
                continue
            if mt == "audience_comment":
                if msg.get("text") == payload["text"] and msg.get("username") == payload["username"]:
                    seen_aud = msg
                    print(f"  → audience_comment: @{msg.get('username')}: {msg.get('text')}")
            elif mt == "routing_decision":
                if msg.get("comment") == payload["text"]:
                    seen_route = msg
                    print(f"  → routing_decision: tool={msg.get('tool')} "
                          f"local={msg.get('was_local')} ms={msg.get('ms')}")

        if not seen_aud:
            print("  FAIL — never saw audience_comment broadcast for the POSTed text")
            return False
        if not seen_route:
            print("  FAIL — never saw routing_decision for the POSTed text "
                  "(run_routed_comment didn't fire for the audience comment)")
            return False
        print("  PASS — audience_comment + routing_decision both observed")
        return True


async def smoke_rate_limit(host: str, port: int) -> bool:
    """Burst above the 5/min cap and confirm a 429 response (no traceback,
    no broken WS). Best-effort: if the previous test left fewer than 5
    successful posts in the bucket this run still drains the bucket and
    asserts 429 on the next post."""
    print("\n=== AUDIENCE RATE LIMIT: 6 posts back-to-back ===")
    statuses: list[int] = []
    for i in range(6):
        status, _ = _post_audience_comment(
            host, port,
            {"username": "smoke_rate", "text": f"rate-limit probe {i}"},
        )
        statuses.append(status)
    print(f"  statuses: {statuses}")
    # The bucket carries 60s of history. If the previous test ran in the
    # same minute, we may already be over the cap — we just assert AT LEAST
    # one 429 in the run.
    if 429 not in statuses:
        print("  FAIL — expected at least one 429 in the burst, got none")
        return False
    if not any(s == 200 for s in statuses):
        # Acceptable, just informational — the bucket was already saturated
        # from prior runs in the same minute. Not an outright failure.
        print("  WARN — bucket already saturated, all six returned 429")
    print("  PASS — rate limiter returned 429 cleanly")
    return True


async def main_async(host: str, port: int, run_rate: bool) -> int:
    print(f"smoke target: http://{host}:{port}  ws://{host}:{port}/ws/dashboard")
    suites = [smoke_audience_qr]
    if run_rate:
        suites.append(smoke_rate_limit)
    failed = 0
    for fn in suites:
        try:
            ok = await fn(host, port)
        except Exception as e:
            print(f"  FAIL — {fn.__name__} raised: {e}")
            ok = False
        if not ok:
            failed += 1
    print("\n" + "=" * 50)
    print(f"smoke: {len(suites) - failed}/{len(suites)} passed")
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--rate", action="store_true",
                    help="also exercise the rate-limit 429 path")
    args = ap.parse_args()
    failed = asyncio.run(main_async(args.host, args.port, args.rate))
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
