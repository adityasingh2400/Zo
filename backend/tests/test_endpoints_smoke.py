"""Smoke tests for the public HTTP + WS endpoints.

Coverage philosophy: every endpoint that's part of the demo's golden path
gets at least one happy-path assertion, plus failure cases for the routes
where misuse is plausible (404 on bad product_id, etc).

What's NOT here yet:
  - Heavy pipeline endpoints (/api/voice_comment, /api/respond_to_comment,
    /api/sell, /api/sell-video, /api/creator/build) — mocking the full
    Bedrock + ElevenLabs + Wav2Lip + rembg + ffmpeg surface burns the
    test budget. They get integration tests in follow-ups when we can
    run against fakes that share the real interface.
  - Long-running flows that depend on Cactus/Gemma being loaded —
    conftest.py skips the lifespan deliberately so tests stay fast.
"""
from __future__ import annotations

import os

import pytest

# ── Health + state ──────────────────────────────────────────────────────────


def test_state_returns_expected_shape(client):
    r = client.get("/api/state")
    assert r.status_code == 200
    body = r.json()
    for key in ("status", "product_data", "active_product_id",
                "products", "has_photo", "has_3d", "log_count"):
        assert key in body, f"missing key {key} in /api/state"
    assert isinstance(body["products"], list)


def test_state_with_product_loaded(client, with_product):
    r = client.get("/api/state")
    body = r.json()
    assert body["active_product_id"] == "test_wallet"
    assert body["product_data"]["name"] == "Test Wallet"
    assert any(p["id"] == "test_wallet" for p in body["products"])


# ── Multi-product endpoint ──────────────────────────────────────────────────


def test_set_active_product_valid(client, with_product):
    r = client.post("/api/state/active_product",
                    data={"product_id": "test_wallet"})
    assert r.status_code == 200
    body = r.json()
    assert body["active_product_id"] == "test_wallet"
    assert body["product_name"] == "Test Wallet"
    assert body["qa_count"] == 1


def test_set_active_product_invalid_returns_404(client, with_product):
    r = client.post("/api/state/active_product",
                    data={"product_id": "nonexistent"})
    assert r.status_code == 404
    assert "not in catalog" in r.json()["detail"]


# ── BRAIN ───────────────────────────────────────────────────────────────────


def test_brain_stats_empty(client):
    r = client.get("/api/brain/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 0
    assert body["by_tool"] == {}
    assert body["top_answers"] == []
    assert body["top_misses"] == []
    assert body["total_cost_saved_usd"] == 0.0


def test_brain_stats_after_record(client):
    """Drive the brain module directly + verify stats reflect it. Bypasses
    the full router pipeline (which requires Cactus loaded) so the test
    isolates BRAIN's persist + aggregate logic."""
    from agents import brain
    brain.record_event(
        stream_id="default",
        product_id="test_wallet",
        comment="is it real leather",
        classify={"type": "question", "source": "fake"},
        decision={
            "tool": "respond_locally",
            "args": {"answer_id": "is_it_real_leather"},
            "reason": "matched",
            "ms": 1,
            "was_local": True,
            "cost_saved_usd": 0.00035,
        },
    )
    body = client.get("/api/brain/stats").json()
    assert body["total"] == 1
    assert body["by_tool"]["respond_locally"] == 1
    assert body["pct_local"] == 100
    assert abs(body["total_cost_saved_usd"] - 0.00035) < 1e-6
    assert body["top_answers"] == [{"answer_id": "is_it_real_leather", "count": 1}]


def test_brain_stats_filters_by_stream(client):
    from agents import brain
    base = {
        "comment": "ship",
        "classify": {"type": "question"},
        "decision": {"tool": "respond_locally", "args": {"answer_id": "shipping"},
                     "reason": "matched", "ms": 1, "was_local": True,
                     "cost_saved_usd": 0.0001},
    }
    brain.record_event(stream_id="alpha", product_id="x", **base)
    brain.record_event(stream_id="beta",  product_id="x", **base)

    all_streams = client.get("/api/brain/stats").json()
    assert all_streams["total"] == 2

    only_alpha = client.get("/api/brain/stats?stream_id=alpha").json()
    assert only_alpha["total"] == 1


def test_brain_top_misses_groups_tokens(client):
    """Recurring tokens in escalate_to_cloud comments rise to the top —
    that's the seller's "Q/A entries to author next" queue."""
    from agents import brain
    for comment in [
        "compare to apple watch please",
        "how does this compare to apple",
        "compare specs apple",
    ]:
        brain.record_event(
            stream_id="default", product_id="x", comment=comment,
            classify={"type": "question"},
            decision={"tool": "escalate_to_cloud",
                      "args": {"comment": comment},
                      "reason": "no match", "ms": 0,
                      "was_local": False, "cost_saved_usd": 0.0},
        )
    body = client.get("/api/brain/stats").json()
    miss_tokens = {m["token"] for m in body["top_misses"]}
    assert "apple" in miss_tokens
    assert "compare" in miss_tokens


# ── Comment intake (fire-and-forget) ────────────────────────────────────────


def test_post_comment_returns_processing(client):
    r = client.post("/api/comment", data={"text": "hello world"})
    assert r.status_code == 200
    assert r.json() == {"status": "processing"}


# ── Static + form serving ───────────────────────────────────────────────────


def test_comment_form_serves_html(client):
    """The audience-comment QR target. Was a false-FAIL in demo_prewarm.sh
    until S0.3 fixed the HEAD-vs-GET bug — keep a regression here."""
    r = client.get("/comment")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert "<textarea" in r.text


# ── CORS lockdown ───────────────────────────────────────────────────────────


def test_cors_allows_configured_origin(client):
    r = client.get("/api/state", headers={"Origin": "http://localhost:5173"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_denies_unallowed_origin(client):
    """No allow-origin echo back when origin isn't on the allow-list."""
    r = client.get("/api/state", headers={"Origin": "http://evil.example.com"})
    # Either header missing entirely or set to something OTHER than evil.
    allow = r.headers.get("access-control-allow-origin", "")
    assert allow != "http://evil.example.com"


# ── WS auth ─────────────────────────────────────────────────────────────────


def test_ws_dashboard_default_off_allows_unauth(client):
    """When WS_SHARED_SECRET is unset, any client can connect (matches
    pre-S2.6 behavior; opt-in security)."""
    os.environ.pop("WS_SHARED_SECRET", None)  # ensure unset for this test
    with client.websocket_connect("/ws/dashboard") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "state_sync"


def test_ws_dashboard_requires_token_when_secret_set(client, monkeypatch):
    """When WS_SHARED_SECRET is set, missing/wrong token closes 1008."""
    monkeypatch.setenv("WS_SHARED_SECRET", "test-secret")
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect("/ws/dashboard") as ws:
            ws.receive_json()  # should never get here
    assert excinfo.value.code == 1008


def test_ws_dashboard_accepts_correct_token(client, monkeypatch):
    monkeypatch.setenv("WS_SHARED_SECRET", "test-secret")
    with client.websocket_connect("/ws/dashboard?token=test-secret") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "state_sync"


# ── Spend cap ───────────────────────────────────────────────────────────────


def test_spend_cap_default_off():
    from agents import _spend
    assert _spend.check("bedrock", 999.0) is True  # any amount fine when no cap


def test_spend_cap_enforces_when_set(monkeypatch):
    from agents import _spend
    monkeypatch.setenv("BEDROCK_USD_PER_MIN_CAP", "0.001")
    _spend.record("bedrock", 0.0009)
    assert _spend.check("bedrock", 0.0001) is True   # 0.0009 + 0.0001 = 0.001 (== cap, ok)
    assert _spend.check("bedrock", 0.0002) is False  # would push over
