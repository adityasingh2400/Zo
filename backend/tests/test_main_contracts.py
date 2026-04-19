"""Contract tests for the cloud-escalate response path.

Locks in the synthetic `comment_response_video` event that
`api_respond_to_comment` emits after the audio dispatch — Agent A's UI
work could easily break the four dashboard contracts that synthetic
event drives if they refactor any of:
  • setPendingComments filter (clears the pending chip)
  • setVoiceStateSafe(null) (drops the voice state pill)
  • setCommentResponse / setResponseVideo (powers ChatPanel + the
    floating comment overlay on /stage)

These tests use the ACTUAL `api_respond_to_comment` function from
`main.py` with the heavy dependencies stubbed out so we get a fast,
deterministic check on the broadcast shape.

Run:
    cd backend && pytest tests/test_main_contracts.py -v
"""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

# Importing main is expensive (boto3, ollama, fastapi) but everything
# below is module-level state assignment that doesn't make network
# calls until invoked. We import once for the whole module; tests
# patch in/out the network-bound symbols.
import main  # noqa: E402


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def captured_broadcasts():
    """Replace broadcast_to_dashboards with a list-appender that captures
    every event the system under test would have sent to dashboards."""
    captured = []

    async def _capture(payload):
        captured.append(payload)

    with patch.object(main, "broadcast_to_dashboards", side_effect=_capture):
        yield captured


@pytest.fixture
def stub_director():
    """A no-op director that records emit/play/fade calls so tests can
    assert the speaking-idle clip got requested. Async methods all return
    immediately."""
    d = types.SimpleNamespace()
    d.emit = AsyncMock()
    d.play_response = AsyncMock()
    d.fade_to_idle = AsyncMock()
    d.play_clip = AsyncMock()
    d.emit_reading_chat = AsyncMock()
    d.reading_chat = AsyncMock()
    d.fade_intro_overlay = AsyncMock()
    with patch.object(main, "director", d):
        yield d


@pytest.fixture
def stub_pipeline_state():
    """Reset pipeline_state to a clean judge-item-demo baseline so tests
    don't leak state into each other."""
    saved = main.pipeline_state.copy()
    main.pipeline_state.clear()
    main.pipeline_state.update({
        "status": "idle",
        "product_data": {"name": "Coffee Mug", "category": "drinkware"},
        "agent_log": [],
    })
    yield main.pipeline_state
    main.pipeline_state.clear()
    main.pipeline_state.update(saved)


# ── 7a) Cloud-escalate happy path: audio dispatched + synthetic video ───────


def test_synthetic_comment_response_video_on_success(
    captured_broadcasts, stub_director, stub_pipeline_state,
):
    """Cloud-escalate success must emit BOTH:
        1. comment_response_audio  — drives the standalone <audio> +
           KaraokeCaptions
        2. synthetic comment_response_video (url=None) — restores the
           pending-chip clear, voice-state clear, and ChatPanel/overlay
           contracts

    If anyone removes the synthetic event, four UI contracts break
    simultaneously and the dashboard sticks mid-comment.
    """
    with (
        patch.object(main, "generate_comment_response",
                     new=AsyncMock(return_value="Test response — buy this mug")),
        patch.object(main, "classify_comment_gemma",
                     new=AsyncMock(return_value={"type": "question"})),
        patch.object(main, "text_to_speech",
                     new=AsyncMock(return_value=(b"mp3-bytes", []))),
        patch.object(main, "_save_response_audio",
                     return_value="/response_audio/fake.mp3"),
        patch.object(main, "_ensure_reading_chat_visible", new=AsyncMock()),
        patch.object(main, "_release_reading_chat", new=AsyncMock()),
        # _probe_audio_duration_ms is imported at function-call time via
        # `from agents.seller import _probe_audio_duration_ms`, so we
        # have to patch it on the seller module itself.
        patch("agents.seller._probe_audio_duration_ms", return_value=3000),
    ):
        result = asyncio.run(main.api_respond_to_comment(comment="hi mug"))

    audio_events = [b for b in captured_broadcasts if b["type"] == "comment_response_audio"]
    video_events = [b for b in captured_broadcasts if b["type"] == "comment_response_video"]
    assert len(audio_events) == 1, f"expected 1 audio event, got {captured_broadcasts}"
    assert len(video_events) == 1, f"expected 1 synthetic video event, got {captured_broadcasts}"

    audio = audio_events[0]
    assert audio["url"] == "/response_audio/fake.mp3"
    assert audio["comment"] == "hi mug"
    assert audio["response"] == "Test response — buy this mug"
    assert audio["expected_duration_ms"] == 3000
    # Word timings make karaoke work; even an empty list must be present.
    assert "word_timings" in audio
    # comment_response_audio must include intent="response" so the dashboard
    # treats it differently from a pitch audio event.
    assert audio.get("intent") == "response"

    video = video_events[0]
    # The four contract-driving fields:
    assert video["url"] is None, "synthetic event MUST set url=None"
    assert video["comment"] == "hi mug"
    assert video["response"] == "Test response — buy this mug"
    assert video.get("audio_already_playing") is True, \
        "audio_already_playing=True signals the dashboard not to re-play TTS"
    assert video.get("lip_synced") is False, \
        "lip_synced=False marks this as the no-Wav2Lip path"
    # Existing audio URL has to round-trip so the dashboard can correlate
    # the synthetic video event back to its audio sibling.
    assert video.get("existing_audio_url") == "/response_audio/fake.mp3"
    assert video.get("expected_duration_ms") == 3000

    # The function's return contract — used by direct REST callers and by
    # run_routed_comment for telemetry.
    assert result["audio_first"] is True
    assert result["lip_synced"] is False
    assert result["audio_url"] == "/response_audio/fake.mp3"
    assert result["url"] is None
    assert result["audio_duration_ms"] == 3000


def test_speaking_idle_loop_is_emitted(
    captured_broadcasts, stub_director, stub_pipeline_state,
):
    """Director.emit must be called with the speaking-idle URL so the
    avatar visibly mouths through the response. This is what makes the
    no-Wav2Lip path read as 'talking' instead of 'frozen' to the
    audience.
    """
    with (
        patch.object(main, "generate_comment_response",
                     new=AsyncMock(return_value="hello")),
        patch.object(main, "classify_comment_gemma",
                     new=AsyncMock(return_value={"type": "question"})),
        patch.object(main, "text_to_speech",
                     new=AsyncMock(return_value=(b"x", []))),
        patch.object(main, "_save_response_audio",
                     return_value="/response_audio/x.mp3"),
        patch.object(main, "_ensure_reading_chat_visible", new=AsyncMock()),
        patch.object(main, "_release_reading_chat", new=AsyncMock()),
        patch("agents.seller._probe_audio_duration_ms", return_value=2000),
    ):
        asyncio.run(main.api_respond_to_comment(comment="hi"))

    assert stub_director.emit.await_count >= 1, \
        "director.emit() must be invoked for the speaking-idle Tier 1 loop"
    # First call: positional ("tier1", "response", URL)
    args, kwargs = stub_director.emit.call_args
    assert "speaking" in args[2].lower() or kwargs.get("loop") is True, \
        f"speaking-idle URL should look like idle_*_speaking.mp4, got args={args}"
    # Must be muted (audio is the standalone <audio>, not the video).
    assert kwargs.get("muted") is True, \
        "Tier 1 video MUST be muted on the no-Wav2Lip path " \
        "(audio plays from the standalone <audio> element)"
    # intent="response" matters — dashboard's overlayVisible flag keys on it.
    assert args[1] == "response", \
        f"emit intent must be 'response' (drives overlayVisible), got {args[1]}"


# ── 7b) TTS failure: comment_failed broadcast, no orphan state ──────────────


def test_tts_failure_emits_comment_failed(
    captured_broadcasts, stub_director, stub_pipeline_state,
):
    """When ElevenLabs returns empty bytes (rate limit, API error,
    network blip), the dashboard MUST receive comment_failed so it can
    clear the pending chip. Regression: an early version of this path
    would silently return without broadcasting, leaving the dashboard
    stuck."""
    with (
        patch.object(main, "generate_comment_response",
                     new=AsyncMock(return_value="some response")),
        patch.object(main, "classify_comment_gemma",
                     new=AsyncMock(return_value={"type": "question"})),
        # Empty bytes = TTS failed.
        patch.object(main, "text_to_speech",
                     new=AsyncMock(return_value=(b"", []))),
        patch.object(main, "_ensure_reading_chat_visible", new=AsyncMock()),
        patch.object(main, "_release_reading_chat", new=AsyncMock()),
        patch("agents.seller._probe_audio_duration_ms", return_value=0),
    ):
        asyncio.run(main.api_respond_to_comment(comment="hi"))

    failed = [b for b in captured_broadcasts if b["type"] == "comment_failed"]
    assert len(failed) == 1, \
        f"TTS failure must emit one comment_failed event, got {captured_broadcasts}"
    assert failed[0]["comment"] == "hi"
    assert failed[0]["response"] == "some response", \
        "fallback text from drafted response must round-trip so the " \
        "dashboard can show it as a degraded reply"
    assert failed[0].get("reason") == "tts_returned_empty"

    # And NO synthetic comment_response_video on this path — it'd leave
    # the dashboard thinking we have audio playing when we don't.
    video_events = [b for b in captured_broadcasts if b["type"] == "comment_response_video"]
    assert video_events == [], \
        "TTS-failure path must NOT emit comment_response_video " \
        "(would set audio_already_playing=True for nonexistent audio)"


# ── LLM failure: graceful fallback text + audio still dispatches ─────────


def test_llm_failure_falls_back_to_apology_text(
    captured_broadcasts, stub_director, stub_pipeline_state,
):
    """If Bedrock raises (timeout, rate limit, throttling), the function
    catches the exception and uses an apology fallback. TTS still runs
    on the fallback text so the avatar still speaks something rather
    than going silent."""
    with (
        patch.object(main, "generate_comment_response",
                     new=AsyncMock(side_effect=RuntimeError("bedrock_timeout"))),
        patch.object(main, "classify_comment_gemma",
                     new=AsyncMock(return_value={"type": "question"})),
        patch.object(main, "text_to_speech",
                     new=AsyncMock(return_value=(b"audio", []))),
        patch.object(main, "_save_response_audio",
                     return_value="/response_audio/y.mp3"),
        patch.object(main, "_ensure_reading_chat_visible", new=AsyncMock()),
        patch.object(main, "_release_reading_chat", new=AsyncMock()),
        patch("agents.seller._probe_audio_duration_ms", return_value=1500),
    ):
        result = asyncio.run(main.api_respond_to_comment(comment="hi"))

    # The synthetic event still fires — the demo doesn't stick.
    video_events = [b for b in captured_broadcasts if b["type"] == "comment_response_video"]
    assert len(video_events) == 1
    # Response text is the fallback, not blank.
    assert video_events[0]["response"], "fallback text must not be empty"
    assert "back to you" in video_events[0]["response"].lower() \
        or "moment" in video_events[0]["response"].lower() \
        or "let me" in video_events[0]["response"].lower(), (
            f"fallback text should sound like an apology, got: "
            f"{video_events[0]['response']!r}"
        )
    assert result["audio_first"] is True


# ── Reading-chat lifecycle: _ensure + _release fire in correct order ─────


def test_reading_chat_lifecycle_called_around_render(
    captured_broadcasts, stub_director, stub_pipeline_state,
):
    """_ensure_reading_chat_visible MUST fire before render starts,
    _release_reading_chat MUST fire after the work is done. If the
    order inverts or one is dropped, the avatar either doesn't visibly
    "read" the comment (UX regression) or stays stuck in reading
    forever (worse)."""
    call_order = []

    async def ensure_call():
        call_order.append("ensure")

    async def release_call(t0):
        call_order.append("release")

    with (
        patch.object(main, "generate_comment_response",
                     new=AsyncMock(return_value="ok")),
        patch.object(main, "classify_comment_gemma",
                     new=AsyncMock(return_value={"type": "question"})),
        patch.object(main, "text_to_speech",
                     new=AsyncMock(return_value=(b"a", []))),
        patch.object(main, "_save_response_audio",
                     return_value="/response_audio/z.mp3"),
        patch.object(main, "_ensure_reading_chat_visible",
                     side_effect=ensure_call),
        patch.object(main, "_release_reading_chat",
                     side_effect=release_call),
        patch("agents.seller._probe_audio_duration_ms", return_value=1000),
    ):
        asyncio.run(main.api_respond_to_comment(comment="hi"))

    assert call_order == ["ensure", "release"], (
        f"reading-chat lifecycle out of order: {call_order} "
        f"(must be ['ensure', 'release'] — ensure before any heavy work, "
        f"release after work is done but before audio dispatch)"
    )
