"""Trace-id-based pipeline logging.

Every API entrypoint that triggers a multi-phase pipeline (comment routing,
voice intake, sell pipeline, etc.) calls `new_trace()` to mint a short id
and capture a t0. Subsequent `phase(label, **kv)` calls log a single line
with the trace id, elapsed ms since t0, the phase label, and any key=value
extras.

At the end of a pipeline run, `summary(label, **kv)` prints a clean banner
with the total elapsed time and any final state. Greppable by trace id —
filtering one comment's full lifecycle is `grep "trace abc1234" backend.log`.

Why contextvars and not a kwarg: pipelines fan out into asyncio.create_task
helpers (Director.reading_chat, Wav2Lip render, audio-first dispatch). All of
those should inherit the parent's trace id without each function having to
thread it through. ContextVar Just Works across `await` and inside tasks
spawned with `asyncio.create_task` (Python copies the current context).

Usage:
    from agents.trace import new_trace, phase, summary

    async def api_respond_to_comment(comment, ...):
        new_trace("respond_to_comment")
        phase("comment_received", text=comment[:60], len=len(comment))
        ...
        phase("tts_done", ms=tts_ms)
        ...
        summary("respond_to_comment", total_ms=..., audio=True, video=False)
"""
from __future__ import annotations

import contextvars
import logging
import time
import uuid
from typing import Any

logger = logging.getLogger("zo.trace")

# Per-task trace state. ContextVar copies on asyncio.create_task so child
# tasks inherit the parent trace by default — exactly what we want for the
# director.reading_chat() / wav2lip-render fan-out pattern.
_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "trace_id", default=None
)
_trace_t0: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "trace_t0", default=None
)
_trace_label: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "trace_label", default=None
)


def new_trace(label: str = "?") -> str:
    """Mint a short trace id, capture t0, set as the current context's trace.
    Returns the id so the caller can include it in HTTP responses if useful.

    A short 7-char hex id is more readable in terminal logs than a full UUID
    while still giving 268M unique ids — collision risk over a hackathon
    demo's lifetime is essentially zero.
    """
    tid = uuid.uuid4().hex[:7]
    _trace_id.set(tid)
    _trace_t0.set(time.time())
    _trace_label.set(label)
    logger.info("[trace %s] %4dms %-22s label=%s", tid, 0, "trace_start", label)
    return tid


def get_id() -> str:
    """Return the current trace id, or '-------' if none active."""
    return _trace_id.get() or "-------"


def phase(label: str, **kv: Any) -> None:
    """Log a single phase line with elapsed-since-t0 timing.

    Format: `[trace abc1234]  XXXms <label-padded-to-22>  k1=v1 k2=v2 ...`
    Elapsed is right-aligned to 4 chars (handles 0-9999ms cleanly; >10s is
    rare for a single phase but still readable).

    Extras are stringified with `repr()` for strings (preserves quotes) and
    direct str() for everything else. Long strings are truncated at 80
    chars so a long comment doesn't blow out one log line.
    """
    tid = _trace_id.get() or "-------"
    t0 = _trace_t0.get()
    at_ms = int((time.time() - t0) * 1000) if t0 else 0
    extras = " ".join(f"{k}={_fmt(v)}" for k, v in kv.items())
    logger.info("[trace %s] %4dms %-22s %s", tid, at_ms, label, extras)


def summary(label: str, **kv: Any) -> None:
    """End-of-pipeline summary line. Same format as phase() but with a
    visual marker (===) so it's easy to spot when scanning logs."""
    tid = _trace_id.get() or "-------"
    t0 = _trace_t0.get()
    at_ms = int((time.time() - t0) * 1000) if t0 else 0
    extras = " ".join(f"{k}={_fmt(v)}" for k, v in kv.items())
    logger.info("[trace %s] %4dms === SUMMARY %s         %s", tid, at_ms, label, extras)


def _fmt(v: Any) -> str:
    if v is None:
        return "None"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    if len(s) > 80:
        s = s[:77] + "..."
    if isinstance(v, str):
        # Quote strings so spaces / colons in values don't look like new keys.
        return f'"{s}"'
    return s
