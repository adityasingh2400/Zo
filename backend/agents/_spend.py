"""Spend cap — per-minute USD ceiling on Bedrock + ElevenLabs calls.

Why: a comment storm against /api/respond_to_comment (or against the audience-
comment QR if it leaks) could burn through your AWS + ElevenLabs budget in
minutes. This module gives a soft ceiling: callers consult `check(provider,
estimated_usd)` before spending; if the rolling per-minute total would exceed
the cap, `check` returns False and the caller falls back to a degraded path
(text-only, cached response, etc.) instead of placing the call.

Default-off: if BEDROCK_USD_PER_MIN_CAP / ELEVENLABS_USD_PER_MIN_CAP env vars
are unset, no cap is enforced. Matches the prior unbounded behavior so this
change doesn't break a fresh clone. Set both in .env to enable.

Estimates are conservative — they don't measure actual usage post-call (we'd
need to parse Bedrock + ElevenLabs response shapes for that). The cap is a
DOS guard, not an accountant. For real cost tracking, BRAIN's per-event
log (cost_saved_usd) is the source of truth.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Deque

logger = logging.getLogger("empire.spend")

# (timestamp, usd_estimate) tuples. Trimmed to the last 60s on every check.
_LOG_LOCK = threading.Lock()
_LOG: dict[str, Deque[tuple[float, float]]] = {}


def _cap_for(provider: str) -> float:
    """Read the per-minute cap from env. Returns 0.0 if unset (no cap)."""
    env_var = f"{provider.upper()}_USD_PER_MIN_CAP"
    raw = os.getenv(env_var, "").strip()
    if not raw:
        return 0.0
    try:
        return float(raw)
    except ValueError:
        logger.warning("[spend] %s=%r is not a float; ignoring", env_var, raw)
        return 0.0


def _trim(provider: str, now: float) -> Deque[tuple[float, float]]:
    """Return the deque for `provider`, dropping entries older than 60s."""
    dq = _LOG.setdefault(provider, deque())
    cutoff = now - 60.0
    while dq and dq[0][0] < cutoff:
        dq.popleft()
    return dq


def usd_in_last_minute(provider: str) -> float:
    """Sum estimated USD spend on this provider in the last 60s."""
    with _LOG_LOCK:
        dq = _trim(provider, time.time())
        return sum(usd for _ts, usd in dq)


def check(provider: str, est_usd: float) -> bool:
    """Returns True if a call costing `est_usd` would NOT push the rolling
    1-minute total over the cap. If the cap is unset (env var missing or 0),
    always returns True. Callers should use the bool to decide whether to
    place the call or fall back to a degraded path.

    `record(provider, usd)` should be called AFTER a successful spend; check
    by itself doesn't update the log. This split lets callers verify before
    spending and only commit on success."""
    cap = _cap_for(provider)
    if cap <= 0:
        return True
    with _LOG_LOCK:
        recent = sum(usd for _ts, usd in _trim(provider, time.time()))
    if recent + est_usd > cap:
        logger.warning(
            "[spend] %s cap exceeded: recent=$%.5f + est=$%.5f > cap=$%.5f",
            provider, recent, est_usd, cap,
        )
        return False
    return True


def record(provider: str, usd: float) -> None:
    """Log a successful spend. Pair with `check` (guard before the call,
    record after the call returns) so failed/aborted calls don't bloat the
    rolling total."""
    if usd <= 0:
        return
    with _LOG_LOCK:
        _trim(provider, time.time()).append((time.time(), usd))


# ── Cost estimates per call ─────────────────────────────────────────────────
# Used by callers as the `est_usd` arg. These are upper-bound estimates for
# typical EMPIRE workloads; the real cost varies with token/char count. Keep
# them conservative (overshoot) so the cap fires before actual spend hits it.
EST_BEDROCK_COMMENT_RESPONSE_USD = 0.0005   # ~150 in + 60 out tokens, Haiku
EST_ELEVENLABS_TTS_PER_RESPONSE_USD = 0.02  # ~60 chars × $0.30/1K chars
