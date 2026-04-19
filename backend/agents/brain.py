"""BRAIN — persistent telemetry + aggregation for the router.

Every routed comment is logged to SQLite (backend/data/brain.db). Aggregates
power the BRAIN dashboard panel and (roadmap) feed conversion-aware
keyword reranking in the router.

Scope (this file):
  - record_event() called once per router decision from main.py.
  - get_stats() returns by-tool counts + cost saved + top answers + top misses.
  - get_top_misses() groups escalate_to_cloud comments by token frequency so
    the operator can see which questions the local Q/A index is missing.

What's NOT here yet (roadmap):
  - Conversion events (BRAIN: conversion-aware Q/A ranking).
  - DM thread persistence (CLOSER).
  - Multi-tenant stream isolation (today stream_id is hardcoded
    "default" because pipeline_state in main.py is process-global).

Connection model: one global sqlite3.Connection opened lazily on first
access. SQLite handles concurrent readers; we serialize writes via the
GIL (record_event is sync — fine for ~hundreds of comments/sec).
"""
from __future__ import annotations

import logging
import re
import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger("empire.brain")

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "brain.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS comment_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    stream_id       TEXT NOT NULL,
    product_id      TEXT NOT NULL,
    comment         TEXT NOT NULL,
    classify_type   TEXT NOT NULL,
    tool            TEXT NOT NULL,
    answer_id       TEXT,
    was_local       INTEGER NOT NULL,
    cost_saved_usd  REAL NOT NULL,
    latency_ms      INTEGER NOT NULL,
    timestamp       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_comment_events_stream    ON comment_events(stream_id);
CREATE INDEX IF NOT EXISTS ix_comment_events_timestamp ON comment_events(timestamp);
CREATE INDEX IF NOT EXISTS ix_comment_events_tool      ON comment_events(tool);
"""

# SQLite handles per-connection. We use one connection per thread because
# sqlite3.Connection objects are not thread-safe by default. The lock guards
# the connection-pool dict itself, not the queries.
_thread_local = threading.local()
_pool_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    """Per-thread SQLite connection. Creates the DB + schema on first call."""
    c = getattr(_thread_local, "conn", None)
    if c is not None:
        return c
    with _pool_lock:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(str(DB_PATH), isolation_level=None)  # autocommit
        c.row_factory = sqlite3.Row
        c.executescript(_SCHEMA)
        _thread_local.conn = c
    return c


# ── Token tokenizer (mirrors router._tokens for top_misses grouping) ─────────
# Used to group escalate_to_cloud comments by the words that didn't match any
# qa_index entry. Stop-words are dropped so "what" / "the" / "is" don't dominate.
_WORD_RE = re.compile(r"[a-z0-9']+")
_STOP_WORDS = {
    "the", "a", "an", "is", "it", "this", "that", "and", "or", "but",
    "of", "to", "in", "on", "at", "for", "with", "from", "by", "as",
    "i", "you", "we", "they", "he", "she", "do", "does", "did", "be",
    "been", "have", "has", "had", "what", "when", "where", "who", "why",
    "how", "your", "my", "our", "their", "yours", "mine", "ours", "theirs",
    "are", "was", "were", "will", "would", "could", "should", "can", "may",
    "might", "must", "shall", "if", "then", "than", "so", "not", "no", "yes",
    "me", "us", "them", "him", "her", "its", "any", "all", "some", "more",
    "most", "less", "much", "few", "very", "just", "really", "also", "too",
}


def _tokens(text: str) -> list[str]:
    return [t for t in _WORD_RE.findall((text or "").lower())
            if t and t not in _STOP_WORDS and len(t) > 1]


# ── Public API ───────────────────────────────────────────────────────────────


def record_event(
    *,
    stream_id: str,
    product_id: str,
    comment: str,
    classify: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    """Log one router decision. Called from main.run_routed_comment after the
    decision lands. Synchronous — sub-millisecond on local SQLite. Failures
    are swallowed (logged at WARN) because BRAIN should never break the router.
    """
    try:
        args = decision.get("args") or {}
        answer_id = args.get("answer_id")  # only present for respond_locally
        _conn().execute(
            """INSERT INTO comment_events
               (stream_id, product_id, comment, classify_type, tool,
                answer_id, was_local, cost_saved_usd, latency_ms, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                stream_id,
                product_id,
                comment[:500],  # truncate runaway comments
                (classify.get("type") or "question").lower(),
                decision["tool"],
                answer_id,
                1 if decision.get("was_local") else 0,
                float(decision.get("cost_saved_usd") or 0.0),
                int(decision.get("ms") or 0),
                time.time(),
            ),
        )
    except Exception as e:
        logger.warning("[brain] record_event failed: %s", e)


def get_stats(
    *,
    stream_id: str | None = None,
    since_seconds: float | None = None,
) -> dict[str, Any]:
    """Aggregate stats. Filters: stream_id (None = all), since_seconds (None
    = all time). Returns the dict shape the dashboard's BrainPanel renders.
    """
    conn = _conn()
    where = []
    params: list[Any] = []
    if stream_id:
        where.append("stream_id = ?")
        params.append(stream_id)
    if since_seconds is not None and since_seconds > 0:
        where.append("timestamp >= ?")
        params.append(time.time() - since_seconds)
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""

    total_row = conn.execute(
        f"SELECT COUNT(*) AS n, COALESCE(SUM(cost_saved_usd), 0) AS saved, "
        f"COALESCE(AVG(latency_ms), 0) AS avg_ms FROM comment_events{where_sql}",
        params,
    ).fetchone()
    total = total_row["n"]
    total_cost_saved = round(float(total_row["saved"]), 5)
    avg_latency_ms = int(total_row["avg_ms"]) if total else 0

    # By-tool breakdown.
    by_tool_rows = conn.execute(
        f"SELECT tool, COUNT(*) AS n FROM comment_events{where_sql} GROUP BY tool",
        params,
    ).fetchall()
    by_tool = {row["tool"]: row["n"] for row in by_tool_rows}

    # Top-N most-matched answer_ids (respond_locally only — others have
    # answer_id = NULL by construction).
    top_answer_rows = conn.execute(
        f"SELECT answer_id, COUNT(*) AS n FROM comment_events"
        f"{where_sql}{' AND' if where else ' WHERE'} answer_id IS NOT NULL "
        f"GROUP BY answer_id ORDER BY n DESC LIMIT 10",
        params,
    ).fetchall()
    top_answers = [{"answer_id": row["answer_id"], "count": row["n"]}
                   for row in top_answer_rows]

    # Top-N missed tokens — words that recur in escalate_to_cloud comments.
    miss_comment_rows = conn.execute(
        f"SELECT comment FROM comment_events"
        f"{where_sql}{' AND' if where else ' WHERE'} tool = 'escalate_to_cloud' "
        f"ORDER BY timestamp DESC LIMIT 200",
        params,
    ).fetchall()
    token_counter: Counter[str] = Counter()
    for row in miss_comment_rows:
        token_counter.update(_tokens(row["comment"]))
    top_misses = [{"token": tok, "count": n}
                  for tok, n in token_counter.most_common(10)]

    # Local-vs-cloud rate. The headline KPI for the moat narrative.
    local_n = sum(by_tool.get(t, 0) for t in (
        "respond_locally", "play_canned_clip", "block_comment"))
    pct_local = int(round(100 * local_n / total)) if total else 0

    return {
        "total": total,
        "local_n": local_n,
        "pct_local": pct_local,
        "total_cost_saved_usd": total_cost_saved,
        "avg_latency_ms": avg_latency_ms,
        "by_tool": by_tool,
        "top_answers": top_answers,
        "top_misses": top_misses,
    }
