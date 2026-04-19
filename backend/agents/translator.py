"""Translator — Claude Haiku translation helper with SQLite cache.

Per EMPIRE PDF: "Real-time multilingual is a stretch goal, not a Phase 1
ship. The architecture supports it (Eleven multilingual voices, Claude
is multilingual, Veo prompts are language-agnostic)."

This module is the first piece: take an English pitch string + a target
language code, return the translated string. Uses Claude Haiku on Bedrock
(same path seller.py uses for pitch generation). Caches (text_hash, lang)
→ translated_text in a sqlite table so re-emitting the same pitch across
languages costs one Bedrock call per unique pitch per language.

v1 scope: 6 languages (en/es/fr/de/zh/tl) declared in SUPPORTED. Expansion
is a .json append. Cache TTL is forever — pitches that need to change
should ship with a new text_hash (reword the pitch).

Cache lives in the same SQLite file brain.py uses (backend/data/brain.db)
but in a separate table (translation_cache) to avoid schema coupling.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import boto3  # type: ignore

from config import AWS_REGION, BEDROCK_MODEL_ID

logger = logging.getLogger("empire.translator")

# Reuse brain's DB location so we don't sprawl sqlite files across data/.
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "brain.db"

# Supported target languages + metadata for ElevenLabs routing. Add rows
# here to expand; no other code changes needed. `eleven_lang` is what
# seller.py passes as language_code when the non-English path is active.
SUPPORTED: dict[str, dict[str, str]] = {
    "en": {"name": "English",    "eleven_lang": "en"},
    "es": {"name": "Spanish",    "eleven_lang": "es"},
    "fr": {"name": "French",     "eleven_lang": "fr"},
    "de": {"name": "German",     "eleven_lang": "de"},
    "zh": {"name": "Mandarin",   "eleven_lang": "zh"},
    "tl": {"name": "Tagalog",    "eleven_lang": "tl"},
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS translation_cache (
    text_hash TEXT NOT NULL,
    lang      TEXT NOT NULL,
    translated TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (text_hash, lang)
);
"""

_thread_local = threading.local()
_pool_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    c = getattr(_thread_local, "conn", None)
    if c is not None:
        return c
    with _pool_lock:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(str(DB_PATH), isolation_level=None)
        c.row_factory = sqlite3.Row
        c.executescript(_SCHEMA)
        _thread_local.conn = c
    return c


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ── Public API ───────────────────────────────────────────────────────────

def get_cached(text: str, lang: str) -> str | None:
    """Return the cached translation or None if not cached."""
    row = _conn().execute(
        "SELECT translated FROM translation_cache WHERE text_hash = ? AND lang = ?",
        (_hash(text), lang),
    ).fetchone()
    return row["translated"] if row else None


def put_cached(text: str, lang: str, translated: str) -> None:
    _conn().execute(
        "INSERT OR REPLACE INTO translation_cache "
        "(text_hash, lang, translated, created_at) VALUES (?, ?, ?, ?)",
        (_hash(text), lang, translated, time.time()),
    )


async def translate(text: str, target_lang: str) -> str:
    """Translate `text` to `target_lang`. English passthrough short-circuits
    (no Bedrock call needed). Cache hits skip the network; misses invoke
    Claude Haiku with a terse translation prompt + store the result.

    Fallback: if Bedrock fails (e.g. AWS creds invalid), returns the
    original text so the pipeline continues in English rather than hanging.
    Callers should check `get_cached(...)` separately if they need to know
    whether a translation actually happened."""
    if target_lang == "en" or not text.strip():
        return text
    if target_lang not in SUPPORTED:
        logger.warning("[translator] unknown target_lang %r, returning original", target_lang)
        return text

    cached = get_cached(text, target_lang)
    if cached is not None:
        logger.info("[translator] cache hit (lang=%s, chars=%d)", target_lang, len(text))
        return cached

    lang_name = SUPPORTED[target_lang]["name"]
    try:
        translated = await _translate_via_bedrock(text, lang_name)
    except Exception as e:
        logger.warning("[translator] Bedrock translate failed, falling back to English: %s", e)
        return text

    put_cached(text, target_lang, translated)
    logger.info("[translator] cached new translation (lang=%s, chars_in=%d, chars_out=%d)",
                target_lang, len(text), len(translated))
    return translated


async def _translate_via_bedrock(text: str, lang_name: str) -> str:
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 400,
        "messages": [{
            "role": "user",
            "content": (
                f"Translate the following English sales pitch to {lang_name}. "
                f"Match the conversational, spoken tone — it'll be read aloud by a "
                f"TTS voice as a livestream sales pitch. Keep the same "
                f"enthusiasm + call-to-action structure. Do not add preamble "
                f'or explanation — output ONLY the translated text.\n\n'
                f'Text to translate:\n"{text}"'
            ),
        }],
    })
    response = await asyncio.to_thread(
        bedrock.invoke_model,
        modelId=BEDROCK_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"].strip()


def stats() -> dict[str, Any]:
    """Diagnostic: per-language cache row counts. Good for verifying that
    translations actually landed in cache after a live demo."""
    rows = _conn().execute(
        "SELECT lang, COUNT(*) AS n FROM translation_cache GROUP BY lang"
    ).fetchall()
    return {
        "supported": list(SUPPORTED.keys()),
        "cache_by_lang": {r["lang"]: r["n"] for r in rows},
    }
