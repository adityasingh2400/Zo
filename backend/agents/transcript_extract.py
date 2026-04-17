"""On-device structured extraction from a seller's spoken transcript.

Runs in PARALLEL with Claude vision during video intake.
- Cactus (Gemma 4 E4B) first, Ollama fallback, regex fallback if both fail.
- Returns structured hints that the dashboard can show immediately (~500ms)
  while Claude vision (~3s) refines them.
- Hints are also injected into Claude's prompt to make vision faster + grounded
  in what the seller actually said.

Output schema (always returns at least the fallback shape):
    {
      "name_hint":            "Casio F-91W watch" | None,
      "category_hint":        "watches" | None,
      "claims":               ["water resistant", "stainless steel"],
      "selling_points":       ["affordable classic", "iconic design"],
      "target_audience_hint": "minimalist enthusiasts" | None,
      "price_hint":           "$20-$30" | None,
      "source":               "cactus" | "ollama" | "regex_fallback",
      "latency_ms":           int,
    }
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

logger = logging.getLogger("empire.transcript_extract")

EMPTY_EXTRACT: dict[str, Any] = {
    "name_hint": None,
    "category_hint": None,
    "claims": [],
    "selling_points": [],
    "target_audience_hint": None,
    "price_hint": None,
    "source": "empty",
    "latency_ms": 0,
}

PROMPT_TMPL = (
    "You are extracting structured product info from a seller's spoken pitch on a livestream.\n"
    "Pitch (verbatim transcript):\n"
    '"""{transcript}"""\n\n'
    "Return ONLY a single JSON object. No prose, no code fences. Schema:\n"
    "{{\n"
    '  "name_hint": "best-guess product name (str or null)",\n'
    '  "category_hint": "broad category like watches/sneakers/headphones (str or null)",\n'
    '  "claims": ["short factual claims the seller made, max 6"],\n'
    '  "selling_points": ["benefits/why-buy phrases the seller emphasized, max 5"],\n'
    '  "target_audience_hint": "who would buy this (str or null)",\n'
    '  "price_hint": "any price the seller named, e.g. $25 or null"\n'
    "}}\n"
    "Be terse. If the seller did not mention something, use null or []."
)


def _parse_json(text: str) -> dict | None:
    s, e = text.find("{"), text.rfind("}") + 1
    if s < 0 or e <= s:
        return None
    try:
        return json.loads(text[s:e])
    except json.JSONDecodeError:
        return None


def _normalize(d: dict, source: str, latency_ms: int) -> dict[str, Any]:
    """Coerce a raw model dict into our schema, dropping junk."""
    def _str_or_none(v):
        if isinstance(v, str) and v.strip() and v.lower() not in ("null", "none", "n/a"):
            return v.strip()
        return None

    def _str_list(v, max_n: int):
        if not isinstance(v, list):
            return []
        out = [str(x).strip() for x in v if isinstance(x, (str, int, float)) and str(x).strip()]
        return out[:max_n]

    return {
        "name_hint": _str_or_none(d.get("name_hint")),
        "category_hint": _str_or_none(d.get("category_hint")),
        "claims": _str_list(d.get("claims"), 6),
        "selling_points": _str_list(d.get("selling_points"), 5),
        "target_audience_hint": _str_or_none(d.get("target_audience_hint")),
        "price_hint": _str_or_none(d.get("price_hint")),
        "source": source,
        "latency_ms": latency_ms,
    }


def _regex_fallback(transcript: str) -> dict[str, Any]:
    """Best-effort extraction with no LLM at all. Cheap signal beats nothing."""
    t = transcript.strip()
    if not t:
        return EMPTY_EXTRACT.copy()

    price_match = re.search(r"\$\s?\d{1,4}(?:[.,]\d{1,2})?(?:\s?-\s?\$?\d{1,4})?", t)
    sentences = [s.strip() for s in re.split(r"[.!?]+", t) if len(s.strip()) > 4]

    return {
        "name_hint": None,
        "category_hint": None,
        "claims": sentences[:3],
        "selling_points": [],
        "target_audience_hint": None,
        "price_hint": price_match.group(0) if price_match else None,
        "source": "regex_fallback",
        "latency_ms": 0,
    }


async def extract_transcript_signals(transcript: str) -> dict[str, Any]:
    """Extract structured product hints from a seller's spoken transcript.

    Order: Cactus -> Ollama -> regex fallback. Always returns the schema.
    """
    transcript = (transcript or "").strip()
    if not transcript:
        return EMPTY_EXTRACT.copy()

    if len(transcript) < 20:
        logger.info("Transcript <20 chars, regex fallback only")
        return _regex_fallback(transcript)

    prompt = PROMPT_TMPL.format(transcript=transcript[:1500])
    t0 = time.time()

    try:
        from agents.eyes import CACTUS_AVAILABLE, _cactus_chat  # noqa: WPS433
        if CACTUS_AVAILABLE:
            import asyncio
            result = await asyncio.to_thread(
                _cactus_chat, [{"role": "user", "content": prompt}], 320,
            )
            text = result.get("response", "")
            parsed = _parse_json(text)
            if parsed:
                ms = int((time.time() - t0) * 1000)
                logger.info("Cactus extract: %d chars in %dms", len(text), ms)
                return _normalize(parsed, "cactus", ms)
            logger.warning("Cactus returned non-JSON, falling through")
    except Exception as e:
        logger.warning("Cactus extract failed: %s", e)

    try:
        from agents.eyes import _ollama_chat  # noqa: WPS433
        result = await _ollama_chat([{"role": "user", "content": prompt}], max_tokens=320)
        if "error" not in result:
            text = result.get("message", {}).get("content", "")
            parsed = _parse_json(text)
            if parsed:
                ms = int((time.time() - t0) * 1000)
                logger.info("Ollama extract: %d chars in %dms", len(text), ms)
                return _normalize(parsed, "ollama", ms)
            logger.warning("Ollama returned non-JSON, falling through")
    except Exception as e:
        logger.warning("Ollama extract failed: %s", e)

    logger.info("LLM extract unavailable, using regex fallback")
    return _regex_fallback(transcript)


def hint_block_for_claude(extract: dict[str, Any]) -> str:
    """Format the extract as a compact context block to inject into Claude's prompt.

    Empty if we have no signal — no point in noisy prompts.
    """
    if not extract or extract.get("source") == "empty":
        return ""

    lines: list[str] = []
    if name := extract.get("name_hint"):
        lines.append(f"- likely product: {name}")
    if cat := extract.get("category_hint"):
        lines.append(f"- likely category: {cat}")
    if claims := extract.get("claims"):
        lines.append(f"- seller claims: {'; '.join(claims)}")
    if sps := extract.get("selling_points"):
        lines.append(f"- seller emphasized: {'; '.join(sps)}")
    if aud := extract.get("target_audience_hint"):
        lines.append(f"- likely audience: {aud}")
    if price := extract.get("price_hint"):
        lines.append(f"- price mentioned: {price}")

    if not lines:
        return ""
    return "ON-DEVICE TRANSCRIPT EXTRACT (use as hints, image still wins on visual details):\n" + "\n".join(lines)
