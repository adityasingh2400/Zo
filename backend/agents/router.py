"""Comment router — picks one of four tools per incoming comment.

The router is the "local-first" wedge: most live-stream comments are
routine (greetings, price questions, spam) and shouldn't pay a cloud
LLM round-trip. The router inspects the comment + on-device Gemma
classify output + loaded product index and routes to:

    respond_locally(answer_id)   instant pre-rendered answer clip
    play_canned_clip(label)      stock acknowledgement via bridge clips
    block_comment(reason)        spam filter — no visual, just a counter
    escalate_to_cloud(comment)   the full Claude + TTS + Wav2Lip path

Hour 4-5 ships the rule-based primary. Hour 6-7 swaps in FunctionGemma
on Gemma 4 as the primary decider, with rule-based kept as the fallback
when Cactus is unavailable or the FunctionGemma call errors.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

logger = logging.getLogger("empire.router")


# ── Cost model ──────────────────────────────────────────────────────────────
# Dollar value avoided per non-escalating decision. Rough estimate:
# Bedrock Claude Haiku at ~$0.00025 input + $0.00125 output per 1K tokens,
# typical comment_response uses ~1000 in + 150 out. Round to $0.00035 per
# dodged call to stay defensible on stage.
COST_PER_CLOUD_COMMENT_USD = 0.00035

COST_SAVED_USD_PER_TOOL = {
    "respond_locally":  COST_PER_CLOUD_COMMENT_USD,
    "play_canned_clip": COST_PER_CLOUD_COMMENT_USD,
    "block_comment":    COST_PER_CLOUD_COMMENT_USD,
    "escalate_to_cloud": 0.0,
}


# ── Tool schema (for FunctionGemma in Hour 6-7) ─────────────────────────────
# Shape matches Cactus' `tools_json` param for cactus_complete. Kept here so
# the FunctionGemma call path and the rule-based fallback share the exact
# same set of valid tools and arg keys.
TOOL_SCHEMA: list[dict[str, Any]] = [
    {
        "name": "respond_locally",
        "description": (
            "Use for routine product questions the seller has pre-authored "
            "answers for (materials, price, shipping, sizing, returns, "
            "warranty). Args: answer_id matches a key in the product index."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "answer_id": {"type": "string"},
            },
            "required": ["answer_id"],
        },
    },
    {
        "name": "escalate_to_cloud",
        "description": (
            "Use for cross-product comparisons, opinions, anything that "
            "requires live reasoning or context the seller hasn't pre-baked. "
            "Args: the original comment text."
        ),
        "parameters": {
            "type": "object",
            "properties": {"comment": {"type": "string"}},
            "required": ["comment"],
        },
    },
    {
        "name": "play_canned_clip",
        "description": (
            "Use to acknowledge compliments or soft objections without "
            "generating a new response. Args: label is one of 'compliment', "
            "'objection', or 'neutral'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "enum": ["compliment", "objection", "neutral"],
                },
            },
            "required": ["label"],
        },
    },
    {
        "name": "block_comment",
        "description": (
            "Use for spam, abuse, or off-topic promotion. No visual response. "
            "Args: short reason tag."
        ),
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    },
]


# ── Product index matcher ────────────────────────────────────────────────────
# A product loaded from backend/data/products.json has a `qa_index` map:
#
#   {
#     "is_it_real_leather": {
#       "keywords": ["real leather", "genuine", "material"],
#       "text": "Yes, full-grain vegetable-tanned leather.",
#       "url":  "/local_answers/wallet_real_leather.mp4"
#     },
#     ...
#   }
#
# _match_product_field returns the first answer_id whose keywords all match
# as substrings of the comment (case-insensitive). Returns None if nothing
# hits — caller escalates to cloud.

_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall((text or "").lower()))


def _match_product_field(comment: str, product: dict | None) -> str | None:
    """Return the answer_id in product['qa_index'] with the best keyword
    match against the comment. Case-insensitive. An entry "matches" if at
    least one of its keywords appears in the comment — multi-word keywords
    checked as substrings, single-word keywords checked against the token
    set to avoid "ship" matching "relationship".

    Tie-breaker: the entry with the most keyword hits wins. First insertion
    wins on ties (Python dicts preserve order), so put more specific
    answers earlier in products.json.

    Returns None if no product loaded or nothing matches."""
    if not product:
        return None
    qa = product.get("qa_index") or {}
    if not qa:
        return None
    c_tokens = _tokens(comment)
    c_lower = (comment or "").lower()

    best_id: str | None = None
    best_hits = 0
    for answer_id, entry in qa.items():
        keywords = entry.get("keywords") or []
        hits = 0
        for kw in keywords:
            kw_l = kw.lower().strip()
            if not kw_l:
                continue
            if " " in kw_l:
                if kw_l in c_lower:
                    hits += 1
            else:
                if kw_l in c_tokens:
                    hits += 1
        if hits > best_hits:
            best_hits = hits
            best_id = answer_id
    return best_id


# ── Rule-based decider ──────────────────────────────────────────────────────

_COMPLIMENT_CUES = {"love", "lovely", "beautiful", "cute", "amazing", "awesome",
                    "perfect", "great", "nice", "cool", "gorgeous", "stunning"}
# Emoji / multi-char cues checked as substrings on the full lowered comment
# (the word-tokenizer drops emoji). Keep the list short — false positives here
# are worse than missing a compliment.
_COMPLIMENT_EMOJI = ("❤️", "😍", "🔥", "💯", "✨", "😊")

_OBJECTION_CUES = {"expensive", "overpriced", "scam", "fake", "cheap",
                   "rip-off", "ripoff", "pricey"}

# Spam cues: URL-like or hard-promotion only. Words like "buy", "follow",
# "visit", "dm" are too ambiguous — legitimate customers ask "where do I
# buy this?" or "dm me the details." Gemma 4's classify is the primary
# spam signal; this list is the URL-heavy safety net.
_SPAM_CUES = ("http://", "https://", ".com/", ".net/", "www.",
              "promo code", "subscribe to", "check out my")


def _rule_based_decide(comment: str, classify: dict, product: dict | None) -> dict:
    """Decision logic when Cactus FunctionGemma isn't available (Hour 4-5
    default; Hour 6-7 promotes FunctionGemma to primary and keeps this as
    the fallback). Returns {tool, args, reason}."""
    t = (classify or {}).get("type", "question")
    c_lower = (comment or "").lower()
    c_tokens = _tokens(comment)

    # 1. Spam filter — either Gemma said so OR obvious URL-spam cues. Words
    #    like "buy" / "follow" are too ambiguous for commerce (real customers
    #    ask "where do I buy this?"), so we stick to URL / promo-phrase cues.
    if t == "spam" or any(cue in c_lower for cue in _SPAM_CUES):
        return {
            "tool": "block_comment",
            "args": {"reason": "spam"},
            "reason": "Classified as spam / URL spam",
        }

    # 2. Compliment → acknowledge with a canned clip. No LLM, no render.
    #    Emoji-only comments like "❤️" land here via the emoji cue list
    #    because the word-tokenizer strips emoji.
    if (t == "compliment"
            or (c_tokens & _COMPLIMENT_CUES)
            or any(em in comment for em in _COMPLIMENT_EMOJI)):
        return {
            "tool": "play_canned_clip",
            "args": {"label": "compliment"},
            "reason": "Acknowledging compliment with pre-rendered bridge clip",
        }

    # 3. Objection → canned reassurance ("totally hear you on that").
    if t == "objection" or (c_tokens & _OBJECTION_CUES):
        return {
            "tool": "play_canned_clip",
            "args": {"label": "objection"},
            "reason": "Defusing objection with pre-rendered bridge clip",
        }

    # 4. Question we've pre-authored an answer for → respond_locally.
    if t == "question" and product:
        hit = _match_product_field(comment, product)
        if hit:
            return {
                "tool": "respond_locally",
                "args": {"answer_id": hit},
                "reason": f"Matched product.qa_index key: {hit}",
            }

    # 5. Default — anything requiring real reasoning goes to cloud.
    return {
        "tool": "escalate_to_cloud",
        "args": {"comment": comment},
        "reason": "Needs cloud reasoning / no local match",
    }


# ── Public API ───────────────────────────────────────────────────────────────

async def decide(
    comment: str,
    classify: dict,
    product: dict | None = None,
) -> dict:
    """Public entry. Currently rule-based; Hour 6-7 wraps this with a
    FunctionGemma call and keeps the rule-based path as the fallback.

    Returns a dict with:
      tool              one of {respond_locally, escalate_to_cloud,
                                play_canned_clip, block_comment}
      args              tool-specific kwargs
      reason            human-readable explanation (shown on RoutingPanel)
      ms                decide latency in ms
      was_local         True if the dispatched tool avoids the cloud
      cost_saved_usd    estimated $ avoided vs cloud escalation
    """
    t0 = time.time()
    decision = _rule_based_decide(comment, classify, product)
    tool = decision["tool"]
    decision["ms"] = int((time.time() - t0) * 1000)
    decision["was_local"] = tool != "escalate_to_cloud"
    decision["cost_saved_usd"] = COST_SAVED_USD_PER_TOOL.get(tool, 0.0)
    logger.info("[router] %s — %s (%dms)", tool, decision["reason"], decision["ms"])
    return decision
