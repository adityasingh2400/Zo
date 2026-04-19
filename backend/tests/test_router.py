"""Parametrized tests for the four-tool comment router.

Guards the demo-critical paths: if someone edits the keyword lists or
products.json and breaks one of the four tool dispatches, CI catches it
before the demo.

Run from the repo root:
    cd backend && pytest tests/test_router.py -v
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

# Make `agents.router` importable without a full FastAPI boot.
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from agents import router  # noqa: E402

# The demo product, trimmed to just the fields the router reads.
WALLET = {
    "name": "Minimal Leather Wallet",
    "qa_index": {
        "is_it_real_leather": {
            "keywords": ["real leather", "genuine", "material", "leather"],
            "text": "...",
            "url": "/local_answers/wallet_real_leather.mp4",
        },
        "return_policy": {
            "keywords": ["return it", "return policy", "refund", "returns"],
            "text": "...",
            "url": "/local_answers/wallet_returns.mp4",
        },
        "price": {
            "keywords": ["price", "cost", "how much", "dollars"],
            "text": "...",
            "url": "/local_answers/wallet_price.mp4",
        },
    },
}


def _decide(comment: str, classify_type: str = "question", product=WALLET) -> dict:
    """Helper so each case stays a single readable line."""
    classify = {"type": classify_type}
    return asyncio.run(router.decide(comment, classify, product))


# ── Four-tool coverage — if any of these break, the demo breaks. ─────────────


@pytest.mark.parametrize(
    "comment,classify_type,expected_tool,expected_intent",
    [
        # respond_locally: matched product Q&A keys (any classify type —
        # the local-match check now runs BEFORE intent-based routing so
        # price / shipping / sizing always hit the sub-300 ms path).
        ("is it real leather",              "question",   "respond_locally", None),
        ("how much does this cost",         "question",   "respond_locally", None),
        ("what's your return policy",       "question",   "respond_locally", None),
        # escalate_to_cloud (= bridge+wav2lip): everything non-spam,
        # non-locally-matched. Compliments + objections now also flow
        # here per the avatar choreography spec — the dispatcher's
        # _run_bridge_with_wav2lip lip-syncs Gemma's draft response onto
        # the intent-specific bridge clip as substrate. The intent_hint
        # arg lets the dispatcher pick the right bucket.
        ("how does this compare to the Apple Watch", "question",  "escalate_to_cloud", "question"),
        ("do you ship to Mars",                       "question",  "escalate_to_cloud", "question"),
        ("I love this wallet",                        "compliment","escalate_to_cloud", "compliment"),
        ("this is amazing",                           "compliment","escalate_to_cloud", "compliment"),
        ("this is overpriced",                        "objection", "escalate_to_cloud", "objection"),
        ("feels cheap to me",                         "objection", "escalate_to_cloud", "objection"),
        # block_comment: Gemma classified as spam OR URL / promo cue.
        ("buy followers at sketchy.site/promo", "spam",    "block_comment", None),
        ("free money https://scam.com",     "question",   "block_comment", None),
        ("check out my link",               "question",   "block_comment", None),
    ],
)
def test_four_tool_dispatch(comment, classify_type, expected_tool, expected_intent):
    decision = _decide(comment, classify_type)
    assert decision["tool"] == expected_tool, (
        f"Expected {expected_tool!r} for {comment!r} (classify={classify_type!r}), "
        f"got {decision['tool']!r} — reason: {decision.get('reason')}"
    )
    if expected_intent is not None:
        assert decision["args"].get("intent_hint") == expected_intent, (
            f"Expected intent_hint={expected_intent!r}, "
            f"got {decision['args'].get('intent_hint')!r}"
        )


# ── Emoji-only compliments — word-tokenizer strips emoji, so we check the
# raw-string emoji cue list. Regression target: "❤️" alone used to get
# escalated as a generic question. Now they correctly route to
# escalate_to_cloud with intent_hint=compliment so the dispatcher fires
# a compliment-bucket bridge clip.


@pytest.mark.parametrize("comment", ["❤️", "this is amazing 🔥", "😍"])
def test_emoji_compliments_route_as_compliment(comment):
    # Pass classify=question so we're forcing the emoji cue to win, not
    # piggybacking off Gemma's compliment classification.
    decision = _decide(comment, classify_type="question")
    assert decision["tool"] == "escalate_to_cloud"
    assert decision["args"].get("intent_hint") == "compliment"


# ── "return" must match only when it's about the product, not when the
# customer asks "will you return my email" etc. Regression target: the
# keywords used to be single tokens like "return", which false-matched.


def test_return_only_matches_product_returns():
    d1 = _decide("will you return my email later")
    assert d1["tool"] == "escalate_to_cloud", f"false match: {d1}"

    d2 = _decide("what's your return policy")
    assert d2["tool"] == "respond_locally"


# ── Router must NEVER crash when no product is loaded (e.g. a fresh
# backend that hasn't ingested any product yet). Regression target:
# KeyError on product["qa_index"] used to take the whole request down.


def test_no_product_falls_through_to_cloud():
    decision = _decide("is it real leather", product=None)
    assert decision["tool"] == "escalate_to_cloud"


# ── Cost model invariants. Judges may grep the code; these asserts keep
# the accounting honest.


def test_cost_saved_usd_invariants():
    # escalate_to_cloud never claims savings
    assert router.COST_SAVED_USD_PER_TOOL["escalate_to_cloud"] == 0.0
    # All non-cloud paths claim the same uniform save (matches pitch math)
    save = router.COST_SAVED_USD_PER_TOOL["respond_locally"]
    assert save > 0
    assert router.COST_SAVED_USD_PER_TOOL["play_canned_clip"] == save
    assert router.COST_SAVED_USD_PER_TOOL["block_comment"] == save


# ── The returned decision shape is stable; RoutingPanel + cost KPI rely
# on these fields existing.


def test_decision_shape_is_stable():
    d = _decide("is it real leather")
    for key in ("tool", "args", "reason", "ms", "was_local", "cost_saved_usd"):
        assert key in d, f"missing key: {key}"
    assert isinstance(d["ms"], int)
    assert isinstance(d["was_local"], bool)
    assert d["was_local"] is True  # local path
    # Escalate → was_local is False
    d2 = _decide("compare to Apple Watch")
    assert d2["was_local"] is False
