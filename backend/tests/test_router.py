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
    "comment,classify_type,expected_tool",
    [
        # respond_locally: all three local Q&A keys.
        ("is it real leather",              "question",   "respond_locally"),
        ("how much does this cost",         "question",   "respond_locally"),
        ("what's your return policy",       "question",   "respond_locally"),
        # escalate_to_cloud: question we can't answer locally.
        ("how does this compare to the Apple Watch", "question", "escalate_to_cloud"),
        ("do you ship to Mars",             "question",   "escalate_to_cloud"),
        # play_canned_clip: compliments + objections.
        ("I love this wallet",              "compliment", "play_canned_clip"),
        ("this is amazing",                 "compliment", "play_canned_clip"),
        ("this is overpriced",              "objection",  "play_canned_clip"),
        ("feels cheap to me",               "objection",  "play_canned_clip"),
        # block_comment: Gemma classified as spam OR URL / promo cue.
        ("buy followers at sketchy.site/promo", "spam",    "block_comment"),
        ("free money https://scam.com",     "question",   "block_comment"),
        ("check out my link",               "question",   "block_comment"),
    ],
)
def test_four_tool_dispatch(comment, classify_type, expected_tool):
    decision = _decide(comment, classify_type)
    assert decision["tool"] == expected_tool, (
        f"Expected {expected_tool!r} for {comment!r} (classify={classify_type!r}), "
        f"got {decision['tool']!r} — reason: {decision.get('reason')}"
    )


# ── Emoji-only compliments — word-tokenizer strips emoji, so we check the
# raw-string emoji cue list. Regression target: "❤️" alone got escalated
# before this was added.


@pytest.mark.parametrize("comment", ["❤️", "this is amazing 🔥", "😍"])
def test_emoji_compliments_stay_local(comment):
    # Pass classify=question so we're forcing the emoji cue to win, not
    # piggybacking off Gemma's compliment classification.
    decision = _decide(comment, classify_type="question")
    assert decision["tool"] == "play_canned_clip"


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
