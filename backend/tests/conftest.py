"""Shared pytest fixtures for backend/tests/.

Pattern: TestClient is created WITHOUT the lifespan context (no `with`),
so we don't trigger the real Cactus model load, rembg pool prewarm, or
Director instantiation. Tests that exercise routed-comment paths set
pipeline_state manually + monkeypatch classify_comment_gemma.

Why no lifespan: the lifespan loads Gemma 4 (8 GB on first call), starts
the Director's idle rotation, and prewarms the rembg pool. None of that
is meaningful in a smoke-test context and all of it slows the suite +
introduces flakiness.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))


@pytest.fixture
def client():
    """FastAPI TestClient. NOT a context manager — skips the lifespan so
    Cactus/Gemma doesn't load. Tests requiring product_data must set it
    via the `with_product` fixture."""
    from fastapi.testclient import TestClient

    from main import app
    return TestClient(app)


@pytest.fixture
def with_product():
    """Populate pipeline_state with a minimal product so routed-comment
    paths have a qa_index to match against. Cleans up after the test."""
    from main import pipeline_state
    snapshot = dict(pipeline_state)

    test_product = {
        "name": "Test Wallet",
        "qa_index": {
            "is_it_real_leather": {
                "keywords": ["real leather", "leather"],
                "text": "Yes, real leather.",
                "url": "/local_answers/wallet_real_leather.mp4",
            },
        },
    }
    pipeline_state["products_catalog"] = {"test_wallet": test_product}
    pipeline_state["product_data"] = test_product
    pipeline_state["active_product_id"] = "test_wallet"

    yield test_product

    # Restore.
    pipeline_state.clear()
    pipeline_state.update(snapshot)


@pytest.fixture(autouse=True)
def _isolated_brain_db(tmp_path, monkeypatch):
    """Point BRAIN's SQLite at a per-test tmp file so tests don't pollute
    the dev DB (and so each test starts from a clean event log). Also
    forces a fresh per-test connection by clearing the thread-local cache."""
    from agents import brain
    monkeypatch.setattr(brain, "DB_PATH", tmp_path / "brain.db")
    # Clear any cached connection so the next access opens against tmp_path.
    if hasattr(brain._thread_local, "conn"):
        del brain._thread_local.conn
    yield
    if hasattr(brain._thread_local, "conn"):
        try:
            brain._thread_local.conn.close()
        except Exception:
            pass
        del brain._thread_local.conn
