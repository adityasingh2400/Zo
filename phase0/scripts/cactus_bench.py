#!/usr/bin/env python
"""Phase 0.4 — Verify Gemma 4 comment classification latency via Cactus SDK.

Target: <500ms round-trip. Runs 5 representative comments through the classifier
using the existing backend pipeline and reports p50/p90.
"""
from __future__ import annotations
import os, sys, time, pathlib, asyncio, json
from dotenv import dotenv_values

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = ROOT.parent
env = dotenv_values(REPO / ".env")
for k, v in env.items():
    os.environ[k] = v

sys.path.insert(0, str(REPO / "backend"))
from agents.eyes import classify_comment_gemma, CACTUS_AVAILABLE, _get_cactus_model

TEST_COMMENTS = [
    ("how much does this cost?", "question"),
    ("omg i love it!!!", "compliment"),
    ("looks kinda cheap tho", "objection"),
    ("WHERE DO I BUY", "call_to_action"),
    ("yooo this is fire", "hype"),
]


async def main():
    print(f"Cactus available: {CACTUS_AVAILABLE}")
    if CACTUS_AVAILABLE:
        print("Warming up Cactus (Gemma 4) model...")
        t = time.perf_counter()
        await asyncio.to_thread(_get_cactus_model)
        print(f"  warmup {time.perf_counter()-t:.2f}s")

    results = []
    for comment, expected in TEST_COMMENTS:
        t = time.perf_counter()
        r = await classify_comment_gemma(comment)
        rt = (time.perf_counter() - t) * 1000  # ms
        ok = r.get("type", "?") == expected or expected in str(r).lower()
        print(f"  {'✓' if ok else '·'} {comment!r:40s} -> type={r.get('type','?'):<12} rt={rt:.0f}ms  src={r.get('source','?')}")
        results.append({"comment": comment, "expected": expected, "got": r.get("type"), "rt_ms": round(rt, 1), "source": r.get("source")})

    rts = sorted(r["rt_ms"] for r in results)
    p50 = rts[len(rts)//2]
    p90 = rts[int(len(rts)*0.9)]
    print(f"\np50={p50:.0f}ms  p90={p90:.0f}ms  (target <500ms)")
    print("GATE:", "PASS" if p50 < 500 else "FAIL — classification too slow for live demo comments")

    out = ROOT / "bench" / "results" / "cactus_bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results, "p50_ms": p50, "p90_ms": p90}, indent=2))
    print(f"Report: {out}")

if __name__ == "__main__":
    asyncio.run(main())
