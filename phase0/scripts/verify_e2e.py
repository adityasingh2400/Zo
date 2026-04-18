#!/usr/bin/env python
"""P1.6: end-to-end verification on 10 arbitrary items.

Hits /api/respond_to_comment with 10 diverse comments, verifies:
  - HTTP 200
  - response.url is downloadable
  - render is at least 720p tall, valid h264, has audio
  - total_ms within budget (default 12s warm)
  - response_text is non-empty and ≤ 25 words

Writes a report to phase0/bench/results/e2e_verify.json.

Bar: ≥95% success rate.

Usage:
  python phase0/scripts/verify_e2e.py
  python phase0/scripts/verify_e2e.py --base http://127.0.0.1:8000 --budget-ms 12000
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys, time
import httpx

ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench" / "results"
BENCH.mkdir(parents=True, exist_ok=True)

# Diverse comment surface area: questions, objections, compliments,
# specific spec asks, and one "spammy" / off-topic to test classifier graceful
# degradation. None are too long so the avatar can stay snappy.
ITEMS = [
    ("What size will fit a small wrist?",      "question"),
    ("Is this real leather or faux?",          "question"),
    ("Does it ship internationally?",          "question"),
    ("How long does the battery last?",        "question"),
    ("Returns policy?",                        "question"),
    ("Looks gorgeous on you!",                 "compliment"),
    ("This is way too expensive for what it is.", "objection"),
    ("Cheaper on Amazon, why buy here?",       "objection"),
    ("first lol",                              "spam"),
    ("Can it sync with my Garmin app?",        "question"),
]


def probe(path: pathlib.Path) -> dict:
    """ffprobe video stream + audio presence."""
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams",
         "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    if p.returncode != 0:
        return {"error": p.stderr.strip()}
    info = json.loads(p.stdout or "{}")
    streams = info.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), {})
    a = next((s for s in streams if s.get("codec_type") == "audio"), {})
    return {
        "width": v.get("width"),
        "height": v.get("height"),
        "video_codec": v.get("codec_name"),
        "video_bitrate": int(v.get("bit_rate") or 0) if v else 0,
        "audio_codec": a.get("codec_name"),
        "audio_channels": a.get("channels"),
        "duration_sec": float(v.get("duration") or 0) if v else 0,
    }


def run(base: str, budget_ms: int, out_height: int) -> list[dict]:
    results = []
    for i, (comment, expected) in enumerate(ITEMS, 1):
        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=120.0) as c:
                r = c.post(f"{base}/api/respond_to_comment",
                           data={"comment": comment, "out_height": str(out_height)})
                r.raise_for_status()
                payload = r.json()
        except Exception as e:
            results.append({
                "i": i, "comment": comment, "expected": expected,
                "ok": False, "stage": "http", "error": str(e),
            })
            print(f"[{i:2d}] HTTP FAIL: {e}")
            continue

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        url = payload.get("url", "")
        response_text = (payload.get("response") or "").strip()
        word_count = len(response_text.split())

        # Download the video and probe it locally
        out = BENCH.parent.parent / "renders_e2e" / f"item_{i:02d}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            with httpx.Client(timeout=60.0) as c:
                vr = c.get(f"{base}{url}")
                vr.raise_for_status()
                out.write_bytes(vr.content)
        except Exception as e:
            results.append({
                "i": i, "comment": comment, "expected": expected,
                "ok": False, "stage": "download", "error": str(e),
                "payload": payload,
            })
            print(f"[{i:2d}] DOWNLOAD FAIL: {e}")
            continue

        probe_info = probe(out)
        size_kb = out.stat().st_size // 1024

        # Pass criteria
        checks = {
            "has_url":           bool(url),
            "has_response":      bool(response_text),
            "response_short":    word_count <= 25,
            "video_tall_enough": (probe_info.get("height") or 0) >= 720,
            "video_h264":        probe_info.get("video_codec") == "h264",
            "video_has_audio":   bool(probe_info.get("audio_codec")),
            "within_budget":     payload.get("total_ms", 0) <= budget_ms,
        }
        ok = all(checks.values())

        results.append({
            "i": i, "comment": comment, "expected": expected,
            "ok": ok, "checks": checks,
            "elapsed_ms": elapsed_ms,
            "total_ms": payload.get("total_ms"),
            "breakdown": payload.get("breakdown"),
            "wav2lip": payload.get("wav2lip"),
            "response": response_text,
            "word_count": word_count,
            "url": url,
            "size_kb": size_kb,
            "probe": probe_info,
        })
        flag = "OK" if ok else "FAIL"
        b = payload.get("breakdown", {})
        print(f"[{i:2d}] {flag} {payload.get('total_ms','?')}ms "
              f"(c={b.get('classify_ms','?')}ms l={b.get('llm_ms','?')}ms "
              f"t={b.get('tts_ms','?')}ms w={b.get('lipsync_ms','?')}ms) "
              f"{probe_info.get('width')}x{probe_info.get('height')} "
              f"{size_kb}KB  {response_text[:60]!r}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--budget-ms", type=int, default=12000,
                    help="warm latency budget (ms) per item")
    ap.add_argument("--out-height", type=int, default=1080)
    args = ap.parse_args()

    print(f"E2E verify against {args.base} "
          f"(budget {args.budget_ms}ms, out_height {args.out_height})")
    print(f"  {len(ITEMS)} items\n")

    results = run(args.base, args.budget_ms, args.out_height)
    n = len(results)
    n_ok = sum(1 for r in results if r["ok"])
    pass_rate = n_ok / n if n else 0
    totals = [r["total_ms"] for r in results if r.get("total_ms")]
    p50 = sorted(totals)[len(totals) // 2] if totals else None
    p95 = sorted(totals)[max(0, int(len(totals) * 0.95) - 1)] if totals else None

    summary = {
        "n": n, "n_ok": n_ok, "pass_rate": round(pass_rate, 3),
        "p50_ms": p50, "p95_ms": p95,
        "budget_ms": args.budget_ms,
        "out_height": args.out_height,
        "items": results,
    }
    out = BENCH / "e2e_verify.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n=== summary ===")
    print(f"  pass_rate: {n_ok}/{n} ({pass_rate*100:.0f}%)")
    print(f"  p50: {p50}ms  p95: {p95}ms")
    print(f"  report: {out}")

    sys.exit(0 if pass_rate >= 0.95 else 1)


if __name__ == "__main__":
    main()
