#!/usr/bin/env python
"""GFPGAN A/B benchmark — same audio + source rendered with and without GFPGAN.

Hits the lipsync server twice per source: once with X-Enhancer expected to be
'gfpgan' (server started with GFPGAN_ENABLED=1), once with the legacy unsharp
path (server restarted with GFPGAN_ENABLED=0). Drops both renders side by side
into phase0/bench/videos/ab/ so we can eyeball the quality lift.

Run order from a laptop with the SSH tunnel live (localhost:8010):

  # 1. On the pod, restart with GFPGAN ON:
  #    GFPGAN_ENABLED=1 python /workspace/wav2lip_server_v2.py
  python phase0/scripts/gfpgan_ab.py --label gfpgan

  # 2. On the pod, restart with GFPGAN OFF:
  #    GFPGAN_ENABLED=0 python /workspace/wav2lip_server_v2.py
  python phase0/scripts/gfpgan_ab.py --label unsharp

  # 3. Compare:
  open phase0/bench/videos/ab/
"""
from __future__ import annotations
import argparse, json, os, pathlib, time, httpx

ROOT = pathlib.Path(__file__).resolve().parent.parent
SERVER = os.environ.get("LIPSYNC_URL", "http://localhost:8010")
OUT_DIR = ROOT / "bench" / "videos" / "ab"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "pitching_silent":   ROOT / "assets" / "states" / "state_pitching_pose_silent.mp4",
    "pitching_speaking": ROOT / "assets" / "states" / "state_pitching_pose_speaking.mp4",
    "explaining":        ROOT / "assets" / "states" / "state_explaining_pose_speaking_1080p.mp4",
}
AUDIO = ROOT / "bench" / "audio" / "pitch_10s.mp3"


def health() -> dict:
    r = httpx.get(f"{SERVER}/health", timeout=10)
    r.raise_for_status()
    return r.json()


def render(label: str, source_label: str, source: pathlib.Path) -> dict:
    if not source.exists():
        print(f"[{source_label}] skip — {source} missing")
        return {"label": label, "source": source_label, "error": "missing source"}
    if not AUDIO.exists():
        raise SystemExit(f"audio missing: {AUDIO}")

    print(f"[{label}/{source_label}] POST")
    t0 = time.perf_counter()
    with source.open("rb") as v, AUDIO.open("rb") as a:
        files = {
            "video": (source.name, v, "video/mp4"),
            "audio": (AUDIO.name, a, "audio/mpeg"),
        }
        # out_height=1080 to match production; full quality A/B
        r = httpx.post(f"{SERVER}/lipsync", files=files, data={"out_height": "1080"}, timeout=300)
    wall = time.perf_counter() - t0

    if r.status_code != 200:
        print(f"[{label}/{source_label}] FAIL {r.status_code}: {r.text[:200]}")
        return {"label": label, "source": source_label, "wall_sec": round(wall, 2),
                "error": f"HTTP {r.status_code}", "body": r.text[:500]}

    out = OUT_DIR / f"{source_label}__{label}.mp4"
    out.write_bytes(r.content)
    h = r.headers
    info = {
        "label": label,
        "source": source_label,
        "wall_sec": round(wall, 2),
        "total_sec": float(h.get("X-Total-Sec", 0)),
        "predict_sec": float(h.get("X-Predict-Sec", 0)),
        "enhance_sec": float(h.get("X-Enhance-Sec", 0)),
        "enhancer": h.get("X-Enhancer", "?"),
        "size_kb": len(r.content) // 1024,
        "out": str(out),
    }
    print(f"[{label}/{source_label}] OK wall={info['wall_sec']}s total={info['total_sec']}s "
          f"predict={info['predict_sec']}s enhance={info['enhance_sec']}s "
          f"({info['enhancer']}) -> {out.name} ({info['size_kb']}KB)")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="label for this run, e.g. gfpgan or unsharp")
    ap.add_argument("--sources", nargs="*", default=list(SOURCES.keys()),
                    help="subset of source labels to render (default: all)")
    ap.add_argument("--runs", type=int, default=2, help="renders per source — first cold, second warm p50")
    args = ap.parse_args()

    h = health()
    print(f"server: {SERVER}  enhancer={h.get('enhancer')}  stride={h.get('gfpgan_stride')}  "
          f"face_cache={h.get('face_cache_size')}\n")

    expected = "gfpgan" if args.label == "gfpgan" else "unsharp"
    if h.get("enhancer") != expected:
        print(f"WARN: server reports enhancer={h.get('enhancer')!r}, expected {expected!r}.")
        print(f"      Restart the pod server with GFPGAN_ENABLED={'1' if expected == 'gfpgan' else '0'}.")

    results = []
    for src_label in args.sources:
        if src_label not in SOURCES:
            print(f"skip unknown source: {src_label}"); continue
        for i in range(args.runs):
            run_label = f"{args.label}_run{i}"
            results.append(render(run_label, src_label, SOURCES[src_label]))

    report = ROOT / "bench" / "results" / f"gfpgan_ab_{args.label}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, indent=2))

    # Quick summary: warm-only p50 per source
    print("\n=== summary (warm runs only) ===")
    by_src: dict[str, list[float]] = {}
    for r in results:
        if "error" in r:
            continue
        if r["label"].endswith("_run0"):  # cold, skip from p50
            continue
        by_src.setdefault(r["source"], []).append(r["total_sec"])
    for src, totals in by_src.items():
        p50 = sorted(totals)[len(totals) // 2] if totals else 0
        print(f"  {src:20s} warm p50 total={p50}s  samples={totals}")
    print(f"\nreport: {report}")


if __name__ == "__main__":
    main()
