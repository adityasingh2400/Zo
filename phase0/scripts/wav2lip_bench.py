#!/usr/bin/env python
"""Phase 0.3b — Lip sync benchmark against Wav2Lip on RunPod 5090.

Tests silent-mouth vs speaking-mouth source substrate A/B with our 10s TTS audio.
Measures end-to-end wall clock from client perspective (upload + render + download).
"""
from __future__ import annotations
import os, sys, time, pathlib, concurrent.futures, json
from dotenv import dotenv_values

ROOT = pathlib.Path(__file__).resolve().parent.parent
env = dotenv_values(ROOT.parent / ".env")
for k, v in env.items():
    os.environ[k] = v

import httpx

ENDPOINT = "http://127.0.0.1:8010/lipsync"
HEALTH   = "http://127.0.0.1:8010/health"

AUDIO = ROOT / "bench" / "audio" / "pitch_10s.mp3"
SOURCES = {
    "silent":   ROOT / "assets" / "states" / "state_pitching_pose_silent.mp4",
    "speaking": ROOT / "assets" / "states" / "state_pitching_pose_speaking.mp4",
}
OUT_DIR = ROOT / "bench" / "videos"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run(label: str, source_path: pathlib.Path, run_idx: int):
    t0 = time.perf_counter()
    try:
        with source_path.open("rb") as v, AUDIO.open("rb") as a:
            files = {
                "video": (source_path.name, v, "video/mp4"),
                "audio": (AUDIO.name, a, "audio/mpeg"),
            }
            r = httpx.post(ENDPOINT, files=files, timeout=180)
        r.raise_for_status()
        out = OUT_DIR / f"wav2lip_{label}_run{run_idx}.mp4"
        out.write_bytes(r.content)
        wall = time.perf_counter() - t0
        render_s = float(r.headers.get("X-Render-Seconds", "0"))
        size_kb = len(r.content) // 1024
        print(f"[{label} run{run_idx}] wall={wall:5.2f}s  render={render_s:5.2f}s  size={size_kb}KB  -> {out.name}")
        return {"label": label, "run": run_idx, "wall_sec": round(wall, 2),
                "render_sec": round(render_s, 2), "size_kb": size_kb, "path": str(out)}
    except httpx.HTTPStatusError as e:
        print(f"[{label} run{run_idx}] HTTP {e.response.status_code}: {e.response.text[:200]}")
        return {"label": label, "run": run_idx, "error": f"HTTP {e.response.status_code}: {e.response.text[:500]}"}
    except Exception as e:
        print(f"[{label} run{run_idx}] FAIL: {e}")
        return {"label": label, "run": run_idx, "error": str(e)}


def main():
    print("Health:", httpx.get(HEALTH, timeout=10).json())
    print()

    # Cold-start first render captures queue warm-up cost; then 2 warm renders for p50
    for source_label, path in SOURCES.items():
        if not path.exists():
            print(f"SKIP {source_label}: source missing at {path}")

    results = []
    for source_label, path in SOURCES.items():
        if not path.exists():
            continue
        for i in range(3):  # 3 runs each: cold, warm, warm
            r = run(source_label, path, i)
            results.append(r)
            time.sleep(0.5)  # gentle

    print("\n================ SUMMARY ================")
    by_source = {}
    for r in results:
        if "error" in r: continue
        by_source.setdefault(r["label"], []).append(r["render_sec"])
    for src, times in by_source.items():
        print(f"  {src:10s}  cold={times[0]:.2f}s  warm_p50={sorted(times[1:])[len(times[1:])//2]:.2f}s  all={[round(t,2) for t in times]}")

    fails = [r for r in results if "error" in r]
    if fails:
        print(f"\n  FAILED: {len(fails)}")
        for r in fails:
            print(f"    {r['label']}/run{r['run']}: {r['error'][:200]}")

    report = ROOT / "bench" / "results" / "wav2lip_bench.json"
    report.write_text(json.dumps(results, indent=2))
    print(f"\nReport: {report}")


if __name__ == "__main__":
    main()
