#!/usr/bin/env python
"""Phase 0.3b — CRITICAL GATE: Lip sync speed benchmark across fal.ai models.

This is THE most important decision in Phase 0. It determines:
  - Winning lip-sync model (for both pre-rendered clips and live demo-time)
  - Talk-track commitment on stage ("under 30s" vs "under a minute")

Tests (fires in parallel — credits are plentiful per user):
  MODEL                                     SOURCE SUBSTRATE       RUNS
  1. fal-ai/sync-lipsync/v3 (sync-3)        silent + speaking      2
  2. fal-ai/sync-lipsync/v2 (lipsync-2)     silent + speaking      2
  3. fal-ai/sync-lipsync/v2/pro (2-pro)     silent + speaking      2
  4. creatify/lipsync (Aurora)              silent + speaking      2
  5. veed/lipsync (Fabric)                  silent + speaking      2
  TOTAL: 10 renders, all parallel

Metrics:
  - wall_clock_sec  (submit -> downloaded mp4)
  - http_queue_sec  (submit -> queued)
  - http_render_sec (started -> completed)
  - output quality (manual visual grading 1-5 after completion)

Requires: FAL_KEY in .env
"""
from __future__ import annotations
import os, sys, time, json, pathlib, concurrent.futures, traceback
from dotenv import dotenv_values

ROOT = pathlib.Path(__file__).resolve().parent.parent
env = dotenv_values(ROOT.parent / ".env")
for k, v in env.items():
    os.environ[k] = v

if not os.environ.get("FAL_KEY"):
    print("ERROR: FAL_KEY missing from .env.", file=sys.stderr)
    print("Create key at https://fal.ai/dashboard/keys and add: FAL_KEY=<key>", file=sys.stderr)
    sys.exit(2)

import fal_client

AUDIO = ROOT / "bench" / "audio" / "pitch_10s.mp3"
OUT_DIR = ROOT / "bench" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR = ROOT / "bench" / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# Source substrates (provided on the command line; defaults assume Phase 0.2c generated them)
SRC_SILENT = ROOT / "assets" / "states" / "state_pitching_pose_silent.mp4"
SRC_SPEAKING = ROOT / "assets" / "states" / "state_pitching_pose_speaking.mp4"

# ==== MODELS ==================================================================
# Each entry: (label, fal_model_id, payload_factory(video_url, audio_url) -> dict)

def sync_v3_payload(video_url, audio_url):
    return {"video_url": video_url, "audio_url": audio_url, "sync_mode": "cut_off"}

def sync_v2_payload(video_url, audio_url):
    return {"video_url": video_url, "audio_url": audio_url, "model": "lipsync-2", "sync_mode": "cut_off"}

def sync_v2_pro_payload(video_url, audio_url):
    return {"video_url": video_url, "audio_url": audio_url, "model": "lipsync-2-pro", "sync_mode": "cut_off"}

def creatify_payload(video_url, audio_url):
    return {"video_url": video_url, "audio_url": audio_url}

def veed_payload(video_url, audio_url):
    return {"video_url": video_url, "audio_url": audio_url}

MODELS = [
    ("sync-3",        "fal-ai/sync-lipsync/v3",  sync_v3_payload),
    ("lipsync-2",     "fal-ai/sync-lipsync/v2",  sync_v2_payload),
    ("lipsync-2-pro", "fal-ai/sync-lipsync/v2",  sync_v2_pro_payload),
    ("creatify",      "creatify/lipsync",        creatify_payload),
    ("veed-fabric",   "veed/lipsync",            veed_payload),
]


def upload(path: pathlib.Path) -> str:
    print(f"   upload {path.name}...", end="", flush=True)
    t = time.perf_counter()
    url = fal_client.upload_file(str(path))
    print(f" {time.perf_counter()-t:.2f}s")
    return url


def run_bench(model_label: str, source_label: str, model_id: str, payload_fn, video_url: str, audio_url: str):
    full_label = f"{model_label}/{source_label}"
    t0 = time.perf_counter()
    try:
        args = payload_fn(video_url, audio_url)
        print(f"[{full_label}] submitting  payload={list(args.keys())}")
        t_submit = time.perf_counter()
        result = fal_client.subscribe(
            model_id,
            arguments=args,
            with_logs=False,
        )
        wall = time.perf_counter() - t0
        # Extract output video URL — varies per model
        out_url = None
        if isinstance(result, dict):
            if "video" in result and isinstance(result["video"], dict):
                out_url = result["video"].get("url")
            elif "url" in result:
                out_url = result["url"]
        if not out_url:
            return {"label": full_label, "error": f"no video url in result: {str(result)[:300]}", "wall_clock_sec": wall}

        # Download
        import httpx
        t_dl = time.perf_counter()
        r = httpx.get(out_url, timeout=60)
        r.raise_for_status()
        out_path = VIDEO_DIR / f"bench_{model_label}_{source_label}.mp4"
        out_path.write_bytes(r.content)
        dl_sec = time.perf_counter() - t_dl

        total = time.perf_counter() - t0
        print(f"[{full_label}] DONE  total={total:.1f}s  dl={dl_sec:.1f}s  size={out_path.stat().st_size//1024}KB  -> {out_path.name}")
        return {
            "label": full_label,
            "model": model_label,
            "source": source_label,
            "fal_model_id": model_id,
            "wall_clock_sec": round(total, 2),
            "download_sec": round(dl_sec, 2),
            "output_path": str(out_path),
            "output_size_kb": out_path.stat().st_size // 1024,
        }
    except Exception as e:
        wall = time.perf_counter() - t0
        err = traceback.format_exc(limit=2)
        print(f"[{full_label}] FAILED after {wall:.1f}s: {str(e)[:200]}")
        return {"label": full_label, "error": str(e), "traceback": err, "wall_clock_sec": round(wall,2)}


def main(test_source: str = "both"):
    # Upload assets ONCE, reuse URLs across all fan-out calls
    audio_url = upload(AUDIO)

    sources = {}
    if SRC_SILENT.exists() and test_source in ("both", "silent"):
        sources["silent"] = upload(SRC_SILENT)
    if SRC_SPEAKING.exists() and test_source in ("both", "speaking"):
        sources["speaking"] = upload(SRC_SPEAKING)
    if not sources:
        print(f"No source videos found. Expected: {SRC_SILENT} and/or {SRC_SPEAKING}")
        sys.exit(3)
    print(f"Testing {len(MODELS)} models x {len(sources)} sources = {len(MODELS)*len(sources)} renders\n")

    tasks = []
    for source_label, video_url in sources.items():
        for model_label, model_id, payload_fn in MODELS:
            tasks.append((model_label, source_label, model_id, payload_fn, video_url, audio_url))

    results = []
    t_bench_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        futs = {ex.submit(run_bench, *t): t[0]+"/"+t[1] for t in tasks}
        for f in concurrent.futures.as_completed(futs):
            results.append(f.result())

    bench_total = time.perf_counter() - t_bench_start
    print(f"\n======================================================")
    print(f"  BENCHMARK COMPLETE — wall-clock {bench_total:.1f}s for {len(tasks)} parallel renders")
    print(f"======================================================")

    ok = [r for r in results if "error" not in r]
    ok.sort(key=lambda r: r["wall_clock_sec"])
    print(f"\n{'rank':<5}{'label':<28}{'seconds':>10}   output")
    for i, r in enumerate(ok):
        print(f"{i+1:<5}{r['label']:<28}{r['wall_clock_sec']:>9.1f}s   {pathlib.Path(r['output_path']).name}")

    failed = [r for r in results if "error" in r]
    if failed:
        print(f"\nFAILED ({len(failed)}):")
        for r in failed:
            print(f"  {r['label']}: {r['error'][:200]}")

    report = OUT_DIR / "lipsync_bench.json"
    report.write_text(json.dumps({"bench_total_sec": round(bench_total, 1),
                                  "results": results}, indent=2))
    print(f"\nFull report: {report}")

if __name__ == "__main__":
    test_source = sys.argv[1] if len(sys.argv) > 1 else "both"
    main(test_source)
