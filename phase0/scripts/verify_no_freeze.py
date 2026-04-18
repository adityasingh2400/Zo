#!/usr/bin/env python
"""Verify rendered response mp4s contain no frozen frame ranges.

Scans every response.mp4 for runs of consecutive frames where the
between-frame mean absolute pixel difference is below a freeze threshold.
A long enough freeze (>= --max-freeze-ms) fails the asset.

This catches three classes of bug:
  * Wav2Lip dropped a batch — black/missing frames midway
  * face-detect missed a chunk — last_box reuse loop got stuck
  * encoder hiccup — duplicate frames pasted multiple times

Usage:
  python phase0/scripts/verify_no_freeze.py                        # last 10 renders
  python phase0/scripts/verify_no_freeze.py --all                  # every resp_*.mp4
  python phase0/scripts/verify_no_freeze.py path/to/file.mp4       # one file
  python phase0/scripts/verify_no_freeze.py --threshold 1.5        # tighter freeze
"""
from __future__ import annotations
import argparse
import pathlib
import subprocess
import sys
import json

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RENDERS = ROOT.parent / "backend" / "renders"
REPORT = ROOT / "bench" / "results" / "verify_no_freeze.json"
REPORT.parent.mkdir(parents=True, exist_ok=True)


def scan(video_path: pathlib.Path,
         freeze_threshold: float = 0.5,
         max_freeze_ms: int = 600) -> dict:
    """Walk every frame, compute per-frame mean-abs-diff vs prior frame.
    Detect contiguous runs where diff < threshold.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"video": str(video_path), "ok": False, "error": "cannot open"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = n_frames / fps if fps > 0 else 0

    prev_small = None
    diffs: list[float] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Downsample for fast diff (256 wide max)
        h, w = frame.shape[:2]
        scale = 256 / max(w, h)
        if scale < 1:
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small = frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype("float32")
        if prev_small is not None:
            d = float(np.mean(np.abs(gray - prev_small)))
            diffs.append(d)
        prev_small = gray
    cap.release()

    if not diffs:
        return {"video": str(video_path), "ok": False, "error": "no frames"}

    # Detect freeze runs: consecutive frames where diff < threshold.
    freeze_runs: list[tuple[int, int, float]] = []  # (start_idx, length, mean_diff)
    i = 0
    while i < len(diffs):
        if diffs[i] < freeze_threshold:
            start = i
            while i < len(diffs) and diffs[i] < freeze_threshold:
                i += 1
            length = i - start
            if length >= 2:
                run_diffs = diffs[start:i]
                freeze_runs.append((start, length, float(np.mean(run_diffs))))
        else:
            i += 1

    max_freeze_frames = int(max_freeze_ms / 1000 * fps)
    bad_runs = [r for r in freeze_runs if r[1] > max_freeze_frames]
    longest = max((r[1] for r in freeze_runs), default=0)
    longest_ms = int(longest / fps * 1000) if fps else 0

    return {
        "video": str(video_path),
        "ok": len(bad_runs) == 0,
        "fps": round(fps, 2),
        "frames": len(diffs) + 1,
        "duration_sec": round(duration, 2),
        "mean_diff": round(float(np.mean(diffs)), 3),
        "min_diff": round(float(np.min(diffs)), 3),
        "freeze_runs": [
            {"start_frame": r[0], "length_frames": r[1],
             "length_ms": int(r[1] / fps * 1000), "mean_diff": round(r[2], 3)}
            for r in freeze_runs
        ],
        "longest_freeze_ms": longest_ms,
        "max_freeze_ms_threshold": max_freeze_ms,
        "bad_runs_count": len(bad_runs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="specific mp4s to scan (default: last 10 renders)")
    ap.add_argument("--all", action="store_true", help="scan every resp_*.mp4")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="per-frame diff value below which a frame counts as frozen vs prior (default 0.5)")
    ap.add_argument("--max-freeze-ms", type=int, default=600,
                    help="freeze runs longer than this fail the file (default 600ms)")
    args = ap.parse_args()

    if args.paths:
        targets = [pathlib.Path(p) for p in args.paths]
    elif args.all:
        targets = sorted(RENDERS.glob("resp_*.mp4"))
    else:
        targets = sorted(RENDERS.glob("resp_*.mp4"),
                         key=lambda p: p.stat().st_mtime)[-10:]

    if not targets:
        print(f"No targets to scan in {RENDERS}")
        sys.exit(0)

    results = []
    for path in targets:
        r = scan(path, freeze_threshold=args.threshold, max_freeze_ms=args.max_freeze_ms)
        results.append(r)
        flag = "OK  " if r.get("ok") else "FAIL"
        if "error" in r:
            print(f"[{flag}] {path.name}  error: {r['error']}")
        else:
            print(f"[{flag}] {path.name:60} "
                  f"frames={r['frames']:3d} dur={r['duration_sec']}s "
                  f"mean_diff={r['mean_diff']} "
                  f"longest_freeze={r['longest_freeze_ms']}ms "
                  f"({len(r['freeze_runs'])} runs)")

    REPORT.write_text(json.dumps(results, indent=2))
    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"\n{n_ok}/{len(results)} passed   ->   {REPORT}")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
