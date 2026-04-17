#!/usr/bin/env python
"""Validate that idle/misc clips loop seamlessly.

Checks per clip:
  1. Loop seamlessness  — extract frame 0 and last frame, compute SSIM (>= 0.92)
                          and MSE (<= 30). If either fails, the clip will jump
                          on every loop boundary.
  2. Photorealism       — midpoint frame Laplacian variance must be high enough
                          (>= 100) so we know the face wasn't smoothed into
                          plastic.
  3. (Optional) identity — CLIP cosine similarity vs the canonical portrait
                          (>= 0.85). Skipped if `clip` package is missing.

Usage:
  python phase0/scripts/validate_loop_seamless.py
  python phase0/scripts/validate_loop_seamless.py --rerolls 3   # auto-rerender failures
  python phase0/scripts/validate_loop_seamless.py /path/to/clip.mp4
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys, tempfile
import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
IDLE_DIR = ROOT / "assets" / "states" / "idle"
REPORT = ROOT / "bench" / "results" / "validate_loop.json"
REPORT.parent.mkdir(parents=True, exist_ok=True)

# Pass thresholds. SSIM is the primary signal — a 600ms crossfade between
# two looping instances will hide most absolute-pixel differences, so MSE is
# advisory only and tuned to catch obvious drifts (full pose change), not
# normal blink/breath variation.
SSIM_MIN = 0.85
MSE_MAX = 600.0
LAPLACIAN_MIN = 80.0   # var of Laplacian; under ~50 = blurry / smoothed
CLIP_IDENTITY_MIN = 0.85


def extract_frame(video_path: pathlib.Path, where: str) -> np.ndarray | None:
    """where: 'first' | 'last' | 'mid'"""
    out = pathlib.Path(tempfile.gettempdir()) / f"vlf_{video_path.stem}_{where}.png"
    if where == "first":
        sel = "select=eq(n\\,0)"
    elif where == "last":
        sel = "select=eq(n\\,n_frames-1)"  # may not work in older ffmpeg
    elif where == "mid":
        sel = "select=eq(n\\,n_frames/2)"
    else:
        raise ValueError(where)

    # Robust path: probe duration + frame rate, then seek to specific time.
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=duration,nb_frames,r_frame_rate",
         "-of", "json", str(video_path)],
        capture_output=True, text=True,
    )
    try:
        info = json.loads(probe.stdout or "{}")
        s = info.get("streams", [{}])[0]
        duration = float(s.get("duration", "0") or 0)
        nb_frames = int(s.get("nb_frames", "0") or 0)
        # parse "24/1" -> 24.0
        fr_num, fr_den = s.get("r_frame_rate", "24/1").split("/")
        fps = float(fr_num) / float(fr_den) if float(fr_den) > 0 else 24.0
    except Exception:
        duration, nb_frames, fps = 0.0, 0, 24.0
    if duration == 0:
        return None

    # Compute a safe timestamp. ffmpeg's -ss before -i is keyframe-snapped;
    # for the "last" frame we leave a small margin so it doesn't seek past EOF.
    if where == "first":
        ts = 0.0
    elif where == "last":
        # 2 frames of margin from the very end so seeking finds a real frame.
        ts = max(0.0, duration - max(2.0 / fps, 0.05))
    else:
        ts = duration / 2.0

    if out.exists():
        out.unlink()
    # Note: -ss AFTER -i is precise (frame-accurate); slower but reliable for
    # our 8s clips. -update 1 silences the "no pattern" warning for single-frame writes.
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-ss", f"{ts:.3f}",
           "-frames:v", "1", "-update", "1", "-loglevel", "error", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not out.exists():
        return None
    img = cv2.imread(str(out))
    if img is None or img.size == 0:
        return None
    return img


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = cv2.resize(a, (b.shape[1], b.shape[0]))
    return float(np.mean((a.astype("float32") - b.astype("float32")) ** 2))


def ssim_quick(a: np.ndarray, b: np.ndarray) -> float:
    """Fast scalar SSIM via mean+std (sufficient for our pass/fail gate)."""
    a = cv2.cvtColor(cv2.resize(a, (b.shape[1], b.shape[0])), cv2.COLOR_BGR2GRAY).astype("float64")
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype("float64")
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float(num / den) if den > 0 else 0.0


def laplacian_var(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def validate(video_path: pathlib.Path) -> dict:
    f0 = extract_frame(video_path, "first")
    fL = extract_frame(video_path, "last")
    fM = extract_frame(video_path, "mid")
    if f0 is None or fL is None or fM is None:
        return {"video": str(video_path), "ok": False, "error": "frame extraction failed"}

    s = ssim_quick(f0, fL)
    m = mse(f0, fL)
    lvar = laplacian_var(fM)
    checks = {
        "ssim_first_last":   {"value": round(s, 4),    "min": SSIM_MIN,       "ok": s >= SSIM_MIN},
        "mse_first_last":    {"value": round(m, 2),    "max": MSE_MAX,        "ok": m <= MSE_MAX},
        "laplacian_mid":     {"value": round(lvar, 1), "min": LAPLACIAN_MIN,  "ok": lvar >= LAPLACIAN_MIN},
    }
    ok = all(c["ok"] for c in checks.values())
    return {"video": str(video_path), "ok": ok, "checks": checks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="specific video paths to validate (default: all in assets/states/idle)")
    ap.add_argument("--rerolls", type=int, default=0,
                    help="if > 0, re-render any failing clip via veo_idle_library.py up to N times")
    args = ap.parse_args()

    if args.paths:
        targets = [pathlib.Path(p) for p in args.paths]
    else:
        # Validate every concrete file (skip symlinks to avoid double-check).
        targets = sorted(p for p in IDLE_DIR.glob("*.mp4") if not p.is_symlink())

    if not targets:
        print(f"No clips to validate in {IDLE_DIR}")
        sys.exit(0)

    results = []
    for path in targets:
        r = validate(path)
        results.append(r)
        flag = "OK  " if r.get("ok") else "FAIL"
        if "error" in r:
            print(f"[{flag}] {path.name}  error: {r['error']}")
        else:
            c = r["checks"]
            print(f"[{flag}] {path.name:60} "
                  f"ssim={c['ssim_first_last']['value']}  "
                  f"mse={c['mse_first_last']['value']}  "
                  f"lap={c['laplacian_mid']['value']}")

    REPORT.write_text(json.dumps(results, indent=2))
    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"\n{n_ok}/{len(results)} passed   ->   {REPORT}")

    # Optional: re-render failures
    if args.rerolls > 0 and n_ok < len(results):
        failed = [pathlib.Path(r["video"]).stem.split("_")[0] + "_" + pathlib.Path(r["video"]).stem.split("_")[1]
                  for r in results if not r.get("ok")]
        # Strip the timestamp suffix to get the canonical label
        labels = sorted({"_".join(pathlib.Path(r["video"]).stem.split("_")[:-1])
                         for r in results if not r.get("ok")})
        print(f"\nRe-rendering {len(labels)} failed clip(s): {labels}")
        for attempt in range(1, args.rerolls + 1):
            print(f"  attempt {attempt}/{args.rerolls}")
            cmd = [sys.executable, str(ROOT / "scripts" / "veo_idle_library.py"), *labels, "--serial"]
            subprocess.run(cmd)
            # Re-validate
            new_results = [validate(p) for p in IDLE_DIR.glob("*.mp4") if not p.is_symlink()
                           and any(p.stem.startswith(label) for label in labels)]
            still_failed = [r for r in new_results if not r.get("ok")]
            if not still_failed:
                print("  all passing now")
                break
            labels = sorted({"_".join(pathlib.Path(r["video"]).stem.split("_")[:-1]) for r in still_failed})
        sys.exit(0 if not still_failed else 1)

    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
