#!/usr/bin/env python
"""Boomerang an idle clip: append a reversed copy so the clip loops perfectly.

Why: Veo can't reliably guarantee that frame N == frame 0 even with explicit
prompting. For symmetric "depart from anchor and return" clips (calm idle
breathing, lean-in, head tilt) playing the clip forward then backward yields
a 16s file where the last frame is byte-for-byte identical to the first
frame, so HTML <video loop> just works.

This is the wrong tool for asymmetric clips like sip-drink or walk-off-return
(reversing them looks like throwing up / walking backward). Those should be
played once as Tier 1 interjections, not boomeranged.

The script:
  1. Reads the source clip.
  2. Builds a forward+reverse concat with ffmpeg.
  3. Writes <stem>_boom.mp4 alongside the source.
  4. Updates the canonical symlink (e.g. idle_calm.mp4) to point at the
     boomerang version so the Director URL doesn't have to change.

Usage:
  python phase0/scripts/boomerang_idle.py                  # all loopable clips
  python phase0/scripts/boomerang_idle.py idle_calm idle_attentive
  python phase0/scripts/boomerang_idle.py --restore-original  # undo, point symlinks back at originals
"""
from __future__ import annotations
import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
IDLE_DIR = ROOT / "assets" / "states" / "idle"

# Clips that are appropriate for boomerang (symmetric depart/return motion).
# Anything not in this list either doesn't need looping or doesn't survive
# being played in reverse.
LOOPABLE = [
    "idle_calm",
    "idle_attentive",
    "idle_thinking",
    "misc_glance_aside",
    "misc_hair_touch",
]


def latest_concrete_for(label: str) -> pathlib.Path | None:
    """Find the most recent concrete (non-symlink) mp4 starting with this label."""
    matches = sorted(
        (p for p in IDLE_DIR.glob(f"{label}_*.mp4") if not p.is_symlink()),
        key=lambda p: p.stat().st_mtime,
    )
    return matches[-1] if matches else None


def boomerang(src: pathlib.Path, out: pathlib.Path) -> bool:
    """ffmpeg: forward then reverse, single mp4. Strips audio (idle is muted)."""
    if out.exists():
        out.unlink()
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-filter_complex",
        "[0:v]split=2[a][b];[b]reverse,setpts=PTS-STARTPTS[r];[a][r]concat=n=2:v=1[out]",
        "-map", "[out]",
        "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-loglevel", "error",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ffmpeg failed: {r.stderr.strip()}")
        return False
    return out.exists() and out.stat().st_size > 0


def repoint_symlink(canonical: pathlib.Path, target_name: str) -> None:
    """Make canonical.mp4 point at target_name. canonical lives in IDLE_DIR."""
    if canonical.exists() or canonical.is_symlink():
        canonical.unlink()
    canonical.symlink_to(target_name)


def process_label(label: str) -> dict:
    src = latest_concrete_for(label)
    if not src:
        return {"label": label, "status": "no_source"}

    boom_path = src.with_name(f"{src.stem}_boom.mp4")
    print(f"[{label}]  src={src.name}")
    if not boomerang(src, boom_path):
        return {"label": label, "status": "ffmpeg_failed", "src": src.name}

    canonical = IDLE_DIR / f"{label}.mp4"
    repoint_symlink(canonical, boom_path.name)

    size_kb = boom_path.stat().st_size // 1024
    print(f"          boom={boom_path.name}  ({size_kb} KB)  symlink: {canonical.name} -> {boom_path.name}")
    return {
        "label": label,
        "status": "ok",
        "src": src.name,
        "boom": boom_path.name,
        "size_kb": size_kb,
    }


def restore_originals(labels: list[str]) -> None:
    """Point the canonical symlink back at the original Veo render."""
    for label in labels:
        src = latest_concrete_for(label)
        if not src:
            print(f"[{label}]  no original found, skipping")
            continue
        canonical = IDLE_DIR / f"{label}.mp4"
        repoint_symlink(canonical, src.name)
        print(f"[{label}]  symlink restored: {canonical.name} -> {src.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="*", help="labels to boomerang (default: all loopable)")
    ap.add_argument("--restore-original", action="store_true",
                    help="point canonical symlinks back at the non-boomerang originals")
    args = ap.parse_args()

    targets = args.labels or LOOPABLE
    bad = [t for t in targets if t not in LOOPABLE]
    if bad:
        print(f"warning: {bad} not in LOOPABLE list — boomerang may look weird "
              f"on these (asymmetric motion). Proceeding anyway.")

    if args.restore_original:
        restore_originals(targets)
        return

    print(f"Boomeranging {len(targets)} clip(s) into {IDLE_DIR}\n")
    results = [process_label(label) for label in targets]
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{ok}/{len(results)} succeeded")
    sys.exit(0 if ok == len(results) else 1)


if __name__ == "__main__":
    main()
