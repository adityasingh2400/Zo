#!/usr/bin/env python
"""Smoke test for the Wav2Lip RunPod server via local SSH tunnel (localhost:8010).

Usage:
  # 1. Start tunnel in another terminal:
  #    ssh -N -L 8010:localhost:8010 -p <SSH_PORT> -i ~/.ssh/id_ed25519 root@<POD_IP>
  # 2. Run this:
  python phase0/scripts/smoketest_wav2lip.py
"""
from __future__ import annotations
import os, sys, time, pathlib, httpx

ROOT = pathlib.Path(__file__).resolve().parent.parent
SERVER = os.environ.get("LIPSYNC_URL", "http://localhost:8010")

SILENT = ROOT / "assets" / "states" / "state_pitching_pose_silent.mp4"
SPEAKING = ROOT / "assets" / "states" / "state_pitching_pose_speaking.mp4"
AUDIO = ROOT / "bench" / "audio" / "pitch_10s.mp3"

OUT_DIR = ROOT / "bench" / "videos"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def health():
    r = httpx.get(f"{SERVER}/health", timeout=10)
    r.raise_for_status()
    print("health:", r.json())


def lipsync(video_path: pathlib.Path, audio_path: pathlib.Path, label: str):
    if not video_path.exists():
        print(f"[{label}] skip — {video_path.name} missing"); return
    print(f"[{label}] POST  video={video_path.name}  audio={audio_path.name}")
    t0 = time.perf_counter()
    with video_path.open("rb") as vf, audio_path.open("rb") as af:
        files = {
            "video": (video_path.name, vf, "video/mp4"),
            "audio": (audio_path.name, af, "audio/mpeg"),
        }
        r = httpx.post(f"{SERVER}/lipsync", files=files, timeout=300)
    total = time.perf_counter() - t0
    if r.status_code != 200:
        print(f"[{label}] FAIL {r.status_code}: {r.text[:500]}")
        return
    render_sec = r.headers.get("X-Render-Seconds", "?")
    out = OUT_DIR / f"smoke_{label}.mp4"
    out.write_bytes(r.content)
    print(f"[{label}] OK  total={total:.1f}s  render={render_sec}s  size={len(r.content)//1024}KB  -> {out.name}")


if __name__ == "__main__":
    print(f"Server: {SERVER}")
    health()
    print()
    lipsync(SILENT, AUDIO, "silent_src")
    lipsync(SPEAKING, AUDIO, "speaking_src")
