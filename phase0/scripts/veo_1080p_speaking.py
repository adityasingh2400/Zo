#!/usr/bin/env python
"""Re-generate the SPEAKING pitching-pose state video at 1080p.

P0.3b established speaking-source > silent-source for lip sync because the
re-animated mouth has motion to work with. For the 3-way benchmark (LatentSync
vs Sonic vs Wav2Lip) we want the highest-quality source we can get so output
resolution is capped by the input, not the model.
"""
from __future__ import annotations
import os, sys, time, pathlib, json
from dotenv import dotenv_values

ROOT = pathlib.Path(__file__).resolve().parent.parent
env = dotenv_values(ROOT.parent / ".env")
for k, v in env.items():
    os.environ[k] = v

from google import genai
from google.genai import types

API_KEY = os.environ["GEMINI_API_KEY"]
PORTRAIT = ROOT / "assets" / "portraits" / "portrait.png"
OUT_DIR = ROOT / "assets" / "states"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = (
    "The same woman stays centered in frame and speaks warmly to camera as if greeting "
    "viewers on a livestream. Hands rise to chest height in open, welcoming gestures as "
    "she talks. Natural micro-expressions, soft blinks every 2-3 seconds. Warm smile. "
    "Lips move naturally, mouth shapes form normal conversational phonemes — not "
    "exaggerated, just genuine speech. The shot is locked off (tripod), cozy bedroom "
    "background stays identical. This footage will have its audio replaced via lip-sync."
)

def main():
    client = genai.Client(api_key=API_KEY)
    image_bytes = PORTRAIT.read_bytes()
    label = "pitching_pose_speaking_1080p"

    print(f"[{label}] submitting Veo 3.1 (1080p, 9:16, 8s)")
    t0 = time.perf_counter()

    op = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt=PROMPT,
        image=types.Image(image_bytes=image_bytes, mime_type="image/png"),
        config=types.GenerateVideosConfig(
            aspect_ratio="9:16",
            duration_seconds=8,
            resolution="1080p",
            person_generation="allow_adult",
        ),
    )
    print(f"[{label}] submitted ({time.perf_counter()-t0:.1f}s), op={op.name}")

    # Poll
    poll = 0
    while not op.done:
        poll += 1
        time.sleep(8)
        op = client.operations.get(op)
        elapsed = time.perf_counter() - t0
        print(f"[{label}] poll {poll} elapsed={elapsed:.0f}s done={op.done}")
        if elapsed > 600:
            raise TimeoutError("Veo 1080p timed out after 10 min")

    if op.error:
        raise RuntimeError(f"Veo error: {op.error}")

    resp = op.response
    videos = getattr(resp, "generated_videos", None) or getattr(resp, "videos", None)
    if not videos and hasattr(resp, "model_dump"):
        raw = resp.model_dump()
        videos = raw.get("generated_videos") or raw.get("videos")
    if not videos:
        raise RuntimeError(f"No videos in response: {resp}")

    v = videos[0]
    video_file = v.video if hasattr(v, "video") else (v.get("video") if isinstance(v, dict) else None)

    ts = int(time.time())
    out = OUT_DIR / f"state_{label}_{ts}.mp4"

    if video_file and hasattr(video_file, "save"):
        try:
            client.files.download(file=video_file)
        except Exception as e:
            print(f"   download() warning: {e}")
        video_file.save(str(out))
    elif isinstance(video_file, dict) and "uri" in video_file:
        import httpx
        r = httpx.get(video_file["uri"], headers={"x-goog-api-key": API_KEY}, timeout=120, follow_redirects=True)
        r.raise_for_status()
        out.write_bytes(r.content)
    else:
        raise RuntimeError(f"Unknown video payload: {type(video_file)} {video_file}")

    total = time.perf_counter() - t0
    size_kb = out.stat().st_size // 1024
    print(f"[{label}] DONE {total:.1f}s -> {out} ({size_kb} KB)")

    # Stable symlink
    dst = OUT_DIR / f"state_{label}.mp4"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(out.name)
    print(f"  symlink: {dst.name} -> {out.name}")

    # Probe resolution via ffprobe (if available)
    import subprocess, shutil
    if shutil.which("ffprobe"):
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,duration,r_frame_rate",
             "-of", "json", str(out)],
            capture_output=True, text=True,
        )
        print("  probe:", probe.stdout.strip())

    report = ROOT / "bench" / "results" / "veo_1080p.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({
        "label": label, "path": str(out), "wall_clock_sec": round(total, 1),
        "size_kb": size_kb, "stable_symlink": str(dst),
    }, indent=2))
    print(f"Report: {report}")

if __name__ == "__main__":
    main()
