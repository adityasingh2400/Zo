#!/usr/bin/env python
"""Generate the 4 remaining state videos for Phase 1.

States we already have: pitching_pose_silent, pitching_pose_speaking (1080p).
States we still need:
  - idle:       neutral standing, gentle breathing, occasional blinks (silent)
  - excited:    enthusiastic gesture, hands up, big smile (speaking)
  - explaining: slight tilt, hands gesturing as if describing details (speaking)
  - reaching:   reaches forward toward camera, like presenting an item (speaking)

All use the same canonical portrait so the lipsync source has a consistent face.
1080p, 9:16, 8s each. Submit-and-poll concurrently to save wall time.

Usage:
  python phase0/scripts/veo_states_batch.py            # all 4
  python phase0/scripts/veo_states_batch.py idle       # one only
  python phase0/scripts/veo_states_batch.py idle excited
"""
from __future__ import annotations
import os, sys, time, pathlib, json, concurrent.futures
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

# Each prompt holds the woman + cozy bedroom set fixed and varies only the body
# language and mouth state. "silent" prompts explicitly close the lips so the
# Veo motion field doesn't fight with our lip-sync overlay.
STATES: dict[str, dict] = {
    "idle": {
        "speaking": False,
        "prompt": (
            "The same woman stands centered, looking warmly toward camera with a calm, "
            "neutral expression. Lips closed in a soft, gentle smile (mouth completely "
            "shut, no speech motion). Subtle breathing motion in shoulders. Soft natural "
            "blinks every 2-3 seconds. Hands rest comfortably at sides. The shot is "
            "locked off (tripod), cozy bedroom background stays identical. This is a "
            "silent idle clip used between live spoken segments."
        ),
    },
    "excited": {
        "speaking": True,
        "prompt": (
            "The same woman speaks energetically to camera with infectious enthusiasm, as "
            "if announcing exciting news. Both hands rise up to shoulder height in open, "
            "celebratory gestures. Big genuine smile. Animated micro-expressions, slight "
            "head bob with rhythm. Lips move naturally — normal conversational phonemes, "
            "not exaggerated. Soft natural blinks. The shot is locked off (tripod), cozy "
            "bedroom background stays identical. Audio will be replaced via lip-sync."
        ),
    },
    "explaining": {
        "speaking": True,
        "prompt": (
            "The same woman speaks thoughtfully to camera as if explaining details about a "
            "product. Slight head tilt. Hands move at chest height in open, descriptive "
            "gestures — counting points on fingers, indicating sizes, drawing shapes in "
            "the air. Warm engaged smile. Natural micro-expressions, soft blinks. Lips "
            "move naturally with conversational phonemes. The shot is locked off (tripod), "
            "cozy bedroom background stays identical. Audio will be replaced via lip-sync."
        ),
    },
    "reaching": {
        "speaking": True,
        "prompt": (
            "The same woman speaks warmly to camera and gently reaches one hand forward "
            "toward the lens, as if presenting a small object directly to the viewer. The "
            "other hand stays at chest height. Inviting smile, eye contact with camera. "
            "Lips move naturally with conversational phonemes. Soft natural blinks. The "
            "shot is locked off (tripod), cozy bedroom background stays identical. Audio "
            "will be replaced via lip-sync."
        ),
    },
}


def gen_one(label: str, spec: dict) -> dict:
    client = genai.Client(api_key=API_KEY)
    image_bytes = PORTRAIT.read_bytes()
    full_label = f"{label}_pose_{'speaking' if spec['speaking'] else 'silent'}_1080p"
    print(f"[{full_label}] submitting Veo 3.1 (1080p, 9:16, 8s)")
    t0 = time.perf_counter()

    op = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt=spec["prompt"],
        image=types.Image(image_bytes=image_bytes, mime_type="image/png"),
        config=types.GenerateVideosConfig(
            aspect_ratio="9:16",
            duration_seconds=8,
            resolution="1080p",
            person_generation="allow_adult",
        ),
    )
    print(f"[{full_label}] submitted ({time.perf_counter()-t0:.1f}s)")

    poll = 0
    while not op.done:
        poll += 1
        time.sleep(8)
        op = client.operations.get(op)
        elapsed = time.perf_counter() - t0
        if poll % 3 == 0:
            print(f"[{full_label}] poll {poll} elapsed={elapsed:.0f}s done={op.done}")
        if elapsed > 600:
            raise TimeoutError(f"{full_label} timed out after 10 min")

    if op.error:
        raise RuntimeError(f"{full_label} Veo error: {op.error}")

    resp = op.response
    videos = getattr(resp, "generated_videos", None) or getattr(resp, "videos", None)
    if not videos and hasattr(resp, "model_dump"):
        raw = resp.model_dump()
        videos = raw.get("generated_videos") or raw.get("videos")
    if not videos:
        raise RuntimeError(f"{full_label} no videos in response: {resp}")

    v = videos[0]
    video_file = v.video if hasattr(v, "video") else (v.get("video") if isinstance(v, dict) else None)

    ts = int(time.time())
    out = OUT_DIR / f"state_{full_label}_{ts}.mp4"
    if video_file and hasattr(video_file, "save"):
        try:
            client.files.download(file=video_file)
        except Exception as e:
            print(f"  [{full_label}] download() warning: {e}")
        video_file.save(str(out))
    elif isinstance(video_file, dict) and "uri" in video_file:
        import httpx
        r = httpx.get(video_file["uri"], headers={"x-goog-api-key": API_KEY},
                      timeout=120, follow_redirects=True)
        r.raise_for_status()
        out.write_bytes(r.content)
    else:
        raise RuntimeError(f"{full_label} unknown video payload: {type(video_file)}")

    total = time.perf_counter() - t0
    size_kb = out.stat().st_size // 1024
    print(f"[{full_label}] DONE {total:.1f}s -> {out.name} ({size_kb} KB)")

    dst = OUT_DIR / f"state_{full_label}.mp4"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(out.name)
    return {
        "label": full_label, "path": str(out), "wall_sec": round(total, 1),
        "size_kb": size_kb, "stable": str(dst),
    }


def main():
    args = sys.argv[1:]
    targets = args or list(STATES.keys())
    unknown = [t for t in targets if t not in STATES]
    if unknown:
        print(f"unknown states: {unknown}; available={list(STATES.keys())}")
        sys.exit(2)

    print(f"Generating {len(targets)} state(s) in parallel: {targets}")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(targets)) as ex:
        futs = {ex.submit(gen_one, label, STATES[label]): label for label in targets}
        for fut in concurrent.futures.as_completed(futs):
            label = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"[{label}] FAILED: {e}")
                results.append({"label": label, "error": str(e)})

    report = ROOT / "bench" / "results" / "veo_states_batch.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, indent=2))
    print(f"Report: {report}")
    for r in results:
        print(" ", r)


if __name__ == "__main__":
    main()
