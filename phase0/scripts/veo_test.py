#!/usr/bin/env python
"""Phase 0.2c — Veo 3.1 image-to-video test: pitching_pose silent state.

Uses the character portrait PNG as the starting frame via Google GenAI SDK.
Generates one 8-second silent state clip with body motion but a neutral, relaxed
mouth (explicitly prompted). Measures wall-clock render time.

Veo is a long-running operation: we submit, then poll until done, then download.
"""
from __future__ import annotations
import os, sys, time, pathlib
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

# Short-form prompt engineered for "silent body language, mouth NEUTRAL"
# Phrased positively without triggering audio/speech safety filters on Veo.
PROMPT_SILENT_PITCH = (
    "The same woman stands relaxed in frame, gently breathing, with a soft closed-lip "
    "smile that holds the entire shot. She thoughtfully tilts her head, blinks softly, "
    "and slowly raises her hands to chest height in an open, welcoming gesture of "
    "presenting something. Her lips stay softly together in a calm, relaxed, thoughtful "
    "expression throughout — the vibe is contemplative listening, not conversation. "
    "Shot is locked off (tripod), cozy bedroom background holds steady. Ambient room "
    "tone only — no dialogue, no music, no foley."
)

# A/B candidate: gently-speaking-no-audio. Some lip-sync models work better when the source
# has mouth motion as a canvas to re-animate.
PROMPT_SPEAKING_PITCH = (
    "The same woman stays centered in frame and speaks warmly to camera as if greeting "
    "viewers on a livestream. Hands rise to chest height in open, welcoming gestures as "
    "she talks. Natural micro-expressions, soft blinks every 2-3 seconds. Warm smile. "
    "Lips move naturally, mouth shapes form normal conversational phonemes — not "
    "exaggerated, just genuine speech. The shot is locked off (tripod), cozy bedroom "
    "background stays identical. This footage will have its audio replaced via lip-sync."
)


def run_one(label: str, prompt: str, aspect_ratio: str = "9:16", duration_sec: int = 8):
    client = genai.Client(api_key=API_KEY)
    image_bytes = PORTRAIT.read_bytes()
    mime = "image/png"

    print(f"[{label}] submitting Veo 3.1 image-to-video  ({duration_sec}s, {aspect_ratio})")
    t_submit = time.perf_counter()

    # Veo 3.1 via Gemini API — long running operation.
    # Note: generate_audio unsupported on Gemini API. Veo will produce audio but we discard it
    # and overlay our TTS via lip sync. The SILENT prompt tells Veo to keep mouth neutral.
    op = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt=prompt,
        image=types.Image(image_bytes=image_bytes, mime_type=mime),
        config=types.GenerateVideosConfig(
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_sec,
            resolution="720p",
            negative_prompt=(
                "mouth open, talking, speaking, teeth visible, wide smile, "
                "open-mouth expression, dialogue, lip movement for speech"
            ),
            person_generation="allow_adult",
        ),
    )
    print(f"[{label}] submitted in {time.perf_counter()-t_submit:.2f}s, operation name: {op.name}")

    # Poll
    t_poll_start = time.perf_counter()
    poll = 0
    while not op.done:
        poll += 1
        time.sleep(8)
        op = client.operations.get(op)
        elapsed = time.perf_counter() - t_poll_start
        print(f"[{label}] poll {poll}  elapsed={elapsed:.1f}s  done={op.done}")
        if elapsed > 600:  # 10 min hard cap
            print(f"[{label}] TIMED OUT after {elapsed:.0f}s")
            return None

    if op.error:
        print(f"[{label}] ERROR: {op.error}")
        return None

    total = time.perf_counter() - t_submit
    print(f"[{label}] operation done — response keys:")
    resp = op.response
    # SDK response can be either .generated_videos or .generateVideoResponse.videos or similar.
    videos = None
    for attr in ("generated_videos", "videos"):
        videos = getattr(resp, attr, None)
        if videos:
            break
    # Dict fallback
    if not videos and hasattr(resp, "model_dump"):
        raw = resp.model_dump()
        print(f"   raw response keys: {list(raw.keys())[:10]}")
        videos = raw.get("generated_videos") or raw.get("videos") or raw.get("generateVideoResponse", {}).get("generatedSamples")
    if not videos:
        print(f"   raw dict: {resp}")
        raise RuntimeError("No video in response (see logs)")

    v = videos[0]
    # v can be either .video (a File) or a dict with {"video": {"uri": ...}}
    ts = int(time.time())
    out = OUT_DIR / f"state_{label}_{ts}.mp4"

    video_file = None
    if hasattr(v, "video"):
        video_file = v.video
    elif isinstance(v, dict):
        video_file = v.get("video")

    # Download: try file.save, or download URL
    if video_file and hasattr(video_file, "save"):
        # It's a genai File object — need to download first
        try:
            client.files.download(file=video_file)
        except Exception as e:
            print(f"   download() warning: {e}")
        video_file.save(str(out))
    elif isinstance(video_file, dict) and "uri" in video_file:
        import httpx
        uri = video_file["uri"]
        headers = {"x-goog-api-key": API_KEY}
        r = httpx.get(uri, headers=headers, timeout=120, follow_redirects=True)
        r.raise_for_status()
        out.write_bytes(r.content)
    else:
        raise RuntimeError(f"Unknown video payload: {type(video_file)} {video_file}")

    size = out.stat().st_size
    print(f"[{label}] DONE in {total:.1f}s  -> {out}  ({size/1024:.0f} KB)")
    return {"label": label, "path": str(out), "wall_clock_sec": round(total, 1), "size_kb": size // 1024}


def main():
    import json, concurrent.futures, sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    specs = []
    if mode in ("both", "silent"):
        specs.append(("pitching_pose_silent", PROMPT_SILENT_PITCH))
    if mode in ("both", "speaking"):
        specs.append(("pitching_pose_speaking", PROMPT_SPEAKING_PITCH))

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(specs)) as ex:
        futs = {ex.submit(run_one, label, prompt, "9:16", 8): label for label, prompt in specs}
        for f in concurrent.futures.as_completed(futs):
            try:
                r = f.result()
                if r:
                    results.append(r)
            except Exception as e:
                print(f"task failed: {e}")

    report = ROOT / "bench" / "results" / "veo_test.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, indent=2))
    print(f"\nReport: {report}")
    # Also emit stable-name symlinks for lipsync_bench to consume
    for r in results:
        label = r["label"]
        src = pathlib.Path(r["path"])
        dst = OUT_DIR / f"state_{label}.mp4"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(src.name)
        except OSError:
            dst.write_bytes(src.read_bytes())
        print(f"  symlink: {dst.name} -> {src.name}")

if __name__ == "__main__":
    main()
