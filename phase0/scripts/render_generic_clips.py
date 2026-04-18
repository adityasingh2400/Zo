#!/usr/bin/env python
"""P1.3: Pre-render 12-18 generic lip-synced bridge/intro/response clips.

Goal: keep the dashboard "always alive" by playing canned LatentSync clips while
the next live render is underway. We render with the *speaking* state video as
the source so the avatar looks natural even on bridge content.

Categories (counts pick up at 18):
  intro      ×3   ("Welcome to the stream!" etc.)
  bridge     ×6   ("Let me check that real quick…", spinning while another render loads)
  question   ×3   ("Great question!")
  compliment ×3   ("Thank you so much!")
  objection  ×3   ("I hear you, let me explain…")

Each clip:
  1. ElevenLabs TTS → mp3
  2. POST to LatentSync /lipsync (steps=10, 1080p out)
  3. Save to phase0/assets/clips/<category>_<slug>.mp4
  4. Track manifest in phase0/assets/clips/manifest.json

Skips clips that already exist on disk (idempotent re-runs).

Usage:
  python phase0/scripts/render_generic_clips.py            # all
  python phase0/scripts/render_generic_clips.py intro      # one category
  python phase0/scripts/render_generic_clips.py --steps 8  # faster, lower quality
"""
from __future__ import annotations
import os, sys, time, json, pathlib, argparse, hashlib
from dotenv import dotenv_values

ROOT = pathlib.Path(__file__).resolve().parent.parent
env = dotenv_values(ROOT.parent / ".env")
for k, v in env.items():
    os.environ.setdefault(k, v)

from elevenlabs import ElevenLabs
import httpx

ELEVEN_API_KEY = os.environ["ELEVENLABS_API_KEY"]
ELEVEN_VOICE = os.environ.get("ELEVENLABS_VOICE_ID", "yj30vwTGJxSHezdAGsv9")

LATENTSYNC_URL = os.environ.get("LATENTSYNC_URL", "http://127.0.0.1:8766")
POD_SPEAKING_1080P = os.environ.get(
    "POD_SPEAKING_1080P", "/workspace/state_pitching_pose_speaking_1080p.mp4"
)
LOCAL_SOURCE = ROOT / "assets" / "states" / "state_pitching_pose_speaking_1080p.mp4"

OUT_DIR = ROOT / "assets" / "clips"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLIPS: dict[str, list[str]] = {
    "intro": [
        "Hey everyone, welcome to the stream!",
        "Hi there, so glad you're here today!",
        "Welcome in, let's check out something special.",
    ],
    "bridge": [
        "Let me grab that for you real quick.",
        "Great question, hold on one second.",
        "One moment, looking that up now.",
        "Yeah totally, give me a sec.",
        "Awesome, let me pull that up.",
        "Good catch, hang tight.",
    ],
    "question": [
        "Yes, that's a great question and the answer is yes!",
        "Absolutely, it can definitely do that.",
        "Yep, that's actually one of my favorite features.",
    ],
    "compliment": [
        "Thank you so much, that means a lot!",
        "Aw, thanks for saying that!",
        "Appreciate the love in the chat!",
    ],
    "objection": [
        "I hear you, let me explain what makes it different.",
        "Totally fair, here's the thing though.",
        "I get that concern, here's why it still works.",
    ],
}

# Per-category audio tag prefixes for Eleven v3 (--v3 mode).
# v3 reads bracketed cues like [warm] or [gentle laugh] as expressive
# direction. They DON'T appear in the spoken output — they shape delivery.
V3_TAGS: dict[str, str] = {
    "intro":      "[warm][bright]",
    "bridge":     "[curious][casual]",
    "question":   "[enthusiastic][curious]",
    "compliment": "[gentle laugh][warm]",
    "objection":  "[soft][confident]",
}


def slugify(text: str) -> str:
    base = "".join(c.lower() if c.isalnum() else "_" for c in text)
    while "__" in base:
        base = base.replace("__", "_")
    return base.strip("_")[:40]


def tts(text: str, eleven: ElevenLabs, v3: bool = False, category: str = "") -> bytes:
    """Eleven TTS. v3=True uses the expressive Eleven v3 model with audio
    tags — slower (~3-5x latency) but dramatically more human delivery,
    correct trade-off for offline pre-renders. Tag prefix per category."""
    if v3:
        prefix = V3_TAGS.get(category, "[warm]")
        prompt = f"{prefix} {text}"
        # Eleven v3 model id. The SDK accepts either eleven_v3 or the
        # 'eleven-v3-alpha' string depending on plan; we let the API
        # error speak if neither works.
        audio = eleven.text_to_speech.convert(
            voice_id=ELEVEN_VOICE,
            text=prompt,
            model_id="eleven_v3",
            output_format="mp3_44100_128",
        )
    else:
        audio = eleven.text_to_speech.convert(
            voice_id=ELEVEN_VOICE,
            text=text,
            model_id="eleven_turbo_v2_5",
            output_format="mp3_44100_128",
        )
    return b"".join(audio)


def render_one(text: str, audio_bytes: bytes, src_bytes: bytes, steps: int, out_height: int) -> tuple[bytes, dict]:
    files = {
        "source_video": ("src.mp4", src_bytes, "video/mp4"),
        "audio": ("a.mp3", audio_bytes, "audio/mpeg"),
    }
    data = {
        "inference_steps": str(steps),
        "guidance_scale": "1.5",
        "enable_deepcache": "1",
        "out_height": str(out_height),
    }
    with httpx.Client(timeout=900.0) as c:
        r = c.post(f"{LATENTSYNC_URL}/lipsync", data=data, files=files)
        r.raise_for_status()
        return r.content, dict(r.headers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("category", nargs="*", help="Categories to render (default: all)")
    ap.add_argument("--steps", type=int, default=10, help="LatentSync inference steps")
    ap.add_argument("--height", type=int, default=1080, help="Output height in px")
    ap.add_argument("--force", action="store_true", help="Re-render even if file exists")
    ap.add_argument("--v3", action="store_true",
                    help="Use Eleven v3 with audio tags ([warm][curious][gentle laugh] etc.) "
                         "for dramatically more expressive delivery. Slower, fine for offline.")
    args = ap.parse_args()

    categories = args.category or list(CLIPS.keys())
    bad = [c for c in categories if c not in CLIPS]
    if bad:
        print(f"unknown categories: {bad}; available={list(CLIPS.keys())}")
        sys.exit(2)

    if not LOCAL_SOURCE.exists():
        print(f"source video missing: {LOCAL_SOURCE}")
        sys.exit(2)
    src_bytes = LOCAL_SOURCE.read_bytes()
    print(f"source: {LOCAL_SOURCE.name} ({len(src_bytes)//1024} KB)")

    eleven = ElevenLabs(api_key=ELEVEN_API_KEY)

    manifest_path = OUT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    plan = [(cat, t) for cat in categories for t in CLIPS[cat]]
    print(f"rendering {len(plan)} clips at {args.height}p, steps={args.steps}")

    for i, (cat, text) in enumerate(plan, 1):
        slug = slugify(text)
        out = OUT_DIR / f"{cat}_{slug}.mp4"
        if out.exists() and not args.force:
            print(f"[{i:2d}/{len(plan)}] SKIP existing {out.name}")
            manifest.setdefault(cat, [])
            if not any(c["file"] == out.name for c in manifest[cat]):
                manifest[cat].append({"text": text, "file": out.name})
            continue

        print(f"[{i:2d}/{len(plan)}] {cat}: {text!r}")
        t0 = time.perf_counter()
        try:
            audio = tts(text, eleven, v3=args.v3, category=cat)
        except Exception as e:
            print(f"   TTS failed: {e}")
            continue
        tts_ms = int((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        try:
            video, _headers = render_one(text, audio, src_bytes, args.steps, args.height)
        except Exception as e:
            print(f"   LatentSync failed: {e}")
            continue
        ls_ms = int((time.perf_counter() - t1) * 1000)

        out.write_bytes(video)
        size_kb = out.stat().st_size // 1024
        print(f"   wrote {out.name} ({size_kb} KB)  tts={tts_ms}ms ls={ls_ms}ms total={int((time.perf_counter()-t0)*1000)}ms")

        manifest.setdefault(cat, [])
        if not any(c["file"] == out.name for c in manifest[cat]):
            manifest[cat].append({"text": text, "file": out.name})
        # Persist after every render so a crash mid-batch keeps progress.
        manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nManifest: {manifest_path}")
    for cat, items in manifest.items():
        print(f"  {cat}: {len(items)} clips")


if __name__ == "__main__":
    main()
