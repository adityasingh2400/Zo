#!/usr/bin/env python
"""Generate the seamless idle + miscellaneous library for the Tier 0 / Tier 1
avatar layers on the EMPIRE dashboard.

Why this script exists separately from veo_states_batch.py:
  * Every clip here MUST loop seamlessly: first frame ~ last frame.
  * Every clip uses a unified cinematography + FACS micro-expression scaffold
    so the seven variants share lighting, lens, identity, and overall texture.
  * Validation step (run separately via validate_loop_seamless.py) auto re-rolls
    any clip whose endpoint doesn't match.

Usage:
  python phase0/scripts/veo_idle_library.py                      # all 7
  python phase0/scripts/veo_idle_library.py idle_calm idle_attentive
  python phase0/scripts/veo_idle_library.py --serial            # one at a time
  python phase0/scripts/veo_idle_library.py --concurrency 3     # parallel cap
"""
from __future__ import annotations
import os, sys, time, pathlib, json, concurrent.futures, argparse
from dotenv import dotenv_values

ROOT = pathlib.Path(__file__).resolve().parent.parent
env = dotenv_values(ROOT.parent / ".env")
for k, v in env.items():
    os.environ[k] = v

from google import genai
from google.genai import types

API_KEY = os.environ["GEMINI_API_KEY"]
PORTRAIT = ROOT / "assets" / "portraits" / "portrait.png"
OUT_DIR = ROOT / "assets" / "states" / "idle"  # rendered into states/idle so they ship next to the speaking-pose source
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared scaffold ─────────────────────────────────────────────────────────
# Veo was trained on professional film, so use cinematography vocabulary, name
# the lighting key (Rembrandt), the lens, the depth of field, and describe
# performance in FACS-like anatomical terms instead of generic adjectives.
SCAFFOLD = (
    "Same woman as the reference portrait: late twenties, warm brown hair to "
    "mid-back, light freckles across the nose bridge, soft natural skin texture "
    "with visible pores under fill light. Wearing a cream knit sweater. Cozy "
    "bedroom set: warm fairy-light bokeh in the background out of focus. "
    "Locked-off medium close-up on a tripod, 50mm equivalent lens, shallow "
    "depth of field (f/2.0), eye level. Soft Rembrandt key light from "
    "camera-left, gentle warm fill from camera-right, subtle backlight rim on "
    "the hair. Photorealistic, 24 fps, 1080p, 9:16. Skin must look like skin — "
    "visible micro-detail, natural blemishes, no plastic smoothing. "
    "FRAMING LOCK: every single frame of the output (including frame 0, "
    "frame 1, the midpoint, and the final frame) must use the EXACT same "
    "crop, zoom level, and head size in frame as the conditioning reference "
    "image. The subject's head must occupy the same percentage of the frame "
    "from the very first frame to the very last. There is no zoom-in, no "
    "zoom-out, no dolly, no push-in, no pull-back. The camera is rigidly "
    "locked off on a tripod. {PERFORMANCE} "
    "LOOP CONSTRAINT: the pose, expression, head angle, hand position, gaze "
    "direction, framing, and shoulder line at second 0 must be visually "
    "identical to second 8 so the clip can loop seamlessly with no visible "
    "cut. No drift, no progressive movement. The clip will be muted in "
    "playback; do not generate any voice."
)

# ── Variant performance clauses ────────────────────────────────────────────
PERFORMANCES: dict[str, str] = {
    "idle_calm": (
        "ABSOLUTELY NO CAMERA MOTION OF ANY KIND. This is a fixed "
        "security-camera-style observation shot, NOT a cinematic portrait. "
        "There is no push-in, no dolly, no zoom, no parallax, no breathing "
        "camera. The framing at frame 1, frame 24, frame 100, and frame 192 "
        "must be byte-for-byte identical in crop and head size. The subject "
        "looks softly toward the lens with a gentle, closed-mouth smile. "
        "Two slow natural blinks across the eight seconds. The shoulders "
        "rise and fall with the smallest possible breath. Apart from those "
        "two blinks and the breath, NOTHING moves: the head does not tilt, "
        "rotate, or nod; the eyes do not drift; the hands do not appear or "
        "shift; the lips do not part. Treat this like a long-exposure "
        "still photograph that happens to include two blinks. The end frame "
        "must match the start frame so the clip can repeat invisibly."
    ),
    "idle_attentive": (
        "She is reacting to a viewer's chat message: a clear, definite "
        "lean toward the camera between second zero and second two — her "
        "head and shoulders move forward by about 12-15cm so the lean is "
        "obviously visible (not subtle). At the same time her eyes shift "
        "down and to HER lower-RIGHT (the viewer's lower-LEFT) and "
        "settle there. She HOLDS this leaned-in, looking-down-right pose "
        "perfectly still from second two through second six — body locked, "
        "head locked, only the eyes very slightly track within the lower-"
        "right region. A clear interested squint (visible lid-tighten, "
        "narrowing of both eyes), a small inner-brow furrow, and a "
        "knowing dimpled half-smile at one corner of the mouth. Her "
        "mouth stays closed — no speech, no jaw motion, no parted lips. "
        "Her hands NEVER appear in frame and never move. NO text, no "
        "chat panel, no UI overlay, no paper, no object, no graphics — "
        "the background stays the same cozy bedroom set with fairy-light "
        "bokeh, completely unchanged. The lean and squint should be "
        "clearly distinguishable from a still portrait — a casual viewer "
        "must instantly read this as 'she just leaned in to read "
        "something.' Camera stays rigidly locked off — the lean is from "
        "the subject moving toward the lens, not the lens moving toward "
        "her. (The clip will be played forward then in reverse to create "
        "a seamless loop, so it does not need to return to the anchor "
        "pose at second eight; just hold the lean cleanly through the end.)"
    ),
    "idle_thinking": (
        "A quiet thinking beat: head tilts gently 8 degrees to her right by "
        "second two, brow lowers slightly (concentration), one hand drifts up "
        "to her chin and rests there for two beats with a faint dimpled "
        "smirk. By second seven the hand returns and the head re-aligns to "
        "anchor."
    ),
    "misc_glance_aside": (
        "She glances briefly off-camera to her left at second three with a "
        "quick outer-brow raise (acknowledging someone off-frame), holds for "
        "half a second, then returns her eyes to lens with a warm Duchenne "
        "smile. Anchor pose restored by second seven."
    ),
    "misc_hair_touch": (
        "She tucks a strand of hair behind her right ear with her right hand, "
        "an unselfconscious human gesture, soft warmth in the smile "
        "(cheek-raise + lip-corner pull). Hands return to rest by second "
        "seven."
    ),
    "misc_sip_drink": (
        "She reaches off-frame to her right at second one, brings a small "
        "ceramic mug to her lips at second three, takes a small sip — chin "
        "raises slightly as she swallows — and the mug exits frame by second "
        "six. By second seven hands and posture match the anchor."
    ),
    "misc_walk_off_return": (
        "She rises calmly out of frame to her left at second one. The empty "
        "bedroom set holds with the fairy lights gently moving in the "
        "background bokeh. She returns into frame from the left at second six "
        "holding a small wrapped item, settles back to the anchor pose by "
        "second eight."
    ),
}


def gen_one(label: str) -> dict:
    if label not in PERFORMANCES:
        return {"label": label, "error": f"unknown performance label"}
    client = genai.Client(api_key=API_KEY)
    image_bytes = PORTRAIT.read_bytes()
    prompt = SCAFFOLD.format(PERFORMANCE=PERFORMANCES[label])
    print(f"[{label}] submitting Veo 3.1 (1080p, 9:16, 8s)")
    t0 = time.perf_counter()

    try:
        op = client.models.generate_videos(
            model="veo-3.1-generate-preview",
            prompt=prompt,
            image=types.Image(image_bytes=image_bytes, mime_type="image/png"),
            config=types.GenerateVideosConfig(
                aspect_ratio="9:16",
                duration_seconds=8,
                resolution="1080p",
                person_generation="allow_adult",
            ),
        )
    except Exception as e:
        return {"label": label, "error": f"submit failed: {e}"}
    print(f"[{label}] submitted ({time.perf_counter()-t0:.1f}s)")

    poll = 0
    while not op.done:
        poll += 1
        time.sleep(8)
        try:
            op = client.operations.get(op)
        except Exception as e:
            print(f"[{label}] poll error: {e}")
            time.sleep(8)
            continue
        elapsed = time.perf_counter() - t0
        if poll % 3 == 0:
            print(f"[{label}] poll {poll} elapsed={elapsed:.0f}s done={op.done}")
        if elapsed > 600:
            return {"label": label, "error": "timeout after 10 min"}

    if op.error:
        return {"label": label, "error": f"Veo error: {op.error}"}

    resp = op.response
    videos = getattr(resp, "generated_videos", None) or getattr(resp, "videos", None)
    if not videos and hasattr(resp, "model_dump"):
        raw = resp.model_dump()
        videos = raw.get("generated_videos") or raw.get("videos")
    if not videos:
        return {"label": label, "error": f"no videos in response: {resp}"}

    v = videos[0]
    video_file = v.video if hasattr(v, "video") else (v.get("video") if isinstance(v, dict) else None)

    ts = int(time.time())
    out = OUT_DIR / f"{label}_{ts}.mp4"
    if video_file and hasattr(video_file, "save"):
        try:
            client.files.download(file=video_file)
        except Exception as e:
            print(f"  [{label}] download() warning: {e}")
        video_file.save(str(out))
    elif isinstance(video_file, dict) and "uri" in video_file:
        import httpx
        r = httpx.get(video_file["uri"], headers={"x-goog-api-key": API_KEY},
                      timeout=120, follow_redirects=True)
        r.raise_for_status()
        out.write_bytes(r.content)
    else:
        return {"label": label, "error": f"unknown video payload: {type(video_file)}"}

    total = time.perf_counter() - t0
    size_kb = out.stat().st_size // 1024
    print(f"[{label}] DONE {total:.1f}s -> {out.name} ({size_kb} KB)")

    # Stable symlink so the dashboard / Director can reference a constant URL.
    dst = OUT_DIR / f"{label}.mp4"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(out.name)
    return {
        "label": label, "path": str(out), "wall_sec": round(total, 1),
        "size_kb": size_kb, "stable": str(dst), "prompt": prompt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="*", help="subset of labels to render (default: all)")
    ap.add_argument("--serial", action="store_true",
                    help="render one at a time (slower but avoids quota bursts)")
    ap.add_argument("--concurrency", type=int, default=3,
                    help="max parallel Veo submissions when not serial (default 3)")
    args = ap.parse_args()

    targets = args.labels or list(PERFORMANCES.keys())
    bad = [t for t in targets if t not in PERFORMANCES]
    if bad:
        print(f"unknown labels: {bad}; available={list(PERFORMANCES.keys())}")
        sys.exit(2)

    print(f"Generating {len(targets)} idle/misc clips: {targets}")
    print(f"  out_dir: {OUT_DIR}")
    print(f"  mode: {'serial' if args.serial else f'parallel cap {args.concurrency}'}")

    results = []
    if args.serial:
        for label in targets:
            results.append(gen_one(label))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.concurrency, len(targets))) as ex:
            futs = {ex.submit(gen_one, label): label for label in targets}
            for fut in concurrent.futures.as_completed(futs):
                label = futs[fut]
                try:
                    results.append(fut.result())
                except Exception as e:
                    results.append({"label": label, "error": str(e)})

    report = ROOT / "bench" / "results" / "veo_idle_library.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, indent=2))
    print(f"\nReport: {report}")
    ok = [r for r in results if "error" not in r]
    bad = [r for r in results if "error" in r]
    print(f"  OK   ({len(ok)}): {[r['label'] for r in ok]}")
    if bad:
        print(f"  FAIL ({len(bad)}): {[(r['label'], r['error'][:60]) for r in bad]}")


if __name__ == "__main__":
    main()
