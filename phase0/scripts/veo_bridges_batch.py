#!/usr/bin/env python
"""Generate the silent body-language bridge clip library via straight Veo 3.1.

These are the substrate clips the Director plays under a Wav2Lip mouth-region
overlay at runtime. They are NOT verbal stallers — every clip is silent body
language only. The avatar's mouth stays in a natural relaxed position so the
runtime lip-sync layer can overwrite the mouth region cleanly.

Why this exists: the original bridge library (`backend/agents/bridge_clips.py`
+ `phase0/assets/clips/`) used TTS → LatentSync to produce the avatar literally
saying phrases like "great question, hold on one second." Diffusion-based
mouth-region inpainting at certain phoneme positions produced the visible
"cracked mouth" artifact you can see at /dev/clips. Wrong tool, wrong model.
The new clips skip TTS + lip-sync entirely.

Two key tricks for the new model:

  1. **Anchor pose** — every clip starts AND ends in the EXACT same neutral
     resting pose (hands relaxed at waist, lips closed in soft natural smile,
     direct eye contact). Mid-clip is the body-language gesture for the intent.
     This makes idle→bridge→idle (and even bridge→bridge) crossfades visually
     seamless because the start/end frames match across the whole library.

  2. **Same reference image** — every prompt uses the canonical portrait so
     the avatar identity, framing, and bedroom background are pixel-locked.

Library shape: 5 intents × 6 variants = 30 clips, 8 s each at 1080p / 9:16.

Usage:
  # See every prompt that would be sent, without burning Veo credit:
  python phase0/scripts/veo_bridges_batch.py --dry-run

  # Render everything (10-15 min wall-clock with default concurrency):
  python phase0/scripts/veo_bridges_batch.py

  # Render specific intents only:
  python phase0/scripts/veo_bridges_batch.py --only question,objection

  # Render fewer variants per intent:
  python phase0/scripts/veo_bridges_batch.py --variants 3

  # Re-render existing files (otherwise idempotent — done files are skipped):
  python phase0/scripts/veo_bridges_batch.py --overwrite

  # Throttle concurrency (default 6 to stay well under Veo 3.1 50 RPM):
  python phase0/scripts/veo_bridges_batch.py --concurrency 4
"""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import os
import pathlib
import sys
import time

from dotenv import dotenv_values

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENV = dotenv_values(ROOT.parent / ".env")
for _k, _v in ENV.items():
    if _v is not None:
        os.environ.setdefault(_k, _v)

# Imported AFTER env load so the SDK picks up GCP project + ADC.
from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

# Vertex AI path (NOT AI Studio). genai.Client(vertexai=True, ...) routes
# through *-aiplatform.googleapis.com using gcloud Application Default
# Credentials (~/.config/gcloud/application_default_credentials.json).
# Vertex has dramatically higher Veo quota than the api_key path
# (which uses AI Studio's per-day daily cap).
GCP_PROJECT = os.environ.get("GCP_PROJECT_ID")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
if not GCP_PROJECT:
    print("FATAL: GCP_PROJECT_ID missing from .env", file=sys.stderr)
    sys.exit(1)

PORTRAIT = ROOT / "assets" / "portraits" / "portrait.png"
OUT_ROOT = ROOT / "assets" / "bridges"
MANIFEST_PATH = OUT_ROOT / "manifest.json"

# ─────────────────────────────────────────────────────────────────────────────
# Prompt template — every bridge clip uses this skeleton with the [MID]
# section swapped per variant.
#
# Two design choices that make the library work end-to-end with Wav2Lip:
#
# 1. Anchor pose at start AND end — when every clip starts and ends in the
#    same neutral conversational rest, idle→bridge→idle (and even
#    bridge→bridge) crossfades have nothing to hide.
#
# 2. The avatar SPEAKS throughout (natural conversational mouth motion, no
#    specific words). Wav2Lip is a mouth-region inpainter — it works
#    dramatically better when the surrounding face (jaw, cheeks, brows) is
#    already in "speaking mode" with natural micro-motion. A closed-mouth
#    substrate makes the inpainted mouth look pasted on; a talking substrate
#    blends invisibly. Same recipe `phase0/scripts/veo_states_batch.py` uses
#    for `state_*_speaking_1080p.mp4`.
#
# Note: deliberately no mention of the words "audio" or "lip-sync" — Veo's
# safety filter rejected ~10% of our first batch on prompts that even
# referenced them in negation. The prompt just describes what the camera
# sees; what we do with it downstream is none of Veo's concern.
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """\
The same woman from the reference image stands centered in her warm cozy
bedroom, with the soft string-lights backdrop and ivy garlands identical to
the reference frame. Locked-off camera on a tripod, no zoom, no pan.

OPENING POSE (frame 1 — must be reproduced exactly at the closing frame):
  - Hands relaxed at her waist, fingers gently open
  - Lips gently parted in a soft conversational smile (a natural between-
    phrases rest position, ready to continue speaking)
  - Calm attentive expression, direct eye contact with camera
  - Neutral upright posture, weight evenly on both feet
  - Soft natural blink rhythm

MID-CLIP MOTION ({duration_seconds} second window):
  {mid}

CLOSING POSE (final frame):
  Returns to the EXACT same opening pose described above — hands at waist,
  lips gently parted in soft conversational smile, calm expression, direct
  eye contact. The closing pose must match the opening pose so the clip
  seamlessly chains into the next one. Hands settle back to waist within
  the last ~1 second so the final still frame is the neutral conversational
  resting pose.

SPEAKING / MOUTH MOTION:
  Throughout the entire clip the avatar is mid-conversation — speaking
  softly and naturally with normal conversational phonemes. Mouth opens
  and closes at a calm rhythm, jaw and cheeks have natural speech
  micro-motion, occasional small teeth visibility consistent with normal
  speech. NO specific words or recognizable phrases — just natural
  conversational mouth motion as if she's casually chatting. NOT yelling,
  NOT exaggerated, NOT silent.

ENVIRONMENT (locked):
  Identical bedroom set, identical lighting, identical wardrobe (cream
  knit sweater), identical hair, identical framing. The only thing that
  changes between clips in this batch is the body-language gesture above.
"""

# 5 intents × 6 variants. Each `mid` is the intent-specific gesture that
# fills the middle ~6 seconds; the framing wraps it so opening and closing
# poses are anchored. Keep these short and visually distinct — the user
# will pick the best 1-2 per intent and we'll discard the rest.
INTENTS: dict[str, list[str]] = {
    "question": [
        # A
        "Right hand rises slowly from waist to chest height with palm open and "
        "facing up, as if presenting a thought, while the head tilts gently to "
        "the right. The hand holds for ~1 s, then settles back to waist.",
        # B
        "Right hand rises and gently cups the chin in a brief considering "
        "gesture, eyes look up-left for ~0.5 s as if recalling, then the hand "
        "lowers back to waist as the eyes return to camera.",
        # C
        "Both hands open at chest level, palms slightly turned up, in a "
        "thoughtful gathering gesture. The torso leans forward maybe 5 cm "
        "before the hands return to waist.",
        # D
        "Right hand rises with a single open finger curling upward briefly "
        "in a warm 'ah, good point' acknowledgement gesture, eyebrows raise "
        "warmly, then the hand softens and lowers smoothly back to waist.",
        # E
        "Eyebrows raise in a warm interested expression, right hand brushes a "
        "lock of hair behind the ear, then both hands return to waist.",
        # F
        "Single soft head nod (yes-acknowledgment), right hand opens at chest "
        "level palm up briefly, then both hands settle back to waist.",
    ],
    "objection": [
        # A
        "Right hand raises palm-out at chest height in a soft 'hear me out' "
        "gesture (NOT a stop sign — relaxed, conciliatory). Held for ~1 s, "
        "head tilts slightly forward, then hand lowers gracefully to waist.",
        # B
        "Both hands open at chest level, palms turned slightly up, while the "
        "head shakes very softly side-to-side once or twice (gentle 'no no, "
        "let me explain'). Hands settle back to waist.",
        # C
        "Slight forward lean from the waist, both hands give a small open-palm "
        "shrug at chest height, eyebrows raise in earnest reassurance, then "
        "the body straightens and hands return to waist.",
        # D
        "Right hand sweeps from over the heart outward toward camera in a "
        "warm earnest gesture, eyes hold direct eye contact, then the hand "
        "returns smoothly to waist.",
        # E
        "Both hands clasp briefly together at chest level (conciliatory) then "
        "open outward into a small palms-up gesture before settling back to "
        "waist.",
        # F
        "Soft knowing head tilt to the right with a warm closed-mouth smile, "
        "right hand lifts to chest with palm slightly up for ~1 s, then "
        "lowers back to waist.",
    ],
    "compliment": [
        # A
        "Right hand presses softly to the heart, head tilts down ~10 degrees "
        "in a brief bashful gesture, eyes flutter closed for half a second, "
        "then everything returns to the opening pose.",
        # B
        "Both hands rise to press together gently near the collarbone in a "
        "modest 'aww thank you' gesture, slight warm smile widens, then "
        "hands lower back to waist.",
        # C
        "Slight bashful smile, right hand brushes once through the hair on "
        "the side of the head, then settles back to waist.",
        # D
        "Eyes brighten with a soft genuine smile, right hand gives a small "
        "wave of gratitude at chest height, then returns to waist.",
        # E
        "Right hand rises and lightly touches the side of the mouth in a "
        "surprised-pleased gesture, head tilts slightly, then hand returns "
        "to waist.",
        # F
        "Both hands clasp gently at the chest in a heartfelt thank-you, held "
        "for ~1 s, then open and lower back to waist.",
    ],
    "neutral": [
        # A
        "Right hand makes a generic open-handed gesture at chest height "
        "(palm up, gentle outward arc), then returns smoothly to waist.",
        # B
        "Both hands move outward in a small expansive 'you know how it is' "
        "gesture at chest level, then settle back to waist.",
        # C
        "Slight head nod with an attentive warm smile, hands stay at waist "
        "but fingers flex once gently. Otherwise still.",
        # D
        "Subtle weight shift onto the right foot, right hand makes an easy "
        "open-handed gesture at hip height, weight returns and hand returns "
        "to waist.",
        # E
        "Right hand briefly rests on the right hip in a casual conversational "
        "stance, then lowers back to waist.",
        # F
        "Calm soft breath, gentle warm smile widens slightly, micro-adjustment "
        "of stance, hands stay at waist throughout.",
    ],
    "intro": [
        # A
        "Both arms gently open outward and slightly upward in a warm "
        "welcoming gesture (palms up, elbows soft), held for ~1 s, then "
        "lower smoothly back to waist.",
        # B
        "Right hand rises to shoulder height in an energetic friendly wave "
        "(2-3 small motions side to side), big genuine smile, then hand "
        "lowers to waist.",
        # C
        "Brief two-handed open-palms 'ta-da' gesture at chest height, "
        "energetic smile, then hands return to waist.",
        # D
        "Right hand presses to the heart for ~1 s in a warm 'so glad you're "
        "here' gesture, then opens outward into a small welcome gesture "
        "before settling back to waist.",
        # E
        "Slight forward bow from the shoulders (NOT a deep bow, just a "
        "polite warm acknowledgement), then straightens back up and hands "
        "stay at waist.",
        # F
        "Both palms turn upward at chest level in a soft welcoming offer, "
        "head tilts gently, then hands lower back to waist.",
    ],
    # ── pitch substrate ───────────────────────────────────────────────────
    # The looping body-language clip that runs UNDER the pitch audio after
    # a phone video upload. Replaces `state_pitching_pose_speaking_1080p.mp4`.
    # Plays for the duration of the TTS pitch audio (~15-25 s) — Veo's max
    # render is 8 s, so this loops 2-3x. Anchor pose at start AND end keeps
    # the loop seam invisible. Talking substrate (mouth moving naturally)
    # so Wav2Lip overlays cleanly.
    "pitch": [
        # A — confident product presentation
        "Both hands rise from waist to chest, cup softly around an imaginary "
        "product as if presenting it to camera, then open outward in a small "
        "flowing 'and here it is' gesture. Slight forward lean toward camera, "
        "warm engaged sales-energy smile. Hands settle back to waist.",
        # B — animated descriptive
        "Hands move in flowing descriptive arcs at chest height — counting "
        "features on fingers briefly, then a small open-handed 'check this "
        "out' motion. Slight rhythmic side-to-side sway. Bright animated "
        "smile. Hands return smoothly to waist.",
    ],
    # ── welcome (Tier 1 ambient — re-greets new viewers) ──────────────────
    # Plays occasionally during idle to convey "I'm welcoming new people who
    # just joined the stream." Talking substrate so Wav2Lip can overlay if
    # we ever want a TTS-driven version; ambient use is silent.
    "welcome": [
        # A — single friendly wave
        "Right hand rises from waist to shoulder height in a warm friendly "
        "wave (2-3 small motions side to side), big genuine smile, eyes "
        "brighten, slight head bob, then hand lowers smoothly back to waist.",
        # B — two-handed enthusiastic welcome
        "Both arms rise slightly outward and upward in an enthusiastic two-"
        "handed welcome wave (1-2 motions), big animated smile, slight "
        "forward bob, then hands lower back to waist.",
    ],
}

DURATION_S = 8
RESOLUTION = "1080p"
ASPECT = "9:16"
# Vertex AI uses the production model name (not the AI Studio "preview" suffix).
MODEL = "veo-3.1-generate-001"
VARIANT_LABELS = list("ABCDEFGH")  # supports up to 8 variants per intent


def build_prompt(mid: str) -> str:
    return PROMPT_TEMPLATE.format(mid=mid.strip(), duration_seconds=DURATION_S)


def out_path_for(intent: str, variant: str) -> pathlib.Path:
    return OUT_ROOT / intent / f"{intent}_{variant}.mp4"


def _submit_with_429_retry(client, *, label: str, prompt: str,
                           image_bytes: bytes,
                           max_retries: int = 5):
    """Submit a Veo generation with exponential-backoff retry on transient
    backend errors:

      - 429 RESOURCE_EXHAUSTED   → per-minute or per-day quota wall
      - 503 / 'Service is currently unavailable' → Vertex backend hiccup
      - 500 INTERNAL              → ditto

    Backoff starts at 30 s and doubles (capped at 5 min). Other errors
    (auth failures, prompt-rejected by safety filter, etc.) raise
    immediately — those won't fix themselves with a retry."""
    delay = 30.0
    for attempt in range(1, max_retries + 1):
        try:
            return client.models.generate_videos(
                model=MODEL,
                prompt=prompt,
                image=types.Image(image_bytes=image_bytes, mime_type="image/png"),
                config=types.GenerateVideosConfig(
                    aspect_ratio=ASPECT,
                    duration_seconds=DURATION_S,
                    resolution=RESOLUTION,
                ),
            )
        except Exception as e:
            msg = str(e)
            is_429 = "429" in msg or "RESOURCE_EXHAUSTED" in msg
            is_503 = ("503" in msg or "currently unavailable" in msg.lower()
                      or "INTERNAL" in msg or "code': 14" in msg)
            transient = is_429 or is_503
            if not transient or attempt == max_retries:
                raise
            kind = "429 quota" if is_429 else "503/transient"
            print(f"[{label}] {kind} — retry {attempt}/{max_retries-1} "
                  f"in {int(delay)}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 300.0)


def gen_one(intent: str, variant: str, prompt: str, *, overwrite: bool) -> dict:
    """Submit + poll + download a single Veo render. Idempotent on disk:
    if the target .mp4 exists and --overwrite is not set, returns 'skipped'."""
    out = out_path_for(intent, variant)
    label = f"{intent}_{variant}"
    if out.exists() and not overwrite:
        size_kb = out.stat().st_size // 1024
        print(f"[{label}] SKIP — already on disk ({size_kb} KB)", flush=True)
        return {"label": label, "intent": intent, "variant": variant,
                "path": str(out), "skipped": True, "size_kb": size_kb}

    out.parent.mkdir(parents=True, exist_ok=True)
    image_bytes = PORTRAIT.read_bytes()

    print(f"[{label}] submit Veo {RESOLUTION} {ASPECT} {DURATION_S}s",
          flush=True)
    t0 = time.perf_counter()

    client = genai.Client(vertexai=True, project=GCP_PROJECT,
                          location=GCP_LOCATION)
    op = _submit_with_429_retry(client, label=label, prompt=prompt,
                                image_bytes=image_bytes)
    print(f"[{label}] submitted ({time.perf_counter() - t0:.1f}s)", flush=True)

    poll = 0
    while not op.done:
        poll += 1
        time.sleep(8)
        op = client.operations.get(op)
        elapsed = time.perf_counter() - t0
        if poll % 3 == 0:
            print(f"[{label}] poll {poll} elapsed={elapsed:.0f}s done={op.done}",
                  flush=True)
        if elapsed > 600:
            raise TimeoutError(f"{label} timed out after 10 min")

    if op.error:
        raise RuntimeError(f"{label} Veo error: {op.error}")

    resp = op.response
    videos = (
        getattr(resp, "generated_videos", None)
        or getattr(resp, "videos", None)
    )
    if not videos and hasattr(resp, "model_dump"):
        raw = resp.model_dump()
        videos = raw.get("generated_videos") or raw.get("videos")
    if not videos:
        raise RuntimeError(f"{label} no videos in response: {resp}")

    v = videos[0]
    video_file = (
        v.video if hasattr(v, "video")
        else (v.get("video") if isinstance(v, dict) else None)
    )

    # Vertex returns either a File object (with .save) or a dict with a
    # gs:// URI (which we resolve via the SDK's files.download — it handles
    # ADC auth). The plain `https?://` + x-goog-api-key fallback only
    # applied to the AI Studio path and is removed.
    if video_file and hasattr(video_file, "save"):
        try:
            client.files.download(file=video_file)
        except Exception as e:
            print(f"  [{label}] download() warning: {e}")
        video_file.save(str(out))
    elif isinstance(video_file, dict) and ("uri" in video_file or "gcsUri" in video_file):
        # Vertex Veo videos land on a Google Cloud Storage URI — fetch via
        # the SDK so ADC handles auth. Falls back to gsutil if the SDK
        # path doesn't return bytes directly.
        from google.cloud import storage  # type: ignore
        gcs_uri = video_file.get("gcsUri") or video_file["uri"]
        if gcs_uri.startswith("gs://"):
            bucket_name, blob_name = gcs_uri[5:].split("/", 1)
            sclient = storage.Client(project=GCP_PROJECT)
            blob = sclient.bucket(bucket_name).blob(blob_name)
            blob.download_to_filename(str(out))
        else:
            raise RuntimeError(f"{label} non-gs URI on Vertex: {gcs_uri}")
    else:
        raise RuntimeError(f"{label} unknown video payload: {type(video_file)}")

    total = time.perf_counter() - t0
    size_kb = out.stat().st_size // 1024
    print(f"[{label}] DONE {total:.0f}s -> {out.relative_to(ROOT.parent)} "
          f"({size_kb} KB)", flush=True)
    return {
        "label": label,
        "intent": intent,
        "variant": variant,
        "path": str(out),
        "url": f"/bridges/{intent}/{out.name}",
        "wall_sec": round(total, 1),
        "size_kb": size_kb,
        "skipped": False,
    }


def write_manifest(results: list[dict]) -> None:
    """Write a manifest grouped by intent, ready for pick_bridge_clip to read.
    Schema mirrors the existing bridge manifests so the consumer code can
    opt-in to the new library with a one-line path change."""
    grouped: dict[str, list[dict]] = {}
    for r in results:
        if r.get("skipped") or r.get("error"):
            # Still include skipped entries — the file exists on disk and
            # we want pick_bridge_clip to discover it.
            if r.get("error"):
                continue
        intent = r.get("intent")
        if not intent:
            # Reconstruct from label for skipped entries that came back from
            # gen_one with limited fields.
            label = r.get("label", "")
            if "_" in label:
                intent, _v = label.rsplit("_", 1)
        if not intent:
            continue
        path = pathlib.Path(r["path"])
        rel = path.relative_to(ROOT.parent)
        grouped.setdefault(intent, []).append({
            "variant": r.get("variant") or path.stem.split("_")[-1],
            "file": str(rel),
            "url": f"/bridges/{intent}/{path.name}",
            "size_kb": r.get("size_kb"),
        })
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(grouped, indent=2))
    print(f"\n[manifest] wrote {MANIFEST_PATH.relative_to(ROOT.parent)} "
          f"({sum(len(v) for v in grouped.values())} entries across "
          f"{len(grouped)} intents)")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Print every prompt that would be sent, no API calls.")
    ap.add_argument("--only", default="",
                    help="Comma-separated intents to render. "
                         f"Default = all ({','.join(INTENTS)}).")
    ap.add_argument("--variants", type=int, default=6,
                    help="How many variants per intent (1-8). Default 6.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-render even if the target file already exists.")
    ap.add_argument("--concurrency", type=int, default=6,
                    help="Max concurrent Veo submissions. Default 6 "
                         "(Veo 3.1 limit is 50 RPM — 6 leaves plenty of "
                         "headroom and avoids overwhelming the GCP polling).")
    return ap.parse_args()


def main():
    args = parse_args()

    if not PORTRAIT.exists():
        print(f"FATAL: portrait missing at {PORTRAIT}", file=sys.stderr)
        sys.exit(1)

    intents = (
        [s.strip() for s in args.only.split(",") if s.strip()]
        if args.only else list(INTENTS.keys())
    )
    unknown = [i for i in intents if i not in INTENTS]
    if unknown:
        print(f"unknown intents: {unknown}; available={list(INTENTS)}",
              file=sys.stderr)
        sys.exit(2)

    n_variants = max(1, min(args.variants, 8))

    # Build the full submission list = (intent, variant_letter, prompt).
    jobs: list[tuple[str, str, str]] = []
    for intent in intents:
        mids = INTENTS[intent][:n_variants]
        for letter, mid in zip(VARIANT_LABELS, mids):
            jobs.append((intent, letter, build_prompt(mid)))

    print(f"Plan: {len(jobs)} clips ({len(intents)} intents × "
          f"{n_variants} variants), {DURATION_S}s @ {RESOLUTION} {ASPECT}")
    print(f"Output root: {OUT_ROOT.relative_to(ROOT.parent)}")
    print(f"Portrait:    {PORTRAIT.relative_to(ROOT.parent)}")
    print(f"Concurrency: {args.concurrency} simultaneous renders")

    if args.dry_run:
        print("\n=== DRY RUN — no API calls ===")
        for intent, letter, prompt in jobs:
            print(f"\n────────── {intent}_{letter} ──────────")
            print(prompt.rstrip())
        print(f"\n[dry-run] {len(jobs)} prompts ready. "
              f"Re-run without --dry-run to render.")
        return

    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as ex:
        futs = {
            ex.submit(gen_one, intent, letter, prompt,
                      overwrite=args.overwrite): (intent, letter)
            for intent, letter, prompt in jobs
        }
        for fut in concurrent.futures.as_completed(futs):
            intent, letter = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"[{intent}_{letter}] FAILED: {e}")
                results.append({
                    "label": f"{intent}_{letter}",
                    "intent": intent,
                    "variant": letter,
                    "path": str(out_path_for(intent, letter)),
                    "error": str(e),
                })

    write_manifest(results)

    failed = [r for r in results if r.get("error")]
    skipped = [r for r in results if r.get("skipped")]
    rendered = [r for r in results
                if not r.get("error") and not r.get("skipped")]
    print(f"\n=== summary === rendered={len(rendered)} "
          f"skipped={len(skipped)} failed={len(failed)}")
    if failed:
        print("Failures:")
        for r in failed:
            print(f"  {r['label']}: {r['error']}")


if __name__ == "__main__":
    main()
