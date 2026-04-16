#!/usr/bin/env python
"""Phase 0.3a — Generate 10-second test TTS audio for lip sync benchmark.

Uses ElevenLabs flash v2.5 (same model the backend already uses). The audio
represents a typical product pitch duration so the benchmark reflects real usage.
"""
from __future__ import annotations
import os, pathlib, time
from dotenv import dotenv_values

ROOT = pathlib.Path(__file__).resolve().parent.parent
env = dotenv_values(ROOT.parent / ".env")
for k, v in env.items():
    os.environ[k] = v

from elevenlabs import ElevenLabs

API_KEY = os.environ["ELEVENLABS_API_KEY"]
VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "KSGYe0fMqoZR5BREvtf1")

# ~10s of speech, reflecting a live product pitch
PITCH_TEXT = (
    "Okay okay okay, y'all are gonna LOVE this. This is hands-down the coolest thing I've "
    "picked up this month — I'm telling you, the second you see the detail on this, you're "
    "gonna want one. Seriously, hit the buy button below, you won't regret it!"
)

OUT = ROOT / "bench" / "audio" / "pitch_10s.mp3"
OUT.parent.mkdir(parents=True, exist_ok=True)

client = ElevenLabs(api_key=API_KEY)
t0 = time.perf_counter()
stream = client.text_to_speech.convert(
    text=PITCH_TEXT,
    voice_id=VOICE_ID,
    model_id="eleven_flash_v2_5",
    output_format="mp3_44100_128",
)
data = b"".join(stream)
OUT.write_bytes(data)
dt = time.perf_counter() - t0
print(f"TTS done in {dt:.2f}s  -> {OUT}  ({len(data)/1024:.0f} KB)")
print(f"Text: {PITCH_TEXT}")
