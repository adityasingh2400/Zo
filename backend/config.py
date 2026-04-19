import os
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-haiku-20241022-v1:0")
BEDROCK_MODEL_ID_HEAVY = os.getenv("BEDROCK_MODEL_ID_HEAVY", "anthropic.claude-sonnet-4-20250514-v1:0")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")

RUNPOD_POD_IP = os.getenv("RUNPOD_POD_IP", "")
RUNPOD_LIVETALKING_PORT = int(os.getenv("RUNPOD_LIVETALKING_PORT", "8010"))
RUNPOD_TRIPOSR_PORT = int(os.getenv("RUNPOD_TRIPOSR_PORT", "8020"))

# Lip-sync servers on RunPod (assumes SSH tunnel from phase0/scripts/open_tunnel.sh
# maps these to localhost). Override with LIPSYNC_HOST=<pod-ip> for direct access.
LIPSYNC_HOST = os.getenv("LIPSYNC_HOST", "127.0.0.1")
WAV2LIP_PORT = int(os.getenv("WAV2LIP_PORT", "8010"))
LATENTSYNC_PORT = int(os.getenv("LATENTSYNC_PORT", "8766"))
WAV2LIP_URL = f"http://{LIPSYNC_HOST}:{WAV2LIP_PORT}"
LATENTSYNC_URL = f"http://{LIPSYNC_HOST}:{LATENTSYNC_PORT}"

# Source state videos uploaded to the pod (prewarm targets)
POD_SPEAKING_1080P = os.getenv("POD_SPEAKING_1080P", "/workspace/state_pitching_pose_speaking_1080p.mp4")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

GEMMA_MODEL_PATH = os.getenv("GEMMA_MODEL_PATH", "weights/gemma-4-E4B-it")

BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))


# ── Avatar realism feature flags (Apr 18 build) ──────────────────────────────
# Every behavioural change for the YC submission demo lands behind a flag so
# we can flip any single piece off at the eleventh hour and still ship the
# rest. Defaults are ON because we want them on for the demo; setting any of
# them to "0" reverts to the prior behaviour.
def _flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


# Audio-first playback: broadcast a `comment_response_audio` event the moment
# TTS finishes, kick Wav2Lip render in the background, dashboard plays audio
# immediately and crossfades video underneath when ready.
USE_AUDIO_FIRST = _flag("USE_AUDIO_FIRST", "1")

# KaraokeCaptions component on the stage — word-by-word reveal synced to the
# playing <audio> element. Synthetic word timings (whitespace-split, evenly
# distributed across audio duration) since we're not on Cartesia today.
USE_KARAOKE = _flag("USE_KARAOKE", "1")

# Veo 30s pitch playback path — pre-rendered "pitching pose" clip + TTS
# audio overlay + karaoke. Shaves the opening pitch from ~8-15s to ~600ms.
USE_PITCH_VEO = _flag("USE_PITCH_VEO", "1")

# Listening-attentive backchannel pose on mic press. Visual only (no "mhm"
# audio per REVISIONS §8 — audio overlap risk during user speech isn't worth
# the win).
USE_BACKCHANNEL = _flag("USE_BACKCHANNEL", "1")

# Speculative bridge clip immediately after voice_transcript lands, before
# the router decides. Fills the gap between transcript and response.
USE_SPECULATIVE_BRIDGE = _flag("USE_SPECULATIVE_BRIDGE", "1")

# Lip-sync provider hook. Wav2Lip is the only path today; MuseTalk was cut
# post-review (REVISIONS §6). Kept as a forward-compat env so post-submission
# we can wire a second provider without touching call sites.
LIPSYNC_PROVIDER = os.getenv("LIPSYNC_PROVIDER", "wav2lip").strip().lower()

# Pad live wav2lip output so the video duration matches the audio. Wav2Lip's
# mel-chunking always produces a video ~120-180ms shorter than the audio
# (MEL_STEP=16 windowing artifact, structural — same drift on flash, v3,
# every length, every fps). The pod's `-shortest` ffmpeg mux then truncates
# the audio tail and the dashboard's 150ms duration handshake (LiveStage.jsx)
# either skips the video or shows a silent video tail after the audio cuts.
#
# When ON (default): re-mux the wav2lip output locally with the FULL audio +
# video padded by holding the last frame for the gap. Drift drops to ±20ms;
# audio plays in full; handshake never fires. Cost: +300-500ms ffmpeg re-mux
# per live escalate (on a 5-7s warm budget).
#
# When OFF: original behavior. ~140ms drift, occasional handshake skip on
# v3 audio. Set USE_LIVE_PAD=0 if pad is interfering with a live demo.
USE_LIVE_PAD = _flag("USE_LIVE_PAD", "1")
