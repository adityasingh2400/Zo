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


# ── Feature flags — actual gating status as of Apr 2026 ─────────────────────
# Each flag's status (FUNCTIONAL / COSMETIC) reflects whether toggling it to
# "0" actually changes runtime behavior in the current code paths. Cosmetic
# flags are kept for forward-compat / observability and may be removed if
# they're still cosmetic by the next sweep.

# COSMETIC. The cloud-escalate path always uses audio-first dispatch
# (standalone <audio> + KaraokeCaptions over a speaking-pose loop on Tier 1)
# regardless of this flag. The "no-Wav2Lip" decision is hard-coded in
# api_respond_to_comment for pristine quality. To restore a pre-Wav2Lip
# kill-switch fallback path, gate the new flow on this flag and fall back
# to a synchronous render in the else branch — currently no else branch
# exists. Logged at boot for visibility.
USE_AUDIO_FIRST = _flag("USE_AUDIO_FIRST", "1")

# COSMETIC. KaraokeCaptions always renders when audioPlaying is set on the
# dashboard (it's the standard response visual now). To make this a real
# kill-switch, gate the <KaraokeCaptions /> mount in LiveStage.jsx behind
# the flag.
USE_KARAOKE = _flag("USE_KARAOKE", "1")

# FUNCTIONAL. Read in run_sell_pipeline (main.py): when ON, the sell
# pipeline uses _run_audio_first_pitch (cached pitch MP3 + word timings +
# muted Veo speaking-pose loop). When OFF, falls back to _run_wav2lip_pitch
# (synchronous Wav2Lip render of the script audio against the active
# substrate). 8-15s render vs ~600ms audio dispatch.
USE_PITCH_VEO = _flag("USE_PITCH_VEO", "1")

# FUNCTIONAL. Read in dashboard_ws (main.py) when handling mic_pressed:
# when ON, fires director.play_listening_attentive() to crossfade in the
# attentive-listening pose on Tier 1 within ~50ms of the mic press.
USE_BACKCHANNEL = _flag("USE_BACKCHANNEL", "1")

# FUNCTIONAL. Read in api_voice_comment (main.py) after the transcript
# broadcast: when ON, fires _fire_speculative_bridge() in parallel with
# run_routed_comment to play a "neutral" bridge clip on Tier 1 while
# classify+router decide. Mostly redundant now that reading_chat fires
# instantly inside run_routed_comment — leaving the flag wired so it can
# be A/B tested.
USE_SPECULATIVE_BRIDGE = _flag("USE_SPECULATIVE_BRIDGE", "1")

# COSMETIC. No call sites branch on this value today. Both Wav2Lip (legacy
# pitch path via _run_wav2lip_pitch) and LatentSync (pre-render scripts)
# are addressed by their own dedicated render functions in agents/seller.py.
# Kept as a forward-compat env so post-submission we can wire a second
# provider without touching call sites.
LIPSYNC_PROVIDER = os.getenv("LIPSYNC_PROVIDER", "wav2lip").strip().lower()

# COSMETIC. _render_response_video and _render_and_broadcast_video (the
# helpers that read this flag) were deleted when api_respond_to_comment
# switched to no-Wav2Lip. The flag remains to avoid breaking .env files in
# the wild. _run_wav2lip_pitch (the only remaining Wav2Lip caller) calls
# render_comment_response_wav2lip directly without padding.
USE_LIVE_PAD = _flag("USE_LIVE_PAD", "1")
