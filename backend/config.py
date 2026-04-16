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
