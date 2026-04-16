import os
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-sonnet-4-20250514-v1:0")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")

RUNPOD_POD_IP = os.getenv("RUNPOD_POD_IP", "")
RUNPOD_LIVETALKING_PORT = int(os.getenv("RUNPOD_LIVETALKING_PORT", "8010"))
RUNPOD_TRIPOSR_PORT = int(os.getenv("RUNPOD_TRIPOSR_PORT", "8020"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

GEMMA_MODEL_PATH = os.getenv("GEMMA_MODEL_PATH", "weights/gemma-4-E4B-it")

BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
