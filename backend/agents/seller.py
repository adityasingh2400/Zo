import json
import base64
import asyncio
import boto3
import httpx
from elevenlabs import ElevenLabs
from config import (
    AWS_REGION, BEDROCK_MODEL_ID,
    ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID,
    RUNPOD_POD_IP, RUNPOD_LIVETALKING_PORT,
)

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
eleven = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None


async def generate_sales_script(product_data: dict, voice_text: str) -> str:
    """Generate a 30-second sales pitch from product data."""
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "messages": [{
            "role": "user",
            "content": f"""Write a compelling 30-second sales pitch for this product.

Product data: {json.dumps(product_data)}
Seller's instruction: "{voice_text}"

Rules:
- Reference specific visual details from the product analysis
- Be enthusiastic but genuine
- Include 2-3 selling points
- End with a call to action
- Keep it under 100 words (for 30 seconds of speech)
- Write it as spoken dialogue, not a script with stage directions""",
        }],
    })

    response = bedrock.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


async def generate_comment_response(
    comment: str, product_data: dict, comment_type: str = "question"
) -> str:
    """Generate a natural response to a viewer comment."""
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 128,
        "messages": [{
            "role": "user",
            "content": f"""You are an AI sales avatar on a livestream selling a product.
A viewer just commented: "{comment}"
Comment type: {comment_type}

Product info: {json.dumps(product_data)}

Write a 1-2 sentence natural spoken response. Be friendly, specific to the product.
No stage directions. Just the words the avatar would say.""",
        }],
    })

    response = bedrock.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


async def text_to_speech(text: str) -> bytes:
    """Convert text to speech via ElevenLabs. Returns audio bytes (mp3)."""
    if not eleven:
        return b""

    audio_gen = eleven.text_to_speech.convert(
        text=text,
        voice_id=ELEVENLABS_VOICE_ID,
        model_id="eleven_flash_v2_5",
        output_format="mp3_44100_128",
    )
    chunks = []
    for chunk in audio_gen:
        chunks.append(chunk)
    return b"".join(chunks)


async def send_audio_to_livetalking(audio_bytes: bytes) -> dict:
    """Send TTS audio to LiveTalking on RunPod for lip-synced avatar rendering."""
    if not RUNPOD_POD_IP:
        return {"error": "RunPod not configured"}

    audio_b64 = base64.b64encode(audio_bytes).decode()
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"http://{RUNPOD_POD_IP}:{RUNPOD_LIVETALKING_PORT}/tts",
                json={"audio": audio_b64, "format": "mp3"},
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        return {"error": str(e)}
