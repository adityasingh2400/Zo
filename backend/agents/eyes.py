import json
import base64
import os
import tempfile
import httpx
import boto3
from config import AWS_REGION, BEDROCK_MODEL_ID

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma4:e4b")


async def _ollama_chat(messages: list, max_tokens: int = 256) -> dict:
    """Call Gemma 4 via Ollama's local API."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": GEMMA_MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {"num_predict": max_tokens},
                },
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        return {"error": str(e)}


async def _ollama_chat_with_image(prompt: str, image_b64: str, max_tokens: int = 256) -> dict:
    """Call Gemma 4 vision via Ollama with an image."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": GEMMA_MODEL,
                    "messages": [{
                        "role": "user",
                        "content": prompt,
                        "images": [image_b64],
                    }],
                    "stream": False,
                    "options": {"num_predict": max_tokens},
                },
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        return {"error": str(e)}


async def analyze_with_gemma(frame_b64: str, voice_text: str) -> dict:
    """On-device product analysis via Gemma 4 on Ollama. Fast, free, private."""
    prompt = f"""Look at this product image. The seller said: "{voice_text}"
Give a brief product analysis as JSON:
{{"name": "...", "category": "...", "materials": ["..."], "selling_points": ["...", "..."], "visual_details": ["...", "..."]}}"""

    result = await _ollama_chat_with_image(prompt, frame_b64, max_tokens=256)

    if "error" in result:
        return {
            "source": "gemma4_unavailable",
            "description": f"Ollama not running: {result['error']}. Run: ollama pull gemma4:e4b && ollama serve",
            "fallback": True,
        }

    response_text = result.get("message", {}).get("content", "")
    latency = result.get("total_duration", 0) / 1_000_000  # ns to ms

    start = response_text.find("{")
    end = response_text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            parsed = json.loads(response_text[start:end])
            parsed["source"] = "gemma4_on_device"
            parsed["latency_ms"] = int(latency)
            parsed["cost"] = "$0.00"
            return parsed
        except json.JSONDecodeError:
            pass

    return {
        "source": "gemma4_on_device",
        "raw_response": response_text,
        "latency_ms": int(latency),
        "cost": "$0.00",
    }


async def classify_comment_gemma(comment: str) -> dict:
    """On-device comment classification via Gemma 4. Instant, free."""
    result = await _ollama_chat([{
        "role": "user",
        "content": f"""Classify this livestream comment and draft a short response.
Comment: "{comment}"
Reply as JSON only: {{"type": "question|compliment|objection|spam", "draft_response": "1 sentence"}}""",
    }], max_tokens=100)

    if "error" in result:
        return {"type": "question", "source": "fallback"}

    response_text = result.get("message", {}).get("content", "")
    latency = result.get("total_duration", 0) / 1_000_000

    start = response_text.find("{")
    end = response_text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            parsed = json.loads(response_text[start:end])
            parsed["source"] = "gemma4_on_device"
            parsed["latency_ms"] = int(latency)
            return parsed
        except json.JSONDecodeError:
            pass

    return {"type": "question", "source": "gemma4_on_device", "latency_ms": int(latency)}


async def parse_voice_intent_gemma(voice_text: str) -> dict:
    """On-device voice intent parsing via Gemma 4. Extracts price, target, action."""
    result = await _ollama_chat([{
        "role": "user",
        "content": f"""Parse this voice command from a seller:
"{voice_text}"
Extract as JSON only: {{"action": "sell|describe|compare", "price": "$X or null", "target_audience": "who or null", "product_notes": "any extra instructions or null"}}""",
    }], max_tokens=100)

    if "error" in result:
        return {"action": "sell", "price": None, "target_audience": None, "source": "fallback"}

    response_text = result.get("message", {}).get("content", "")
    latency = result.get("total_duration", 0) / 1_000_000

    start = response_text.find("{")
    end = response_text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            parsed = json.loads(response_text[start:end])
            parsed["source"] = "gemma4_on_device"
            parsed["latency_ms"] = int(latency)
            return parsed
        except json.JSONDecodeError:
            pass

    return {"action": "sell", "source": "gemma4_on_device", "latency_ms": int(latency)}


async def analyze_with_claude(frame_b64: str, voice_text: str) -> dict:
    """Rich product analysis via Claude on AWS Bedrock. Slower but more detailed."""
    prompt = f"""Analyze this product for an e-commerce listing.
The seller said: "{voice_text}"
Return ONLY valid JSON with these fields:
{{
    "name": "product name",
    "category": "category",
    "materials": ["material1", "material2"],
    "dimensions_estimate": "W x H x D estimate",
    "selling_points": ["point1", "point2", "point3", "point4", "point5"],
    "target_audience": "who would buy this",
    "suggested_price_range": "$X - $Y",
    "visual_details": ["detail1", "detail2", "detail3"]
}}"""

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": frame_b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    })

    response = bedrock.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )

    result = json.loads(response["body"].read())
    text = result["content"][0]["text"]

    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        parsed = json.loads(text[start:end])
        parsed["source"] = "claude_cloud"
        return parsed
    return {"raw": text, "source": "claude_cloud"}
