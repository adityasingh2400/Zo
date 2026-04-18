import json
import asyncio
import base64
import os
import sys
import tempfile
import logging
import httpx
import boto3
from config import AWS_REGION, BEDROCK_MODEL_ID

logger = logging.getLogger("empire.eyes")
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

# ── Cactus SDK (primary, on-device) ─────────────────────────────────────────

CACTUS_PYTHON_PATH = os.getenv("CACTUS_PYTHON_PATH", "/Users/aditya/Desktop/cactus/python/src")
CACTUS_WEIGHTS_PATH = os.getenv("CACTUS_WEIGHTS_PATH", "/Users/aditya/Desktop/cactus/weights/gemma-4-e4b-it")

CACTUS_AVAILABLE = False
_cactus_model = None

if CACTUS_PYTHON_PATH and os.path.exists(CACTUS_PYTHON_PATH):
    sys.path.insert(0, CACTUS_PYTHON_PATH)
    try:
        from cactus import cactus_init, cactus_complete, cactus_destroy
        CACTUS_AVAILABLE = True
        logger.info("Cactus SDK loaded from %s", CACTUS_PYTHON_PATH)
    except ImportError as e:
        logger.warning("Cactus SDK import failed: %s", e)
else:
    logger.info("Cactus SDK not found at %s, using Ollama fallback", CACTUS_PYTHON_PATH)


def _get_cactus_model():
    global _cactus_model
    if _cactus_model is None and CACTUS_AVAILABLE:
        _cactus_model = cactus_init(
            CACTUS_WEIGHTS_PATH.encode() if isinstance(CACTUS_WEIGHTS_PATH, str) else CACTUS_WEIGHTS_PATH,
            None, False
        )
    return _cactus_model


def _cactus_chat(messages: list, max_tokens: int = 256, images: list[str] | None = None) -> dict:
    """Call Gemma 4 via Cactus SDK. Returns parsed response dict."""
    model = _get_cactus_model()
    if model is None:
        return {"error": "Cactus model not loaded"}

    if images:
        messages[0]["images"] = images

    messages_json = json.dumps(messages)
    options_json = json.dumps({"max_tokens": max_tokens})
    raw = cactus_complete(model, messages_json, options_json, None, None)
    return json.loads(raw)


# ── Ollama (fallback) ────────────────────────────────────────────────────────

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma4:e4b")


async def _ollama_chat(messages: list, max_tokens: int = 256) -> dict:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={"model": GEMMA_MODEL, "messages": messages, "stream": False,
                      "options": {"num_predict": max_tokens}},
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        return {"error": str(e)}


async def _ollama_chat_with_image(prompt: str, image_b64: str, max_tokens: int = 256) -> dict:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={"model": GEMMA_MODEL, "messages": [{
                    "role": "user", "content": prompt, "images": [image_b64],
                }], "stream": False, "options": {"num_predict": max_tokens}},
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        return {"error": str(e)}


# ── Shared helpers ───────────────────────────────────────────────────────────

def _parse_json_from_text(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return None


# ── EYES: Product Analysis ───────────────────────────────────────────────────

async def analyze_with_gemma(frame_b64: str, voice_text: str) -> dict:
    """On-device product analysis. Tries Cactus SDK first, falls back to Ollama."""
    logger.info("[GEMMA] analyze_with_gemma called (frame: %d chars, cactus: %s)", len(frame_b64), CACTUS_AVAILABLE)
    prompt = f"""Look at this product image. The seller said: "{voice_text}"
Give a brief product analysis as JSON:
{{"name": "...", "category": "...", "materials": ["..."], "selling_points": ["...", "..."], "visual_details": ["...", "..."]}}"""

    # Try Cactus SDK first
    if CACTUS_AVAILABLE:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(base64.b64decode(frame_b64))
            img_path = f.name
        try:
            import asyncio
            result = await asyncio.to_thread(
                _cactus_chat,
                [{"role": "user", "content": prompt}],
                256,
                [img_path],
            )
            response_text = result.get("response", "")
            latency = result.get("total_time_ms", 0)

            parsed = _parse_json_from_text(response_text)
            if parsed:
                parsed["source"] = "cactus_on_device"
                parsed["latency_ms"] = int(latency)
                parsed["cost"] = "$0.00"
                return parsed

            return {"source": "cactus_on_device", "raw_response": response_text,
                    "latency_ms": int(latency), "cost": "$0.00"}
        finally:
            os.unlink(img_path)

    # Fallback to Ollama
    result = await _ollama_chat_with_image(prompt, frame_b64, max_tokens=256)
    if "error" in result:
        return {"source": "gemma4_unavailable", "description": str(result["error"]), "fallback": True}

    response_text = result.get("message", {}).get("content", "")
    latency = result.get("total_duration", 0) / 1_000_000

    parsed = _parse_json_from_text(response_text)
    if parsed:
        parsed["source"] = "ollama_on_device"
        parsed["latency_ms"] = int(latency)
        parsed["cost"] = "$0.00"
        return parsed

    return {"source": "ollama_on_device", "raw_response": response_text,
            "latency_ms": int(latency), "cost": "$0.00"}


async def classify_comment_gemma(comment: str) -> dict:
    """On-device comment classification."""
    prompt = f"""Classify this livestream comment and draft a short response.
Comment: "{comment}"
Reply as JSON only: {{"type": "question|compliment|objection|spam", "draft_response": "1 sentence"}}"""

    if CACTUS_AVAILABLE:
        import asyncio
        result = await asyncio.to_thread(_cactus_chat, [{"role": "user", "content": prompt}], 100)
        response_text = result.get("response", "")
        latency = result.get("total_time_ms", 0)
        parsed = _parse_json_from_text(response_text)
        if parsed:
            parsed["source"] = "cactus_on_device"
            parsed["latency_ms"] = int(latency)
            return parsed
        return {"type": "question", "source": "cactus_on_device", "latency_ms": int(latency)}

    result = await _ollama_chat([{"role": "user", "content": prompt}], max_tokens=100)
    if "error" in result:
        return {"type": "question", "source": "fallback"}
    response_text = result.get("message", {}).get("content", "")
    latency = result.get("total_duration", 0) / 1_000_000
    parsed = _parse_json_from_text(response_text)
    if parsed:
        parsed["source"] = "ollama_on_device"
        parsed["latency_ms"] = int(latency)
        return parsed
    return {"type": "question", "source": "ollama_on_device", "latency_ms": int(latency)}


async def parse_voice_intent_gemma(voice_text: str) -> dict:
    """On-device voice intent parsing."""
    prompt = f"""Parse this voice command from a seller:
"{voice_text}"
Extract as JSON only: {{"action": "sell|describe|compare", "price": "$X or null", "target_audience": "who or null", "product_notes": "any extra instructions or null"}}"""

    if CACTUS_AVAILABLE:
        import asyncio
        result = await asyncio.to_thread(_cactus_chat, [{"role": "user", "content": prompt}], 100)
        response_text = result.get("response", "")
        latency = result.get("total_time_ms", 0)
        parsed = _parse_json_from_text(response_text)
        if parsed:
            parsed["source"] = "cactus_on_device"
            parsed["latency_ms"] = int(latency)
            return parsed
        return {"action": "sell", "source": "cactus_on_device", "latency_ms": int(latency)}

    result = await _ollama_chat([{"role": "user", "content": prompt}], max_tokens=100)
    if "error" in result:
        return {"action": "sell", "price": None, "target_audience": None, "source": "fallback"}
    response_text = result.get("message", {}).get("content", "")
    latency = result.get("total_duration", 0) / 1_000_000
    parsed = _parse_json_from_text(response_text)
    if parsed:
        parsed["source"] = "ollama_on_device"
        parsed["latency_ms"] = int(latency)
        return parsed
    return {"action": "sell", "source": "ollama_on_device", "latency_ms": int(latency)}


# ── EYES: Claude Vision (cloud, rich analysis) ──────────────────────────────

async def analyze_and_script_claude(frame_b64: str, voice_text: str) -> dict:
    """Single Claude call: product analysis + sales script from image.
    Returns dict with product_data and sales_script."""
    logger.info("[CLAUDE] analyze_and_script called (frame: %d chars)", len(frame_b64))
    prompt = f"""You are analyzing a product for a live e-commerce stream.

SELLER'S VOICE (TRUST THIS for product identity — the seller knows what they're selling):
"{voice_text}"

Look at the product image for visual details (color, condition, materials, accessories visible).

Do TWO things. Return ONLY valid JSON:
{{
    "product": {{
        "name": "exact product name from seller's narration + visual confirmation",
        "category": "category",
        "materials": ["material1", "material2"],
        "selling_points": ["point1", "point2", "point3", "point4", "point5"],
        "target_audience": "who would buy this",
        "suggested_price_range": "$X - $Y"
    }},
    "script": "A compelling 30-second sales pitch (under 100 words). Enthusiastic but genuine. Reference specific visual details from the image. Include 2-3 selling points and a call to action. Spoken dialogue only, no stage directions."
}}"""

    media_type = "image/jpeg"
    if frame_b64[:4] == "iVBO":
        media_type = "image/png"

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": frame_b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    })

    response = await asyncio.to_thread(
        bedrock.invoke_model,
        modelId=BEDROCK_MODEL_ID, contentType="application/json",
        accept="application/json", body=body,
    )
    result = json.loads(response["body"].read())
    text = result["content"][0]["text"]

    parsed = _parse_json_from_text(text)
    if parsed:
        parsed["source"] = "claude_cloud"
        return parsed
    return {"raw": text, "source": "claude_cloud"}


async def analyze_with_claude(frame_b64: str, voice_text: str) -> dict:
    """Standalone product analysis (used by comment pipeline for product context)."""
    logger.info("[CLAUDE] analyze_with_claude called (frame: %d chars)", len(frame_b64))
    prompt = f"""Analyze this product for an e-commerce listing.
The seller said: "{voice_text}"
Return ONLY valid JSON with these fields:
{{
    "name": "product name",
    "category": "category",
    "materials": ["material1", "material2"],
    "selling_points": ["point1", "point2", "point3", "point4", "point5"],
    "target_audience": "who would buy this",
    "suggested_price_range": "$X - $Y"
}}"""

    media_type = "image/jpeg"
    if frame_b64[:4] == "iVBO":
        media_type = "image/png"

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 256,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": frame_b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    })

    response = await asyncio.to_thread(
        bedrock.invoke_model,
        modelId=BEDROCK_MODEL_ID, contentType="application/json",
        accept="application/json", body=body,
    )
    result = json.loads(response["body"].read())
    text = result["content"][0]["text"]

    parsed = _parse_json_from_text(text)
    if parsed:
        parsed["source"] = "claude_cloud"
        return parsed
    return {"raw": text, "source": "claude_cloud"}
