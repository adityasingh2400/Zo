import asyncio
import base64
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time

import boto3
import httpx

from config import AWS_REGION, BEDROCK_MODEL_ID

logger = logging.getLogger("empire.eyes")
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

# ── Cactus SDK (primary, on-device) ─────────────────────────────────────────

CACTUS_PYTHON_PATH = os.path.expanduser(os.getenv("CACTUS_PYTHON_PATH", "~/cactus/python/src"))
CACTUS_WEIGHTS_PATH = os.path.expanduser(os.getenv("CACTUS_WEIGHTS_PATH", "~/cactus/weights/gemma-4-e4b-it"))
# Whisper-tiny used for on-device voice transcription. Gemma 4 E4B is the
# multimodal flagship, but whisper-tiny's cactus_transcribe path is the
# proven / fastest route for speech-to-text (62 MB vs 8 GB, sub-second).
CACTUS_WHISPER_WEIGHTS_PATH = os.path.expanduser(os.getenv("CACTUS_WHISPER_WEIGHTS_PATH", "~/cactus/weights/whisper-tiny"))

CACTUS_AVAILABLE = False
_cactus_model = None
_cactus_whisper_model = None
# Cactus' C library is not re-entrant on a single handle. Concurrent calls
# crash the whole Python process with no traceback. Serialize inference per
# handle. These are threading.Locks (not asyncio) because the raw cactus
# calls run inside asyncio.to_thread workers — the sync lock correctly
# blocks across worker threads.
_cactus_whisper_lock = threading.Lock()
_cactus_model_lock = threading.Lock()

if CACTUS_PYTHON_PATH and os.path.exists(CACTUS_PYTHON_PATH):
    sys.path.insert(0, CACTUS_PYTHON_PATH)
    try:
        from cactus import cactus_complete, cactus_destroy, cactus_init, cactus_transcribe
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
    """Call Gemma 4 via Cactus SDK. Returns parsed response dict.

    Serialized on _cactus_model_lock because concurrent `cactus_complete`
    calls on one handle crash the process."""
    model = _get_cactus_model()
    if model is None:
        return {"error": "Cactus model not loaded"}

    if images:
        messages[0]["images"] = images

    messages_json = json.dumps(messages)
    options_json = json.dumps({"max_tokens": max_tokens})
    with _cactus_model_lock:
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

async def analyze_and_script_text_only(voice_text: str) -> dict:
    """Text-only fused product analysis + pitch script via Cactus Gemma 4.

    The "blazingly fast" path. Skips the 15-18s vision pass on the product
    image entirely — uses the Deepgram-extracted seller narration in
    voice_text as the only input. One Cactus call, ~2-3s on Apple Neural
    Engine vs ~18s for the vision-fused variant.

    Mirrors swarmsell's intake pattern: Deepgram → text → SLM → structured
    output. Vision is reserved for the photo upload path (no audio
    transcript) where there's literally nothing else for Gemma to chew on.
    For video uploads with Deepgram narration ("Is a black sports watch.
    We do same day shipping. I want you to sell..."), Gemma can infer
    every product field from the text and write a believable pitch
    without ever looking at the pixels.

    Trade-off: vision-derived fields (category, materials, visual_details)
    become educated text-based guesses. The audience never sees the
    `product_data` JSON directly — they hear the script + see the actual
    video frames in the carousel — so guess-quality matters less than
    pipeline latency. The router (classify_comment_gemma) consumes the
    name/category for live chat routing; "Black Sports Watch" → "watches"
    is enough to route correctly.

    Returns the same shape as analyze_and_script_gemma:
      {product, script, source, latency_ms, cost}

    On parse failure / Cactus error, returns {"source": "gemma_failed"}
    so the caller can cascade to the vision path (or to Claude cloud).
    """
    logger.info("[GEMMA-text] analyze_and_script_text_only called "
                "(voice_text: %d chars, cactus: %s)",
                len(voice_text), CACTUS_AVAILABLE)

    if not CACTUS_AVAILABLE:
        return {"source": "gemma_failed", "reason": "cactus_unavailable"}

    prompt = f"""The seller said: "{voice_text}"

Return EXACTLY this JSON and nothing else — no preamble, no markdown fences, no trailing commentary:
{{"product":{{"name":"...","category":"...","materials":["..."],"selling_points":["...","...","..."],"visual_details":["...","..."]}},"script":"..."}}

Rules:
- "name": short product name based on what the seller described.
- "category": one of: clothing, accessories, electronics, home, fitness, beauty, toys, sports, other.
- "materials", "selling_points", "visual_details": educated guesses derived from the seller's description (NOT vision — text only).
- "script": ONE paragraph, under 70 words, second-person, TikTok Live energy. Mention something the seller said. End with a call to action. No stage directions, no quotation marks inside the script, no line breaks."""

    r = await asyncio.to_thread(
        _cactus_chat,
        [{"role": "user", "content": prompt}],
        512,
        None,           # NO image — text-only is the entire point
    )
    text = (r.get("response") or "").strip()
    latency = int(r.get("total_time_ms", 0))

    if r.get("error"):
        logger.warning("[GEMMA-text] call error: %s", r.get("error"))
        return {"source": "gemma_failed", "reason": str(r.get("error"))[:120],
                "latency_ms": latency}

    parsed = _parse_json_from_text(text)
    if not parsed:
        logger.info("[GEMMA-text] JSON unparseable (latency %dms); raw: %s",
                    latency, text[:300])
        return {"source": "gemma_failed", "reason": "unparseable_json",
                "latency_ms": latency}

    product = parsed.get("product") if isinstance(parsed.get("product"), dict) else None
    script_raw = parsed.get("script")
    script = script_raw.strip() if isinstance(script_raw, str) else ""
    if not product or not script:
        logger.info("[GEMMA-text] parsed but missing fields "
                    "(product=%s, script_len=%d)", bool(product), len(script))
        return {"source": "gemma_failed", "reason": "missing_fields",
                "latency_ms": latency}

    logger.info("[GEMMA-text] OK in %dms (1 text-only call) — "
                "%dx faster than vision path", latency,
                max(1, 18000 // max(latency, 1)))
    return {
        "product": product,
        "script": script,
        "source": "cactus_on_device",
        "latency_ms": latency,
        "cost": "$0.00",
    }


async def analyze_and_script_gemma(frame_b64: str, voice_text: str) -> dict:
    """On-device product analysis + pitch script via Cactus Gemma 4.

    Matches the return shape of `analyze_and_script_claude`: `{product,
    script}`. Two code paths share the same image on disk:

      - FUSED (default): one Gemma call returns {"product":{...},
        "script":"..."} — one prefill, one decode. ~35-45% faster than
        split when the model produces parseable JSON.
      - SPLIT (fallback): two sequential calls (product JSON, then
        pitch script). Proven reliable on Gemma 4 E4B when fused JSON
        gets truncated or malformed. Kicks in automatically if the
        fused response doesn't parse.

    Toggle with `EMPIRE_GEMMA_FUSED=0` to force split for A/B comparison
    without redeploy. Returns `{"source": "gemma_failed", ...}` if
    Cactus is unavailable or both paths fail — caller at main.py then
    falls back to Claude cloud.
    """
    logger.info("[GEMMA] analyze_and_script_gemma called (frame: %d chars, cactus: %s)",
                len(frame_b64), CACTUS_AVAILABLE)

    if not CACTUS_AVAILABLE:
        return {"source": "gemma_failed", "reason": "cactus_unavailable"}

    # Write frame to temp JPEG — Cactus wants an image path on disk.
    # Both paths reuse the same path so Cactus's per-source vision
    # cache can warm up if the fused path falls through to split.
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(base64.b64decode(frame_b64))
        img_path = f.name

    try:
        if os.getenv("EMPIRE_GEMMA_FUSED", "1") == "1":
            fused = await _analyze_and_script_fused(voice_text, img_path)
            if fused is not None:
                return fused
            logger.warning("[GEMMA] fused path returned unusable result; falling back to split")
        return await _analyze_and_script_split(voice_text, img_path)
    finally:
        try:
            os.unlink(img_path)
        except Exception:
            pass


async def _analyze_and_script_fused(voice_text: str, img_path: str) -> dict | None:
    """Single Gemma call returning product JSON + pitch in one response.

    Returns the full analyze_and_script result dict on success, or None
    if the response didn't parse cleanly or was missing required fields.
    Caller should fall back to the split-call path on None.
    """
    prompt = f"""Look at this product image. The seller said: "{voice_text}"

Return EXACTLY this JSON and nothing else — no preamble, no markdown fences, no trailing commentary:
{{"product":{{"name":"...","category":"...","materials":["..."],"selling_points":["...","...","..."],"visual_details":["...","..."]}},"script":"..."}}

Rules for "script": ONE paragraph, under 70 words, second-person, TikTok Live energy. Mention one visual detail AND one thing the seller said. End with a call to action. No stage directions, no quotation marks inside the script, no line breaks."""

    r = await asyncio.to_thread(
        _cactus_chat,
        [{"role": "user", "content": prompt}],
        512,           # product JSON + 70-word script + scaffolding
        [img_path],
    )
    text = (r.get("response") or "").strip()
    latency = int(r.get("total_time_ms", 0))

    if r.get("error"):
        logger.warning("[GEMMA] fused call error: %s", r.get("error"))
        return None

    parsed = _parse_json_from_text(text)
    if not parsed:
        logger.info("[GEMMA] fused JSON unparseable (latency %dms); raw: %s",
                    latency, text[:300])
        return None

    product = parsed.get("product") if isinstance(parsed.get("product"), dict) else None
    script_raw = parsed.get("script")
    script = script_raw.strip() if isinstance(script_raw, str) else ""
    if not product or not script:
        logger.info("[GEMMA] fused parsed but missing fields (product=%s, script_len=%d)",
                    bool(product), len(script))
        return None

    logger.info("[GEMMA] fused OK in %dms (1 call)", latency)
    return {
        "product": product,
        "script": script,
        "source": "cactus_on_device",
        "latency_ms": latency,
        "cost": "$0.00",
    }


async def _analyze_and_script_split(voice_text: str, img_path: str) -> dict:
    """Two-call path: product JSON then pitch script. Preserved as the
    safety net for when the fused prompt doesn't yield parseable JSON.
    """
    total_ms = 0

    # ── Call 1: product analysis (short, structured, image-grounded) ──
    product_prompt = f"""Look at this product image. The seller said: "{voice_text}"

Give a brief product analysis as JSON only:
{{"name": "...", "category": "...", "materials": ["..."], "selling_points": ["...", "...", "..."], "visual_details": ["...", "..."]}}"""

    r1 = await asyncio.to_thread(
        _cactus_chat,
        [{"role": "user", "content": product_prompt}],
        320,           # tighter — one JSON object, no script
        [img_path],
    )
    text1 = (r1.get("response") or "").strip()
    total_ms += int(r1.get("total_time_ms", 0))
    if r1.get("error"):
        return {"source": "gemma_failed", "reason": f"product_call: {r1.get('error')}"}

    product = _parse_json_from_text(text1) or {}
    # If parsing failed, synthesize a minimal product from voice_text
    # rather than giving up — the script call can still produce a
    # decent pitch from the narration alone.
    if not product:
        logger.warning("[GEMMA] product JSON unparseable; falling back. raw: %s",
                       text1[:200])
        product = {
            "name": (voice_text[:60] or "Product"),
            "selling_points": [],
            "visual_details": [],
        }

    # ── Call 2: spoken pitch script (plain text, given the product) ──
    # Feed Gemma the structured fields it just produced so the script
    # is grounded in them — avoids hallucination from re-looking at
    # the image.
    name = product.get("name", "product")
    vd = ", ".join((product.get("visual_details") or [])[:2]) or "what you can see"
    sp = (product.get("selling_points") or [])
    sp_text = "; ".join(str(s) for s in sp[:3]) or "quality build"

    script_prompt = f"""Write a short livestream sales pitch an avatar will read aloud.

Product: {name}
Seller said: "{voice_text}"
Visual details: {vd}
Selling points: {sp_text}

Write ONE paragraph under 70 words. Second-person, conversational, TikTok Live energy. Mention one visual detail AND one thing the seller said. End with a call to action. Output the script only — no JSON, no stage directions, no preamble."""

    r2 = await asyncio.to_thread(
        _cactus_chat,
        [{"role": "user", "content": script_prompt}],
        200,           # tight — ~70 words ≈ 100 tokens, give headroom
        None,          # no image needed, second call is text-only
    )
    text2 = (r2.get("response") or "").strip()
    total_ms += int(r2.get("total_time_ms", 0))

    # Strip common "here's the pitch:" preambles if Gemma added them.
    for prefix in ("Here's the pitch:", "Pitch:", "Script:", "Here's a script:"):
        if text2.lower().startswith(prefix.lower()):
            text2 = text2[len(prefix):].strip()

    if not text2:
        # Build a fallback script from the product fields so the
        # avatar still has something to say.
        text2 = (f"Check out this {name}. "
                 f"{sp_text}. Tap the basket.").strip()

    logger.info("[GEMMA] split OK in %dms (2 calls)", total_ms)
    return {
        "product": product,
        "script": text2,
        "source": "cactus_on_device",
        "latency_ms": total_ms,
        "cost": "$0.00",
    }


async def analyze_and_script_claude(frame_b64: str, voice_text: str) -> dict:
    """Single Claude call: product analysis + sales script from image.
    Returns dict with product_data and sales_script.

    The script is structured as a 4-beat livestream pitch (HOOK → DEMO →
    PROOF → CTA) so it sounds like a TikTok Live creator, not generic
    marketing copy. Hallucination guard: must reference at least 2 visual
    details from the image AND 1 specific phrase from the seller's voice.
    """
    logger.info("[CLAUDE] analyze_and_script called (frame: %d chars)", len(frame_b64))

    # Decide structure based on whether the seller actually said anything
    # meaningful. The judge-item demo will often have voice_text="sell this"
    # (the default when an item is uploaded with no spoken context). In that
    # case PROOF can't legitimately quote the seller — we drop that beat
    # rather than force Claude to fabricate one.
    voice_clean = (voice_text or "").strip()
    voice_meaningful = len(voice_clean) > 12 and voice_clean.lower() not in {
        "sell this", "sell this thing", "sell this product", "sell it"
    }

    if voice_meaningful:
        voice_block = f"""SELLER'S VOICE (CONTEXT — trust this for product identity, anything the seller
explicitly said is true; do NOT contradict them):
\"\"\"
{voice_clean}
\"\"\""""
        script_beats = """A 25-30 second spoken pitch with FOUR beats: HOOK (1 sentence — get
attention with a specific visual detail), DEMO (1-2 sentences — name 2 more visual
details so the audience knows you're really looking at it), PROOF (1-2
sentences — quote or paraphrase 1 specific phrase from the seller's voice
above), CTA (1 sentence — call to action with urgency)."""
    else:
        # Judge-item / no-voice path: lean entirely on what's visible.
        voice_block = """SELLER'S VOICE: (none — the seller dropped this in without commentary.
Do NOT pretend they said anything. Carry the pitch on your own as a
livestream host who just saw the item for the first time.)"""
        script_beats = """A 25-30 second spoken pitch with FOUR beats: HOOK (1 sentence — react to a
specific visual detail like you just saw it: "okay no wait"…, "guys, look at
this"…), DEMO (1-2 sentences — name 2-3 more visual details: color, material,
form factor, vibe), VIBE (1-2 sentences — who'd love this and when they'd
use it, drawn from what you can see), CTA (1 sentence — playful call to
action with urgency)."""

    prompt = f"""You are writing copy for a live e-commerce stream where an AI avatar
will read the script aloud in real time. Treat this like TikTok Live, not Amazon
copy — punchy, conversational, second-person.

{voice_block}

VISUAL: study the product image carefully for color, materials, condition,
accessories, finish, scale, and any notable details visible in frame.

Return ONLY valid JSON:
{{
    "product": {{
        "name": "exact product name (use seller's words if given, otherwise infer from visual)",
        "category": "broad category — e.g. watches, sneakers, headphones, drinkware",
        "materials": ["primary material", "secondary material"],
        "selling_points": ["5 short benefit phrases the script will draw from"],
        "target_audience": "one-line buyer persona",
        "suggested_price_range": "$X - $Y"
    }},
    "script": "{script_beats} No stage directions, no '[pause]', no headings — just the spoken words run together as one paragraph. Under 90 words total. Conversational, second person ('you'), genuine enthusiasm. Do NOT start with 'Hi everyone' or 'Welcome back' — open with the hook."
}}

HALLUCINATION GUARD: every sentence in the script must be supported by either
a visual detail YOU CAN SEE in the image or a specific phrase YOU CAN QUOTE
from the seller's voice. Do not invent features, dimensions, prices, materials,
specs, or brand names that aren't in either source. If you don't know a detail,
omit it — don't bluff."""

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


# ── VOICE: Transcription (whisper-tiny on Cactus → Gemini 2.5 Flash cloud) ──
#
# Primary: whisper-tiny loaded as a second Cactus handle (62 MB). Uses
# `cactus_transcribe` with the Whisper prompt template. ~200-400ms on M-series.
#
# Fallback: Gemini 2.5 Flash via google-genai. Hackathon credits cover this.
#
# Both return the same shape: {text, source, latency_ms}. The voice endpoint
# in main.py broadcasts `voice_transcript` with this shape unchanged.


def _get_cactus_whisper_model():
    """Lazy-load whisper-tiny as a separate Cactus model handle. Kept
    distinct from `_cactus_model` (Gemma 4) so neither blocks the other.
    Returns None if weights aren't present — caller falls through to cloud."""
    global _cactus_whisper_model
    if _cactus_whisper_model is not None:
        return _cactus_whisper_model
    if not CACTUS_AVAILABLE:
        return None
    if not os.path.isdir(CACTUS_WHISPER_WEIGHTS_PATH):
        logger.info("Whisper weights not found at %s — voice will use cloud", CACTUS_WHISPER_WEIGHTS_PATH)
        return None
    try:
        _cactus_whisper_model = cactus_init(
            CACTUS_WHISPER_WEIGHTS_PATH.encode() if isinstance(CACTUS_WHISPER_WEIGHTS_PATH, str) else CACTUS_WHISPER_WEIGHTS_PATH,
            None, False,
        )
        logger.info("Cactus whisper model loaded from %s", CACTUS_WHISPER_WEIGHTS_PATH)
    except Exception as e:
        logger.warning("Cactus whisper init failed: %s", e)
        _cactus_whisper_model = None
    return _cactus_whisper_model


class AudioDecodeError(RuntimeError):
    """Raised when ffmpeg can't decode the uploaded bytes as audio."""


# Filler utterances that whisper / Gemini commonly emit for silent or
# near-silent inputs. We drop these rather than feeding them to the cloud
# comment pipeline — silence is not a comment.
_NOISE_TRANSCRIPTS = {
    "", "mhm", "uh", "um", "ah", "oh", "hm", "huh", "hmm", "uhh", "umm",
    "you", "thank you", "thanks", ".", "...", "[silence]", "[music]",
    "[inaudible]",
}


def _is_noise_transcript(text: str) -> bool:
    """Heuristic: treat trivially short / filler transcripts as silence.
    Prevents hallucinated responses to empty mic input."""
    t = (text or "").strip().lower().rstrip(".!?")
    if len(t) < 3:
        return True
    return t in _NOISE_TRANSCRIPTS


def _to_wav_16k_mono(audio_bytes: bytes) -> bytes:
    """Convert any ffmpeg-decodable audio (webm/ogg/mp3/wav) to 16 kHz mono
    PCM WAV — the format whisper expects. Returns WAV bytes.

    Raises AudioDecodeError (with stderr snippet) if ffmpeg can't decode.
    Runs synchronously; call via asyncio.to_thread from async contexts."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as in_f:
        in_f.write(audio_bytes)
        in_path = in_f.name
    out_path = in_path + ".wav"
    try:
        proc = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", in_path,
             "-ar", "16000", "-ac", "1", "-f", "wav", out_path],
            capture_output=True,
        )
        if proc.returncode != 0:
            # Sanitize: no temp paths or full argv in user-facing error.
            stderr_snippet = (proc.stderr or b"").decode("utf-8", "ignore")[:200].strip()
            raise AudioDecodeError(
                f"ffmpeg could not decode audio (rc={proc.returncode}): {stderr_snippet}"
            )
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in (in_path, out_path):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass


async def _cactus_transcribe_audio(audio_bytes: bytes) -> dict:
    """On-device transcription via whisper on Cactus. Returns
    {text, source: 'cactus_on_device', latency_ms}. Raises on error so
    the caller can fall through to Gemini.

    Serialized on _cactus_whisper_lock because the Cactus C library is not
    re-entrant on a single handle — concurrent calls crash the process."""
    model = _get_cactus_whisper_model()
    if model is None:
        raise RuntimeError("Cactus whisper model not available")

    wav_bytes = await asyncio.to_thread(_to_wav_16k_mono, audio_bytes)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        wav_path = f.name

    try:
        t0 = time.time()
        prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"

        def _locked_transcribe():
            with _cactus_whisper_lock:
                return cactus_transcribe(model, wav_path, prompt, None, None, None)

        raw = await asyncio.to_thread(_locked_transcribe)
        latency_ms = int((time.time() - t0) * 1000)
        try:
            result = json.loads(raw)
            segments = result.get("segments") or []
            text = " ".join((s.get("text") or "").strip() for s in segments).strip()
        except json.JSONDecodeError:
            # Some builds return bare text — accept it.
            text = (raw or "").strip()
        return {"text": text, "source": "cactus_on_device", "latency_ms": latency_ms}
    finally:
        try:
            os.unlink(wav_path)
        except FileNotFoundError:
            pass


async def _gemini_transcribe(audio_bytes: bytes) -> dict:
    """Cloud fallback: Gemini 2.5 Flash. Uses the `google-genai` SDK.
    Returns {text, source: 'gemini_cloud', latency_ms}."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    wav_bytes = await asyncio.to_thread(_to_wav_16k_mono, audio_bytes)

    def _sync_call():
        # Imported lazily so the backend starts even if google-genai isn't
        # installed in a given environment.
        from google import genai
        from google.genai import types as genai_types
        client = genai.Client(api_key=api_key)
        audio_part = genai_types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")
        resp = client.models.generate_content(
            model=model_name,
            contents=[
                "Transcribe this audio verbatim. Return ONLY the transcribed "
                "text, no quotes, no commentary, no labels.",
                audio_part,
            ],
        )
        return (resp.text or "").strip()

    t0 = time.time()
    text = await asyncio.to_thread(_sync_call)
    latency_ms = int((time.time() - t0) * 1000)
    return {"text": text, "source": "gemini_cloud", "latency_ms": latency_ms}


async def transcribe_voice(audio_bytes: bytes) -> dict:
    """Public entry point. Tries on-device Cactus whisper first; falls back
    to Gemini 2.5 Flash. Always returns a dict with {text, source,
    latency_ms}. On total failure, sets source='transcription_failed' with
    a sanitized error string rather than raising. If the decoded speech is
    silence / filler (see _is_noise_transcript), returns empty text with
    source='no_speech' so the voice endpoint can bail cleanly."""
    # Decode errors are typically user-audio problems, not service outages;
    # surface a clean error rather than falling through to Gemini on bad
    # bytes (Gemini will also fail, but noisily and after a network RTT).
    cactus_err: str | None = None
    try:
        result = await _cactus_transcribe_audio(audio_bytes)
        text = (result.get("text") or "").strip()
        if _is_noise_transcript(text):
            return {"text": "", "source": "no_speech",
                    "latency_ms": result.get("latency_ms", 0),
                    "reason": f"heard silence/filler (cactus: {text!r})"}
        return result
    except AudioDecodeError:
        return {"text": "", "source": "transcription_failed",
                "latency_ms": 0, "error": "bad_audio"}
    except Exception as e:
        cactus_err = str(e)
        logger.warning("Cactus transcription failed: %s — falling back to Gemini", e)

    try:
        result = await _gemini_transcribe(audio_bytes)
        text = (result.get("text") or "").strip()
        if _is_noise_transcript(text):
            return {"text": "", "source": "no_speech",
                    "latency_ms": result.get("latency_ms", 0),
                    "reason": f"heard silence/filler (gemini: {text!r})"}
        return result
    except AudioDecodeError:
        return {"text": "", "source": "transcription_failed",
                "latency_ms": 0, "error": "bad_audio"}
    except Exception:
        logger.exception("Gemini transcription failed")
        return {
            "text": "",
            "source": "transcription_failed",
            "latency_ms": 0,
            "error": "transcription_unavailable",
        }
