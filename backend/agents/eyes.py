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

async def analyze_and_script_gemma(frame_b64: str, voice_text: str) -> dict:
    """On-device product analysis + pitch script via Cactus Gemma 4.

    Matches the return shape of `analyze_and_script_claude`: `{product,
    script}`. Gemma 4 E4B chokes on one big nested-JSON prompt (hits a
    stop token around 300 chars) so we split into TWO sequential calls:

      1. Product analysis — JSON only, short + structured
      2. Script — plain text, given the product fields as context

    This halves the tokens per call and gets reliable completion out of
    the small model. Total latency is ~2x one call (Cactus lock serializes
    them), but each stays under ~1kb so both complete cleanly.

    Fallback: if Cactus is unavailable or either call fails badly,
    returns `{"source": "gemma_failed", ...}` so run_sell_pipeline can
    fall back to Claude.
    """
    logger.info("[GEMMA] analyze_and_script_gemma called (frame: %d chars, cactus: %s)",
                len(frame_b64), CACTUS_AVAILABLE)

    if not CACTUS_AVAILABLE:
        return {"source": "gemma_failed", "reason": "cactus_unavailable"}

    # Write frame to temp JPEG — Cactus wants an image path on disk. We
    # reuse the same path for both calls so Cactus's per-source face/
    # vision cache can warm up between them.
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(base64.b64decode(frame_b64))
        img_path = f.name

    try:
        import asyncio
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

        return {
            "product": product,
            "script": text2,
            "source": "cactus_on_device",
            "latency_ms": total_ms,
            "cost": "$0.00",
        }
    finally:
        try:
            os.unlink(img_path)
        except Exception:
            pass


async def analyze_and_script_claude(frame_b64: str, voice_text: str) -> dict:
    """Single Claude call: product analysis + sales script from image.
    Returns dict with product_data and sales_script.

    The script is structured as a 4-beat livestream pitch (HOOK → DEMO →
    PROOF → CTA) so it sounds like a TikTok Live creator, not generic
    marketing copy. Hallucination guard: must reference at least 2 visual
    details from the image AND 1 specific phrase from the seller's voice.
    """
    logger.info("[CLAUDE] analyze_and_script called (frame: %d chars)", len(frame_b64))
    prompt = f"""You are writing copy for a live e-commerce stream where an AI avatar
will read the script aloud in real time. Treat this like TikTok Live, not Amazon
copy — punchy, conversational, second-person.

SELLER'S VOICE (CONTEXT — trust this for product identity, anything the seller
explicitly said is true; do NOT contradict them):
\"\"\"
{voice_text}
\"\"\"

VISUAL: study the product image for color, materials, condition, accessories,
notable details visible in frame.

Return ONLY valid JSON:
{{
    "product": {{
        "name": "exact product name (from seller's words + visual confirmation)",
        "category": "broad category — e.g. watches, sneakers, headphones",
        "materials": ["primary material", "secondary material"],
        "selling_points": ["5 short benefit phrases the script will draw from"],
        "target_audience": "one-line buyer persona",
        "suggested_price_range": "$X - $Y"
    }},
    "script": "A 30-second spoken pitch with FOUR beats: HOOK (1 sentence — get
attention), DEMO (1-2 sentences — name 2 specific visual details from the image
so the audience knows you're really looking at the product), PROOF (1-2
sentences — quote or paraphrase 1 specific phrase from the seller's voice
above), CTA (1 sentence — call to action with urgency). No stage directions,
no '[pause]', no headings — just the spoken words run together as one paragraph.
Under 90 words total. Conversational, second person ('you'), genuine
enthusiasm."
}}

HALLUCINATION GUARD: every sentence in the script must be supported by either
a visual detail YOU CAN SEE in the image or a specific phrase YOU CAN QUOTE
from the seller's voice. Do not invent features, dimensions, prices, or
specs that aren't in either source."""

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
