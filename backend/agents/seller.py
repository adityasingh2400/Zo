import json
import base64
import asyncio
import logging
import time
import boto3
import httpx
from elevenlabs import ElevenLabs
from pathlib import Path
from config import (
    AWS_REGION, BEDROCK_MODEL_ID,
    ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID,
    RUNPOD_POD_IP, RUNPOD_LIVETALKING_PORT,
    WAV2LIP_URL, LATENTSYNC_URL, POD_SPEAKING_1080P,
)

logger = logging.getLogger("empire.seller")
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
eleven = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None
logger.info("ElevenLabs: %s", "configured" if eleven else "NOT configured (no API key)")
logger.info("RunPod: %s", f"{RUNPOD_POD_IP}:{RUNPOD_LIVETALKING_PORT}" if RUNPOD_POD_IP else "NOT configured")

# LiveTalking session ID, set after WebRTC handshake
_livetalking_session_id = None


def set_livetalking_session(session_id: str):
    global _livetalking_session_id
    _livetalking_session_id = session_id


def get_livetalking_url():
    return f"http://{RUNPOD_POD_IP}:{RUNPOD_LIVETALKING_PORT}"


async def generate_sales_script(product_data: dict, voice_text: str) -> str:
    """Generate a 30-second sales pitch from product data via Claude on Bedrock."""
    logger.info("[SCRIPT] Generating sales pitch from product data...")
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 256,
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

    import asyncio
    response = await asyncio.to_thread(
        bedrock.invoke_model,
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
    """Generate a natural response to a viewer comment via Claude on Bedrock."""
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 60,
        "messages": [{
            "role": "user",
            "content": f"""You are an AI sales avatar on a livestream.
Viewer comment: "{comment}"
Product: {json.dumps(product_data)[:400]}

Reply in ONE short sentence (max 15 words). Spoken dialogue only.
No preamble, no stage directions, no hedging like "Great question". Start with the answer.""",
        }],
    })

    response = await asyncio.to_thread(
        bedrock.invoke_model,
        modelId=BEDROCK_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


async def make_avatar_speak(text: str, interrupt: bool = True) -> dict:
    """Send text to LiveTalking. It handles TTS + lip sync + video streaming internally.
    The video comes back through the already-open WebRTC connection."""
    logger.info("[AVATAR] make_avatar_speak called (text: %d chars, session: %s)",
                len(text), _livetalking_session_id[:20] if _livetalking_session_id else "NONE")
    if not RUNPOD_POD_IP:
        return {"error": "RunPod not configured. Set RUNPOD_POD_IP in .env"}

    if not _livetalking_session_id:
        return {"error": "No LiveTalking session. Dashboard must connect WebRTC first."}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{get_livetalking_url()}/human",
                json={
                    "sessionid": _livetalking_session_id,
                    "text": text,
                    "type": "echo",
                    "interrupt": interrupt,
                },
            )
            resp.raise_for_status()
            return {"status": "speaking", "text": text}
    except Exception as e:
        return {"error": str(e)}


# ── Lip-sync clients (RunPod) ──────────────────────────────────────────────

async def _post_lipsync(
    url: str,
    audio_bytes: bytes,
    audio_mime: str = "audio/mpeg",
    source_path_on_pod: str | None = None,
    source_field: str = "video",
    extra_data: dict | None = None,
    timeout: float = 900.0,
) -> tuple[bytes, dict]:
    """POST audio (+ reference to a source video already on the pod) to a lip-sync
    server. We send the source video by reference if possible; otherwise expect
    the server to accept a file upload at `source_field`."""
    files = {"audio": ("audio.mp3", audio_bytes, audio_mime)}
    data = dict(extra_data or {})
    if source_path_on_pod:
        # Try server-side path first (fast path for both our servers)
        data["source_path"] = source_path_on_pod
    # Our Wav2Lip/LatentSync servers also accept direct uploads; if a source_path
    # is already resolved server-side they'll skip the upload read.
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{url}/lipsync", data=data, files=files)
        r.raise_for_status()
        return r.content, dict(r.headers)


async def render_comment_response_wav2lip(
    audio_bytes: bytes,
    source_path_on_pod: str = POD_SPEAKING_1080P,
    out_height: int = 1920,
) -> tuple[bytes, dict]:
    """FAST path: Wav2Lip on the pod. Target p50 warm ~6-10s for a ~5s response.
    Used for LIVE comment responses where sub-8s matters.
    Output is rendered at the source video's native 1920px height (1080x1920 9:16)
    so it visually matches the Tier 0 idle clips on the dashboard cinema stage.
    Earlier we shipped at out_height=1080, which downscaled the 1920-tall source
    to 606x1080; that read as a noticeable resolution drop when the response
    crossfaded in over the native idle layer. Native height fixes that.
    Face-detect cache is keyed on out_height so the warm path is unchanged.
    Returns (mp4_bytes, timing_headers)."""
    logger.info("[LIPSYNC] Wav2Lip /lipsync_fast — audio=%dB source=%s", len(audio_bytes), source_path_on_pod)
    t0 = time.perf_counter()
    # Use /lipsync_fast: source video is already on the pod; only upload audio.
    # Saves ~1MB upload per call (~500ms-1s) and avoids reading the source on the client.
    files = {"audio": ("audio.mp3", audio_bytes, "audio/mpeg")}
    data = {"source_path": source_path_on_pod, "out_height": str(out_height)}
    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.post(f"{WAV2LIP_URL}/lipsync_fast", data=data, files=files)
        r.raise_for_status()
        content = r.content
        headers = dict(r.headers)
    elapsed = time.perf_counter() - t0
    logger.info("[LIPSYNC] Wav2Lip done in %.2fs, %d bytes", elapsed, len(content))
    return content, headers


async def render_pitch_latentsync(
    audio_bytes: bytes,
    source_path_on_pod: str = POD_SPEAKING_1080P,
    inference_steps: int = 10,
    guidance_scale: float = 1.5,
    out_height: int = 1080,
) -> tuple[bytes, dict]:
    """HIGH-QUALITY path: LatentSync. Target 6-8 min per 10s pitch.
    Used once per product to render the main pitch video (non-live)."""
    logger.info("[LIPSYNC] LatentSync request — audio=%dB steps=%d cfg=%.1f", len(audio_bytes), inference_steps, guidance_scale)
    t0 = time.perf_counter()
    with open_pod_video(source_path_on_pod) as src_bytes:
        files = {
            "source_video": ("src.mp4", src_bytes, "video/mp4"),
            "audio": ("audio.mp3", audio_bytes, "audio/mpeg"),
        }
        data = {
            "inference_steps": str(inference_steps),
            "guidance_scale": str(guidance_scale),
            "enable_deepcache": "1",
            "out_height": str(out_height),
        }
        async with httpx.AsyncClient(timeout=1200.0) as client:
            r = await client.post(f"{LATENTSYNC_URL}/lipsync", data=data, files=files)
            r.raise_for_status()
            content = r.content
            headers = dict(r.headers)
    elapsed = time.perf_counter() - t0
    logger.info("[LIPSYNC] LatentSync done in %.2fs, %d bytes", elapsed, len(content))
    return content, headers


# Local cache of the source speaking video. The pod has it, but Wav2Lip v2 expects
# an upload each call. Cache once locally for fast reuse.
_SOURCE_CACHE: dict[str, bytes] = {}


def open_pod_video(pod_path: str):
    """Return a context-manager yielding bytes of the source video.
    Looks for the video locally first (phase0/assets/states), else streams
    the cached copy."""
    local_candidates = [
        Path(__file__).resolve().parents[2] / "phase0" / "assets" / "states" / Path(pod_path).name,
        Path(__file__).resolve().parents[2] / Path(pod_path).name,
    ]
    for p in local_candidates:
        if p.exists():
            if str(p) not in _SOURCE_CACHE:
                _SOURCE_CACHE[str(p)] = p.read_bytes()
            return _BytesCtx(_SOURCE_CACHE[str(p)])
    raise FileNotFoundError(
        f"Source video not found locally. Tried: {local_candidates}. "
        f"Drop a copy at one of these paths (the pod has it at {pod_path})."
    )


class _BytesCtx:
    def __init__(self, data: bytes):
        self.data = data
    def __enter__(self):
        import io as _io
        self._buf = _io.BytesIO(self.data)
        return self._buf
    def __exit__(self, *a):
        self._buf.close()


async def text_to_speech(text: str) -> bytes:
    """ElevenLabs TTS fallback. Used when LiveTalking is unavailable."""
    logger.info("[TTS] text_to_speech called (text: %d chars, eleven: %s)", len(text), "yes" if eleven else "no")
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
