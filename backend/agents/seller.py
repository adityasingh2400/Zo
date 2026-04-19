import json
import base64
import asyncio
import logging
import re
import shutil
import subprocess
import tempfile
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


def _eleven_tts_sync(
    text: str,
    voice_id: str | None = None,
    *,
    model_id: str | None = None,
) -> bytes:
    """Sync ElevenLabs TTS call. Wrapped via asyncio.to_thread so the
    event loop isn't blocked while audio bytes are being generated +
    streamed; the WS broadcast for play_clip events can interleave.

    Default model is eleven_flash_v2_5 (~75ms inference, ~400ms TTFB).
    That's the right pick for the live escalate path where every ms of
    latency between mic-release and audio-out is felt by the viewer.

    Pass `model_id="eleven_v3"` for pre-rendered tiers (bridge clips,
    pre-rendered Q&A, pitch). v3 has >1s inference but honours inline
    audio tags ([curious], [pauses], [laughs softly], [excited], etc.)
    that make the voice read as a person reacting, not a chatbot. The
    extra latency is paid offline at render time, so viewers never feel
    it. See backend/agents/bridge_clips.BRIDGE_SCRIPTS for tag usage.

    Per-model parameter shaping:
      - language_code='en' locks Flash to English. Flash is multilingual
        and auto-detects, but edge cases (mostly-numeric scripts, brand
        names, ASCII art in product descriptions) flip detection and
        produce phonetically-warped output that reads as gibberish.
      - v3 deliberately drops language_code so the model can interpret
        accent tags ([British accent], etc.) and multi-language inserts
        without a hard lock fighting the script.
      - v3 outputs at 192kbps (Creator+ tier; we have the grant) since
        the render is already slow and the bitrate cost is invisible
        on disk.
    """
    chosen_model = model_id or "eleven_flash_v2_5"
    kwargs: dict = {
        "text": text,
        "voice_id": voice_id or ELEVENLABS_VOICE_ID,
        "model_id": chosen_model,
    }
    if chosen_model == "eleven_v3":
        kwargs["output_format"] = "mp3_44100_192"
    else:
        kwargs["output_format"] = "mp3_44100_128"
        kwargs["language_code"] = "en"
    audio_gen = eleven.text_to_speech.convert(**kwargs)
    return b"".join(audio_gen)


def _eleven_tts_with_timestamps_sync(
    text: str,
    voice_id: str | None = None,
    *,
    model_id: str | None = None,
) -> tuple[bytes, list[dict]]:
    """ElevenLabs TTS via the /with-timestamps endpoint. Returns
    (audio_bytes, word_timings) where word_timings is the same shape
    synthesize_word_timings produces — [{word, start, end}, ...] in
    seconds — but derived from the real per-character alignment the
    API returns alongside the audio.

    Aggregation: walk the character stream, collect runs of non-whitespace
    chars into words, take word.start = first char start, word.end =
    last char end. Punctuation stays glued to the preceding word so
    karaoke renders 'forty-nine,' as one highlighted unit.

    Same model + output format + language lock as _eleven_tts_sync so
    the audio character is identical to the legacy path; only the API
    surface changes. Latency is ~50-100ms higher than convert() because
    the response is a single JSON blob (not a streaming generator) — the
    caller pays this cost only when timings are needed (return_word_timings=True).

    Raises on any API/network error so the caller can fall back to
    convert() + synthesize_word_timings cleanly.
    """
    chosen_model = model_id or "eleven_flash_v2_5"
    kwargs: dict = {
        "text": text,
        "voice_id": voice_id or ELEVENLABS_VOICE_ID,
        "model_id": chosen_model,
    }
    if chosen_model == "eleven_v3":
        kwargs["output_format"] = "mp3_44100_192"
    else:
        kwargs["output_format"] = "mp3_44100_128"
        kwargs["language_code"] = "en"

    resp = eleven.text_to_speech.convert_with_timestamps(**kwargs)
    audio_bytes = base64.b64decode(resp.audio_base_64) if resp.audio_base_64 else b""

    alignment = getattr(resp, "alignment", None)
    if (not alignment
            or not alignment.characters
            or not alignment.character_start_times_seconds
            or not alignment.character_end_times_seconds):
        # API returned audio but no alignment — caller will handle by
        # falling back to synthesize_word_timings.
        return audio_bytes, []

    return audio_bytes, _aggregate_chars_to_words(
        alignment.characters,
        alignment.character_start_times_seconds,
        alignment.character_end_times_seconds,
    )


def _aggregate_chars_to_words(
    chars: list[str],
    starts: list[float],
    ends: list[float],
) -> list[dict]:
    """Walk a character timeline and group runs of non-whitespace chars
    into words. Whitespace is treated as a separator; punctuation
    glued to a word stays attached. Pure function for easy testing."""
    words: list[dict] = []
    cur_chars: list[str] = []
    cur_start: float | None = None
    cur_end: float | None = None

    for ch, start, end in zip(chars, starts, ends):
        if ch.isspace():
            if cur_chars and cur_start is not None and cur_end is not None:
                words.append({
                    "word": "".join(cur_chars),
                    "start": round(cur_start, 3),
                    "end": round(cur_end, 3),
                })
            cur_chars = []
            cur_start = None
            cur_end = None
            continue
        if cur_start is None:
            cur_start = start
        cur_chars.append(ch)
        cur_end = end

    if cur_chars and cur_start is not None and cur_end is not None:
        words.append({
            "word": "".join(cur_chars),
            "start": round(cur_start, 3),
            "end": round(cur_end, 3),
        })
    return words


# ── Audio duration probe ─────────────────────────────────────────────────────
# Synthetic word-timing generation needs to know how long the rendered MP3
# actually is. ffprobe is the most reliable source; if it's not installed
# we fall back to a heuristic of 12 chars/second of speech (close enough for
# the karaoke window to track). The fallback under-estimates long pauses but
# we re-sync the visible word every 500ms on the dashboard, so a few hundred
# ms of drift is invisible.
def _probe_audio_duration_ms(audio_bytes: bytes) -> int | None:
    """Returns the audio duration in milliseconds, or None if ffprobe
    isn't installed / the bytes don't decode."""
    if not audio_bytes:
        return None
    if not shutil.which("ffprobe"):
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error",
                 "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1",
                 tmp_path],
                capture_output=True, text=True, timeout=5,
            )
            secs = float(out.stdout.strip())
            return int(secs * 1000)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    except Exception as e:
        logger.debug("[tts] ffprobe failed: %s", e)
        return None


# ── Synthetic word timings ───────────────────────────────────────────────────
# Cartesia would give us exact per-word {start, end} for free. We're on
# ElevenLabs, which doesn't return timings. The synthetic version splits
# the text into words, measures each word's character length (a decent
# proxy for spoken duration), and distributes the audio duration in
# proportion. Plus a small constant for inter-word gap (~25ms) so trailing
# punctuation doesn't crowd the next word.
#
# Accuracy versus real per-phoneme timings is roughly 80-95% — good enough
# for the karaoke caption window, which slides through 8-12 words at a
# time. The dashboard re-syncs every frame from audioElement.currentTime,
# so any drift inside a word is invisible at the active-word boundary.
_WORD_SPLIT_RE = re.compile(r"\S+")
_INTER_WORD_GAP_MS = 25
# Speaking rate fallback (chars/sec) when we have no audio_duration_ms
# (degraded path: render_pitch_assets without ffprobe). 12 chars/sec is
# eleven_flash_v2_5's measured average over the demo's response set.
_FALLBACK_CHARS_PER_SEC = 12.0


def synthesize_word_timings(
    text: str,
    audio_duration_ms: int | None,
    *,
    leading_pad_ms: int = 60,
    trailing_pad_ms: int = 80,
) -> list[dict]:
    """Split `text` on whitespace and distribute `audio_duration_ms` across
    the words proportionally to character count. Returns a list of
    {word, start, end} dicts where start/end are in seconds (matching the
    Cartesia contract so the dashboard never has to branch on units).

    `leading_pad_ms` accounts for the typical 50-100ms intake breath
    before any speech audio starts. `trailing_pad_ms` reserves a small
    silence after the last word so the active-word highlight doesn't
    hang on the final word for a noticeable gap.

    If `audio_duration_ms` is None or non-positive, falls back to the
    chars-per-sec heuristic so the dashboard still gets a usable timing
    list (slightly drifty but works).
    """
    words = _WORD_SPLIT_RE.findall(text or "")
    if not words:
        return []

    total_chars = sum(max(1, len(w)) for w in words)
    gap_ms = _INTER_WORD_GAP_MS * (len(words) - 1)

    if audio_duration_ms and audio_duration_ms > 0:
        speech_ms = max(0, audio_duration_ms - leading_pad_ms - trailing_pad_ms - gap_ms)
        if speech_ms <= 0:
            # Audio is shorter than just the padding budget — pretend the
            # whole clip is one continuous run, no padding, even split.
            speech_ms = audio_duration_ms
            leading_pad_ms = 0
    else:
        # Heuristic estimate. Each word needs ~len(word)/chars_per_sec seconds.
        speech_ms = int(total_chars / _FALLBACK_CHARS_PER_SEC * 1000)

    out: list[dict] = []
    cursor_ms = leading_pad_ms
    for i, word in enumerate(words):
        word_share = max(1, len(word)) / total_chars
        word_dur_ms = max(80, int(speech_ms * word_share))  # 80ms floor
        start_ms = cursor_ms
        end_ms = cursor_ms + word_dur_ms
        out.append({
            "word": word,
            "start": round(start_ms / 1000, 3),
            "end": round(end_ms / 1000, 3),
        })
        cursor_ms = end_ms + (_INTER_WORD_GAP_MS if i < len(words) - 1 else 0)

    return out


async def text_to_speech(
    text: str,
    *,
    voice: str | None = None,
    model_id: str | None = None,
    return_word_timings: bool = False,
) -> bytes | tuple[bytes, list[dict]]:
    """ElevenLabs TTS. Default model flash_v2_5 = ~400ms TTFB for a
    15-word reply (live path). Off-loaded to a worker thread so it
    doesn't stall the asyncio loop.

    Default behaviour (`return_word_timings=False`) returns just the MP3
    bytes — same contract every existing caller relies on.

    With `return_word_timings=True` returns `(bytes, word_timings)` where
    word_timings is `[{word, start, end}, ...]` in seconds. Real per-word
    timings are pulled from the ElevenLabs `/with-timestamps` endpoint
    (per-character alignment aggregated to word boundaries) for ~10ms
    sync accuracy on karaoke captions. Falls back to the whitespace-split
    + ffprobe-duration synthesizer if the API errors or returns no
    alignment — the caller never sees the difference.

    `voice` overrides ELEVENLABS_VOICE_ID when set — used by
    bridge_clips.render_all to render a per-character voice without mutating
    the env.

    `model_id` overrides eleven_flash_v2_5. Pass "eleven_v3" for the
    pre-rendered expressive tier (audio tags honoured). See
    `_eleven_tts_sync` docstring for the full tiering rationale.
    """
    logger.info("[TTS] text_to_speech (text: %d chars, voice=%s, model=%s, timings=%s, eleven=%s)",
                len(text), voice or "default", model_id or "default(flash)",
                return_word_timings, "yes" if eleven else "no")
    if not eleven:
        if return_word_timings:
            return b"", []
        return b""

    if return_word_timings:
        # Real per-character timings via /with-timestamps. ~50-100ms
        # higher TTFB than convert() because the response is a single
        # JSON blob, not a streaming generator. Worth it for precise
        # karaoke sync; live path doesn't request timings unless karaoke
        # actually needs them.
        try:
            audio_bytes, word_timings = await asyncio.to_thread(
                _eleven_tts_with_timestamps_sync, text, voice, model_id=model_id,
            )
            if audio_bytes and word_timings:
                logger.info("[TTS] real timings: %d words from API alignment",
                            len(word_timings))
                return audio_bytes, word_timings
            # API returned audio but no alignment (rare, model-specific).
            # Fall through to synthesize against measured duration so
            # karaoke still tracks.
            if audio_bytes:
                duration_ms = await asyncio.to_thread(
                    _probe_audio_duration_ms, audio_bytes,
                )
                synth = synthesize_word_timings(text, duration_ms)
                logger.info("[TTS] API alignment empty; synthesized %d words "
                            "over %s ms", len(synth),
                            duration_ms if duration_ms else "(estimate)")
                return audio_bytes, synth
            return b"", []
        except Exception as e:
            logger.warning("[TTS] with_timestamps failed (%s) — falling back to "
                           "convert() + synthesize", e)
            # Fall through to the convert() path below, then synthesize.

    audio_bytes = await asyncio.to_thread(_eleven_tts_sync, text, voice, model_id=model_id)

    if not return_word_timings:
        return audio_bytes

    # Probe duration off the wire so timings line up with playback. ffprobe
    # call is ~30ms — well inside the budget for the audio-first path.
    duration_ms = await asyncio.to_thread(_probe_audio_duration_ms, audio_bytes)
    timings = synthesize_word_timings(text, duration_ms)
    logger.info("[TTS] synth timings: %d words over %s ms",
                len(timings), duration_ms if duration_ms else "(estimate)")
    return audio_bytes, timings
