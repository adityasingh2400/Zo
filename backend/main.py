import asyncio
import contextvars
import json
import os
import time
import base64
import logging
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("empire")

from config import (
    BACKEND_HOST, BACKEND_PORT,
    USE_AUDIO_FIRST, USE_KARAOKE, USE_PITCH_VEO, USE_BACKCHANNEL,
    USE_SPECULATIVE_BRIDGE, LIPSYNC_PROVIDER, USE_LIVE_PAD,
)
from agents.eyes import analyze_with_claude, analyze_and_script_claude, classify_comment_gemma, transcribe_voice
from agents.creator import remove_background, generate_3d_model
from agents.seller import (
    generate_comment_response,
    make_avatar_speak,
    text_to_speech,
    synthesize_word_timings,
    set_livetalking_session,
    render_comment_response_wav2lip,
    render_pitch_latentsync,
)
from agents.intake import process_video
from agents.threed import carousel_from_video, glb_from_image
from agents.bridge_clips import pick_bridge_clip, all_bridges
from agents.avatar_director import Director
from agents import router as comment_router
from agents import trace
from agents.router import _match_product_field  # used by speculative bridge

logger.info(
    "[flags] USE_AUDIO_FIRST=%s USE_KARAOKE=%s USE_PITCH_VEO=%s "
    "USE_BACKCHANNEL=%s USE_SPECULATIVE_BRIDGE=%s USE_LIVE_PAD=%s "
    "LIPSYNC_PROVIDER=%s",
    USE_AUDIO_FIRST, USE_KARAOKE, USE_PITCH_VEO,
    USE_BACKCHANNEL, USE_SPECULATIVE_BRIDGE, USE_LIVE_PAD,
    LIPSYNC_PROVIDER,
)

# ── State ──────────────────────────────────────────────

RENDER_DIR = Path(__file__).resolve().parent / "renders"
RENDER_DIR.mkdir(exist_ok=True)

pipeline_state: dict[str, Any] = {
    "status": "idle",
    "product_data": None,
    "product_photo_b64": None,
    "product_clean_b64": None,
    "model_3d": None,
    "view_3d": None,  # {kind, frames|url, ms, source}
    "transcript_extract": None,  # on-device structured pitch extraction
    "sales_script": None,
    "pitch_video_url": None,
    "last_response_video_url": None,
    "agent_log": [],
}

dashboard_clients: list[WebSocket] = []
phone_clients: list[WebSocket] = []

# ── Audience comment rate limiter ───────────────────────────────────────────
# 300 people in the room behind a conference NAT all egress from the same
# public IP, so we set the per-IP cap permissively (5 / minute) — enough
# headroom for one person typing fast without inviting a single bad actor
# from spamming the chat scroll. Comments above the cap are dropped silently
# (HTTP 429) so the form just looks unresponsive rather than scolding the
# user mid-demo.
AUDIENCE_RATE_PER_MIN = 5
AUDIENCE_TEXT_MAX_CHARS = 240
_audience_recent: dict[str, list[float]] = {}


def _audience_rate_check(ip: str) -> bool:
    """Return True if `ip` is allowed to post. Side-effect: records the
    timestamp on success and prunes entries older than 60s. Pure in-process
    state — fine for a 300-person live demo, would need Redis to scale."""
    now = time.time()
    bucket = _audience_recent.setdefault(ip, [])
    # Drop timestamps older than the 60s window. Keeps the dict bounded
    # under heavy load — without pruning, a long demo could leak memory.
    cutoff = now - 60.0
    while bucket and bucket[0] < cutoff:
        bucket.pop(0)
    if len(bucket) >= AUDIENCE_RATE_PER_MIN:
        return False
    bucket.append(now)
    return True

# Single Director instance owns all play_clip emission. Bound to the
# dashboard broadcast helper so it can talk to every connected client.
director: "Director | None" = None  # set in app startup, see below


PRODUCTS_PATH = Path(__file__).resolve().parent / "data" / "products.json"


def _load_active_product() -> None:
    """Read backend/data/products.json and pick an active product. The active
    product lives in pipeline_state["product_data"] so the router's
    respond_locally path can match against its qa_index immediately —
    no prior /api/sell upload required for the demo.

    INTENTIONALLY EMPTY (`{}`) for the judge-item demo. The judge hands
    over an arbitrary item live → /api/sell or /api/sell-video →
    run_sell_pipeline → product_data populates dynamically from Gemma 4
    vision + Claude. Pre-loading a product here would (a) show the wrong
    item name in the BUY card before the judge's item analysis lands and
    (b) cause respond_locally to fire wrong pre-rendered answers for any
    audience question whose keywords accidentally match the pre-loaded
    product's qa_index. Add products back here ONLY for non-judge-item
    rehearsal demos.

    Selection order:
      1. ACTIVE_PRODUCT_ID env var if set and present in the file
      2. First key in the JSON (dicts preserve insertion order)
    Missing file or unreadable JSON: log + skip (router falls back to cloud).
    """
    if not PRODUCTS_PATH.exists():
        logger.info("No products.json at %s — skipping pre-load", PRODUCTS_PATH)
        return
    try:
        with PRODUCTS_PATH.open() as f:
            products = json.load(f)
    except Exception as e:
        logger.warning("Failed to read products.json: %s", e)
        return
    if not products:
        return
    if not isinstance(products, dict):
        logger.warning("products.json must be a JSON object keyed by product id "
                       "(got %s) — skipping", type(products).__name__)
        return

    active_id = os.getenv("ACTIVE_PRODUCT_ID") or next(iter(products.keys()))
    product = products.get(active_id)
    if not isinstance(product, dict):
        logger.warning("Product %r is not an object — skipping", active_id)
        return
    if not product:
        logger.warning("ACTIVE_PRODUCT_ID=%s not in products.json", active_id)
        return

    pipeline_state["product_data"] = product
    pipeline_state["active_product_id"] = active_id
    qa_count = len(product.get("qa_index") or {})
    logger.info('[products] Loaded "%s" (id=%s) with %d Q/A entries',
                product.get("name", "?"), active_id, qa_count)


def log_event(agent: str, message: str, data: Any = None):
    entry = {
        "agent": agent,
        "message": message,
        "timestamp": time.time(),
        "data": data,
    }
    pipeline_state["agent_log"].append(entry)
    asyncio.ensure_future(broadcast_to_dashboards({
        "type": "agent_log",
        "entry": entry,
    }))


async def broadcast_to_dashboards(msg: dict):
    dead = []
    for ws in dashboard_clients:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    # A disconnecting client may already have been removed by its own
    # /ws/dashboard handler's disconnect path; tolerate that race.
    for ws in dead:
        try:
            dashboard_clients.remove(ws)
        except ValueError:
            pass
    # Director observer (REVISIONS §12) — every outgoing message also feeds
    # the motivated-idle state machine. Clean event-listener pattern: the
    # rest of the codebase doesn't need to call director.notify() at every
    # lifecycle moment, the broadcast wrapper does it for them.
    if director is not None:
        try:
            await director.observe(msg)
        except Exception:
            logger.exception("director.observe failed (non-fatal)")


# ── App ────────────────────────────────────────────────

def _startup_banner() -> None:
    """Print a comprehensive config snapshot at boot so a fresh terminal
    log immediately tells me what the running backend will and won't do.
    Single source of truth — anything that can affect demo behavior at
    runtime should be visible here. Compact + greppable."""
    from config import (
        BACKEND_HOST as H, BACKEND_PORT as P,
        WAV2LIP_URL as W, LATENTSYNC_URL as L,
        RUNPOD_POD_IP as POD, ELEVENLABS_VOICE_ID as VID,
        ELEVENLABS_API_KEY as KEY,
    )
    from agents.avatar_director import (
        TIER0_CROSSFADE_MS_DEFAULT, TIER1_CROSSFADE_MS_DEFAULT,
        READING_CHAT_HOLD_MS,
    )
    local_count = len(list(LOCAL_ANSWERS_DIR.glob("*.mp4"))) \
        if LOCAL_ANSWERS_DIR.exists() else 0
    generic_count = len(list((LOCAL_ANSWERS_DIR / "_generic").glob("*.mp4"))) \
        if (LOCAL_ANSWERS_DIR / "_generic").exists() else 0
    sep = "═" * 72
    logger.info(sep)
    logger.info("  Zo backend  %s:%s", H, P)
    logger.info(sep)
    logger.info("  feature flags        USE_AUDIO_FIRST=%s  USE_PITCH_VEO=%s  USE_KARAOKE=%s",
                USE_AUDIO_FIRST, USE_PITCH_VEO, USE_KARAOKE)
    logger.info("  reading_chat hold    %dms (avatar visibly reads before responding)",
                READING_CHAT_HOLD_MS)
    logger.info("  tier0 crossfade      %dms (idle rotation)",
                TIER0_CROSSFADE_MS_DEFAULT)
    logger.info("  tier1 crossfade      %dms (response transitions)",
                TIER1_CROSSFADE_MS_DEFAULT)
    logger.info("  wav2lip pod          %s", W or "(unset)")
    logger.info("  latentsync pod       %s", L or "(unset)")
    logger.info("  runpod ip            %s", POD or "(unset)")
    logger.info("  elevenlabs           voice=%s  key=%s",
                VID or "(default)", "yes" if KEY else "MISSING")
    logger.info("  local_answers/       %d product clips, %d _generic clips",
                local_count, generic_count)
    logger.info(sep)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global director
    _startup_banner()
    # Bring up the Avatar Director early so the first dashboard connect can
    # immediately receive a Tier 0 idle clip.
    director = Director(broadcast_to_dashboards)
    director.start_idle_rotation()
    logger.info("Avatar Director instantiated; idle rotation running.")
    # Pre-load Cactus models in background threads so the first request isn't
    # slow. Gemma 4 (vision + classify + script) and whisper-base (voice
    # transcription) live on separate Cactus handles; we load them
    # sequentially to avoid any re-entrant SDK init on startup.
    from agents.eyes import _get_cactus_model, _get_cactus_whisper_model, CACTUS_AVAILABLE
    if CACTUS_AVAILABLE:
        logger.info("Pre-loading Cactus Gemma 4 model (background thread)...")
        await asyncio.to_thread(_get_cactus_model)
        logger.info("Cactus/Gemma 4 model ready.")
        logger.info("Pre-loading Cactus whisper-base model (background thread)...")
        await asyncio.to_thread(_get_cactus_whisper_model)
        logger.info("Cactus/whisper-base model ready.")

    # Pre-warm the rembg pool. CoreML compiles a kernel on first call to a
    # given model — paying that cost here means the first user video upload
    # doesn't eat 30+ extra seconds. Fire and forget; if it fails the live
    # path still works (just pays the compile cost on first real call).
    try:
        from agents.threed import prewarm_rembg
        import asyncio as _aio
        _aio.create_task(prewarm_rembg("u2net"))
    except Exception as e:
        logger.warning("rembg prewarm scheduling failed: %s", e)

    # Load demo products + pre-select an active one for respond_locally. The
    # Hour 5-6 scope: the router can answer routine questions without any
    # prior /api/sell call. ACTIVE_PRODUCT_ID env var overrides the first-
    # key default if you want to swap between demo objects without editing
    # code on stage.
    _load_active_product()
    yield

app = FastAPI(title="Zo", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── WebSocket: Phone ───────────────────────────────────

@app.websocket("/ws/phone")
async def phone_ws(ws: WebSocket):
    await ws.accept()
    phone_clients.append(ws)
    log_event("SYSTEM", "Phone connected")
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            await handle_phone_message(msg, ws)
    except WebSocketDisconnect:
        phone_clients.remove(ws)
        log_event("SYSTEM", "Phone disconnected")


async def handle_phone_message(msg: dict, ws: WebSocket):
    msg_type = msg.get("type")

    if msg_type == "sell_command":
        asyncio.ensure_future(run_sell_pipeline(
            frame_b64=msg.get("frame", ""),
            voice_text=msg.get("voice_text", "sell this"),
        ))

    elif msg_type == "sell_video":
        # Phone-recorded product video, base64-encoded for WS transport.
        # Phone sends: {type: "sell_video", video_b64: "<base64 .mp4 or
        # .mov>", filename?: "clip.mp4", voice_text?: "sell this"}.
        #
        # We decode, drop to a temp file (the existing intake pipeline
        # works with a path, not bytes), kick off run_video_sell_pipeline
        # which extracts audio + frames, transcribes, and fires the
        # audio-first pitch via Director.dispatch_audio_first_pitch.
        #
        # Acks are sent back to the phone so the recorder UI can
        # display "received → analyzing → going live". Dashboard also
        # gets a phone_video_received event for operator visibility.
        await _handle_phone_sell_video(msg, ws)

    elif msg_type == "comment":
        asyncio.ensure_future(run_comment_pipeline(
            comment=msg.get("text", ""),
        ))

    elif msg_type == "frame":
        pipeline_state["product_photo_b64"] = msg.get("frame", "")
        await broadcast_to_dashboards({
            "type": "phone_frame",
            "frame": msg.get("frame", "")[:100] + "...",
        })


async def _handle_phone_sell_video(msg: dict, ws: WebSocket) -> None:
    """Decode a phone-uploaded video over WS, write to temp file, kick
    off run_video_sell_pipeline. Sends progress acks back to the phone
    so the recorder UI can show received → analyzing → going_live, and
    a phone_video_received event to the dashboards for operator visibility.

    Phone contract:
      send: {type: "sell_video", video_b64: <str>,
             filename?: <str>, voice_text?: <str>, mime?: <str>}
      recv: {type: "phone_ack", stage: "received", bytes: <int>}
      recv: {type: "phone_ack", stage: "pipeline_started", session_id: <str>}
            (any pipeline failure surfaces via the dashboard agent_log.)
    """
    import tempfile, uuid as _uuid

    video_b64 = msg.get("video_b64") or ""
    if not video_b64:
        try:
            await ws.send_json({"type": "phone_ack", "stage": "error",
                                "reason": "missing_video_b64"})
        except Exception:
            pass
        log_event("SYSTEM", "Phone sell_video missing video_b64")
        return

    try:
        video_bytes = base64.b64decode(video_b64)
    except Exception as e:
        try:
            await ws.send_json({"type": "phone_ack", "stage": "error",
                                "reason": f"b64_decode_failed: {e}"})
        except Exception:
            pass
        log_event("SYSTEM", f"Phone sell_video b64 decode failed: {e}")
        return

    if not video_bytes:
        try:
            await ws.send_json({"type": "phone_ack", "stage": "error",
                                "reason": "empty_after_decode"})
        except Exception:
            pass
        return

    filename = msg.get("filename") or "phone_clip.mp4"
    suffix = Path(filename).suffix or ".mp4"
    voice_text = msg.get("voice_text") or "sell this"
    session_id = f"phone_{_uuid.uuid4().hex[:10]}"

    # Write to a temp file — run_video_sell_pipeline expects a path
    # because the intake pipeline shells out to ffmpeg/ffprobe which
    # want a real file on disk.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(video_bytes)
        video_path = f.name
    log_event("SYSTEM", f"Phone video received: {len(video_bytes)} bytes "
              f"({filename}) session={session_id}")
    try:
        await ws.send_json({"type": "phone_ack", "stage": "received",
                            "bytes": len(video_bytes),
                            "session_id": session_id})
    except Exception:
        pass
    await broadcast_to_dashboards({
        "type": "phone_video_received",
        "bytes": len(video_bytes),
        "filename": filename,
        "session_id": session_id,
    })

    asyncio.ensure_future(run_video_sell_pipeline(video_path, voice_text))
    try:
        await ws.send_json({"type": "phone_ack", "stage": "pipeline_started",
                            "session_id": session_id})
    except Exception:
        pass


# ── WebSocket: Dashboard ──────────────────────────────

@app.websocket("/ws/dashboard")
async def dashboard_ws(ws: WebSocket):
    await ws.accept()
    dashboard_clients.append(ws)

    await ws.send_json({
        "type": "state_sync",
        "state": {
            "status": pipeline_state["status"],
            "product_data": pipeline_state["product_data"],
            "sales_script": pipeline_state.get("sales_script"),
            "pitch_video_url": pipeline_state.get("pitch_video_url"),
            "last_response_video_url": pipeline_state.get("last_response_video_url"),
            "view_3d": pipeline_state.get("view_3d"),
            "transcript_extract": pipeline_state.get("transcript_extract"),
            "director_state": director.replay_state() if director else None,
            "agent_log": pipeline_state["agent_log"][-50:],
        },
    })

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "simulate_comment":
                # Route the comment through the 4-tool dispatcher. Local
                # tools (respond_locally / play_canned_clip / block_comment)
                # resolve in <300ms; cloud tool forwards to the no-Wav2Lip
                # api_respond_to_comment audio + speaking-idle pipeline.
                comment_text = msg.get("text", "")
                # Echo to all dashboards as an audience_comment so the
                # /stage TikTokShopOverlay chat rail surfaces operator-
                # typed test comments (mission item 3 — operator can
                # rehearse without a phone). Tagged username='operator'
                # so it visually distinguishes from QR submissions.
                # The originating client's useEmpireSocket dedups on
                # text+t0 so its own ChatPanel pending pill (created
                # optimistically by sendComment) doesn't double-up.
                if comment_text.strip():
                    await broadcast_to_dashboards({
                        "type": "audience_comment",
                        "username": "operator",
                        "text": comment_text,
                        "ts": int(time.time() * 1000),
                    })
                async def _run():
                    try:
                        await run_routed_comment(comment_text)
                    except Exception as e:
                        log_event("SELLER", f"comment pipeline error: {e}")
                asyncio.create_task(_run())

            elif msg.get("type") == "simulate_sell":
                frame = pipeline_state.get("product_photo_b64", "")
                asyncio.ensure_future(run_sell_pipeline(
                    frame_b64=frame,
                    voice_text=msg.get("voice_text", "sell this"),
                ))

            elif msg.get("type") == "livetalking_session":
                set_livetalking_session(msg.get("session_id", ""))
                log_event("SYSTEM", f"LiveTalking WebRTC session: {msg.get('session_id', '')[:20]}...")

            elif msg.get("type") == "stage_ready":
                # Dashboard tells us Tier 0 is painting frames; safe to send tier 1.
                if director:
                    director.mark_ready()

            elif msg.get("type") == "clip_ack":
                # Dashboard playback telemetry: started / ended / stalled / skipped.
                # Emit to logs for now; future versions can use this to detect stuck clips.
                logger.info("[clip_ack] %s/%s status=%s",
                            msg.get("intent"), msg.get("url"), msg.get("status"))

            elif msg.get("type") == "mic_pressed":
                # USE_BACKCHANNEL: VoiceMic fires this on pointer-down BEFORE
                # the audio recording even starts so the listening-attentive
                # pose can swap into Tier 1 within ~50ms of the press. Visual
                # only (REVISIONS §8) — no "mhm" audio.
                if USE_BACKCHANNEL and director:
                    asyncio.create_task(director.play_listening_attentive())

    except WebSocketDisconnect:
        dashboard_clients.remove(ws)


# ── Sell Pipeline ──────────────────────────────────────

async def run_sell_pipeline(frame_b64: str, voice_text: str):
    pipeline_state["status"] = "analyzing"
    pipeline_state["agent_log"] = []
    logger.info("=" * 60)
    logger.info("SELL PIPELINE START")
    logger.info("  frame_b64 length: %d chars", len(frame_b64))
    logger.info("  voice_text: %s", voice_text[:100])
    logger.info("=" * 60)
    pipeline_start = time.time()

    # PHASE 1: Single Claude call (vision + script) + background removal in parallel
    log_event("EYES", "Analyzing product + writing script (single Claude call + bg removal)...")
    t0 = time.time()

    async def _claude_combined():
        try:
            return await analyze_and_script_claude(frame_b64, voice_text)
        except Exception as e:
            logger.error("Claude combined error: %s", e)
            return {"error": str(e), "source": "claude_error"}

    async def _bg_removal():
        try:
            return await remove_background(frame_b64)
        except Exception as e:
            logger.error("Background removal error: %s", e)
            return None

    claude_result, clean_b64 = await asyncio.gather(
        _claude_combined(), _bg_removal()
    )
    phase1_ms = int((time.time() - t0) * 1000)

    # Extract product data and script from combined result
    product_data = claude_result.get("product", claude_result)
    product_data["source"] = "claude_cloud"
    script = claude_result.get("script", "")
    if not script:
        script = f"Check out this amazing {product_data.get('name', 'product')}!"

    log_event("EYES", f"Claude: {product_data.get('name', 'done')} ({phase1_ms}ms)")
    pipeline_state["product_data"] = product_data
    await broadcast_to_dashboards({"type": "product_data", "data": product_data})

    pipeline_state["sales_script"] = script
    log_event("SELLER", f"Sales pitch ready ({phase1_ms}ms)", {"script": script})
    await broadcast_to_dashboards({"type": "sales_script", "script": script})

    if clean_b64:
        pipeline_state["product_clean_b64"] = clean_b64
        log_event("CREATOR", f"Clean product photo ready ({phase1_ms}ms)")
        await broadcast_to_dashboards({"type": "product_photo", "photo": clean_b64})
    else:
        log_event("CREATOR", "Background removal failed")

    asyncio.ensure_future(run_3d_generation(frame_b64))

    # PHASE 2: pitch playback. Two paths:
    #   USE_PITCH_VEO=1 (default): audio-first dispatch. TTS the script,
    #     save the audio, broadcast pitch_audio + emit a muted looping
    #     speaking-pose Veo clip on Tier 1. Karaoke captions populate
    #     word-by-word. ~600-900ms from script ready → first audible
    #     syllable.
    #   USE_PITCH_VEO=0 (kill switch): legacy Wav2Lip render (5-7s+).
    #     Kept as a one-env-flag rollback in case the audio-first path
    #     misbehaves on stage.
    pipeline_state["status"] = "selling"
    log_event("SELLER", "Avatar going live...")
    if director:
        await director.set_voice_state("responding")

    pitch_t0 = time.time()
    try:
        if USE_PITCH_VEO:
            await _run_audio_first_pitch(script, pitch_t0)
        else:
            await _run_wav2lip_pitch(script, pitch_t0)
    except Exception as e:
        logger.exception("[pitch] render failed")
        log_event("SELLER", f"Pitch render failed: {e}")
        # Last-ditch: TTS-only fallback so at least audio plays.
        try:
            audio_bytes = await text_to_speech(script)
            if audio_bytes:
                await broadcast_to_dashboards({
                    "type": "tts_audio",
                    "audio": base64.b64encode(audio_bytes).decode(),
                    "format": "mp3",
                })
                log_event("SELLER", "Fell back to TTS-only audio (no lipsync video)")
        except Exception as e2:
            log_event("SELLER", f"TTS fallback also failed: {e2}")
        if director:
            await director.set_voice_state(None)

    pipeline_state["status"] = "live"
    total_ms = int((time.time() - pipeline_start) * 1000)
    log_event("SYSTEM", f"Zo is LIVE. Total pipeline: {total_ms}ms")
    logger.info("=" * 60)
    logger.info("SELL PIPELINE COMPLETE — %dms total", total_ms)
    logger.info("=" * 60)
    await broadcast_to_dashboards({"type": "status", "status": "live"})


async def _run_audio_first_pitch(script: str, pitch_t0: float) -> None:
    """Audio-first pitch: TTS → save → broadcast pitch_audio → muted
    looping Tier 1 video. Used for the phone-uploaded video pipeline
    where the script is freshly generated by Claude per upload.

    The dashboard plays the audio through its standalone <audio> element
    (KaraokeCaptions tracks word-by-word), the looped speaking-pose Veo
    clip runs underneath muted, the TranslationChip mounts top-right.
    Total time from script ready → first audible syllable: ~600-900ms.
    """
    from agents.seller import _probe_audio_duration_ms
    t0 = time.time()
    audio_bytes, word_timings = await text_to_speech(
        script, return_word_timings=True,
    )
    tts_ms = int((time.time() - t0) * 1000)
    if not audio_bytes:
        raise RuntimeError("TTS returned empty audio")
    log_event("SELLER", f"TTS ready ({tts_ms}ms, {len(audio_bytes)}B, {len(word_timings)} words)")

    # Save audio to /response_audio (existing static mount; reused so we
    # don't add yet another dir for ephemeral pitch dispatches). The
    # `pitch_dispatch_` prefix distinguishes these from per-comment
    # response audio in operator debug listings.
    import uuid as _uuid
    fname = f"pitch_dispatch_{_uuid.uuid4().hex[:12]}.mp3"
    audio_path = RESPONSE_AUDIO_DIR / fname
    audio_path.write_bytes(audio_bytes)
    audio_url = f"/response_audio/{fname}"
    audio_ms = await asyncio.to_thread(_probe_audio_duration_ms, audio_bytes) or 0

    # Stash pitch URL for state_sync (replays audio on dashboard refresh
    # mid-pitch). The pitch_audio event will fire the actual playback.
    pipeline_state["pitch_video_url"] = audio_url

    if director:
        await director.dispatch_audio_first_pitch(
            audio_url=audio_url,
            word_timings=word_timings,
            audio_ms=audio_ms,
            script=script,
            slug=pipeline_state.get("active_product_id") or "video_upload",
        )

    # Legacy pitch_video event so the existing dashboard plumbing
    # (CostTicker, RoutingPanel, anything tracking pitches) sees the
    # transition. URL points at the audio dispatch since there's no
    # rendered video file in the audio-first path.
    await broadcast_to_dashboards({
        "type": "pitch_video",
        "url": audio_url,
        "render_ms": 0,                   # no Wav2Lip render fired
        "tts_ms": tts_ms,
        "audio_ms": audio_ms,
        "backend": "audio_first",
        "audio_already_playing": True,
    })
    pitch_total_ms = int((time.time() - pitch_t0) * 1000)
    log_event(
        "SELLER",
        f"Audio-first pitch dispatched! ({pitch_total_ms}ms total, "
        f"{audio_ms}ms audio, {len(word_timings)} timed words)",
    )


async def _run_wav2lip_pitch(script: str, pitch_t0: float) -> None:
    """Legacy Wav2Lip pitch render. Kept as the USE_PITCH_VEO=0 kill switch
    rollback. Renders TTS audio against the active speaking substrate via
    the pod's Wav2Lip server; broadcasts a pitch_video event the dashboard
    plays on Tier 1 with embedded audio. Slower (5-7s+) but lip-synced."""
    t0 = time.time()
    audio_bytes = await text_to_speech(script)
    tts_ms = int((time.time() - t0) * 1000)
    if not audio_bytes:
        raise RuntimeError("TTS returned empty audio")
    log_event("SELLER", f"TTS ready ({tts_ms}ms, {len(audio_bytes)}B)")

    substrate = director.current_substrate_pod_path() if director else None
    t0 = time.time()
    try:
        if substrate:
            video_bytes, headers = await render_comment_response_wav2lip(
                audio_bytes, source_path_on_pod=substrate, out_height=1080,
            )
        else:
            video_bytes, headers = await render_comment_response_wav2lip(
                audio_bytes, out_height=1080,
            )
    except Exception as e:
        err_str = str(e).lower()
        if substrate and director and ("404" in err_str or "400" in err_str or "not found" in err_str):
            logger.warning("[pitch] substrate %s unavailable, falling back: %s", substrate, e)
            director.mark_substrate_status(substrate, False)
            video_bytes, headers = await render_comment_response_wav2lip(
                audio_bytes, out_height=1080,
            )
        else:
            raise
    lipsync_ms = int((time.time() - t0) * 1000)
    log_event("SELLER", f"Wav2Lip pitch rendered ({lipsync_ms}ms)")

    url = _save_render("pitch", video_bytes)
    pipeline_state["pitch_video_url"] = url

    if director:
        await director.play_response(url)
        rendered_path = RENDER_DIR / Path(url).name
        play_ms = _probe_video_duration_ms(rendered_path)
        if play_ms is None:
            word_count = len(script.split())
            play_ms = int(max(2500, word_count * 350))
        play_ms_with_tail = play_ms + 400

        async def _release_pitch_to_idle(delay_ms: int):
            await asyncio.sleep(delay_ms / 1000)
            if director:
                await director.fade_to_idle()
                await director.set_voice_state(None)
        asyncio.ensure_future(_release_pitch_to_idle(play_ms_with_tail))

    await broadcast_to_dashboards({
        "type": "pitch_video",
        "url": url,
        "render_ms": lipsync_ms,
        "tts_ms": tts_ms,
        "backend": "wav2lip",
    })
    pitch_total_ms = int((time.time() - pitch_t0) * 1000)
    log_event("SELLER", f"Avatar speaking! ({pitch_total_ms}ms total, lipsynced via Wav2Lip)")


async def run_3d_generation(frame_b64: str):
    log_event("CREATOR", "Generating 3D model (TripoSR)...")
    t0 = time.time()
    result = await generate_3d_model(frame_b64)
    ms = int((time.time() - t0) * 1000)
    if result and "error" not in result:
        pipeline_state["model_3d"] = result
        log_event("CREATOR", f"3D model ready ({ms}ms)")
        await broadcast_to_dashboards({"type": "model_3d", "data": result})
    else:
        log_event("CREATOR", f"3D model skipped ({result})")


async def run_carousel_pipeline(video_path: str):
    """Tier-1 3D view: extract N rembg-cleaned angle frames, broadcast carousel."""
    log_event("CREATOR", "Building 3D angle carousel from video...")
    try:
        # 24 frames @ 640px with rembg+stabilization = silky spin, product
        # stays centered + at constant size, no edge flicker between frames.
        view = await carousel_from_video(
            video_path, n_frames=24, out_size=640, clean_bg=True,
            rembg_model="u2net", stabilize=True,
        )
    except Exception as e:
        log_event("CREATOR", f"Carousel failed: {e}")
        logger.exception("carousel pipeline error")
        return
    pipeline_state["view_3d"] = view
    n = len(view.get("frames", []))
    cached = " (cached)" if view.get("cached") else ""
    log_event("CREATOR", f"3D spin ready: {n} frames in {view.get('ms', 0)}ms{cached}", {
        "kind": view.get("kind"), "source": view.get("source"),
    })
    await broadcast_to_dashboards({"type": "view_3d", "data": view})


# ── Comment Pipeline ──────────────────────────────────

async def run_comment_pipeline(comment: str):
    product_data = pipeline_state.get("product_data", {})

    # Step 1: Gemma 4 classifies + drafts response (on-device, FREE)
    log_event("EYES", f'Comment: "{comment}" — classifying on-device...')
    t0 = time.time()
    classification = await classify_comment_gemma(comment)
    class_ms = int((time.time() - t0) * 1000)
    comment_type = classification.get("type", "question")
    log_event("EYES", f"Classified as {comment_type} ({class_ms}ms, FREE)", classification)

    # Step 2: Claude refines the response with full product context
    log_event("SELLER", "Refining response with product context (Claude)...")
    t0 = time.time()
    try:
        response_text = await generate_comment_response(comment, product_data, comment_type)
    except Exception as e:
        response_text = "Thanks for the question! Let me look into that."
        log_event("SELLER", f"Response gen error: {e}")
    resp_ms = int((time.time() - t0) * 1000)
    log_event("SELLER", f"Response generated ({resp_ms}ms)", {"response": response_text})

    await broadcast_to_dashboards({
        "type": "comment_response",
        "comment": comment,
        "response": response_text,
    })

    # Step 3: Avatar speaks (text → LiveTalking handles TTS + lip sync)
    log_event("SELLER", "Avatar responding...")
    t0 = time.time()
    lt_result = await make_avatar_speak(response_text)
    lt_ms = int((time.time() - t0) * 1000)
    if "error" in lt_result:
        log_event("SELLER", f"LiveTalking: {lt_result['error']} ({lt_ms}ms)")
        # Fallback: TTS audio only
        try:
            audio_bytes = await text_to_speech(response_text)
            if audio_bytes:
                await broadcast_to_dashboards({
                    "type": "tts_audio",
                    "audio": base64.b64encode(audio_bytes).decode(),
                    "format": "mp3",
                })
        except Exception:
            pass
    else:
        log_event("SELLER", f"Avatar responding with lip sync! ({lt_ms}ms)")


# ── REST endpoints (for testing without WebSocket) ─────

@app.post("/api/analyze")
async def api_analyze(file: UploadFile = File(...), voice_text: str = Form("sell this")):
    contents = await file.read()
    frame_b64 = base64.b64encode(contents).decode()
    result = await analyze_with_claude(frame_b64, voice_text)
    return result


@app.post("/api/sell")
async def api_sell(file: UploadFile = File(...), voice_text: str = Form("sell this")):
    contents = await file.read()
    frame_b64 = base64.b64encode(contents).decode()
    asyncio.ensure_future(run_sell_pipeline(frame_b64, voice_text))
    return {"status": "pipeline_started"}


@app.post("/api/sell-video")
async def api_sell_video(file: UploadFile = File(...), voice_text: str = Form("sell this")):
    """Upload a product video. Extracts frames + transcript, runs full pipeline."""
    import tempfile
    logger.info("[API] /api/sell-video called — file: %s, size: uploading, voice: %s",
                file.filename, voice_text[:50])
    contents = await file.read()
    logger.info("[API] Video received: %d bytes (%s)", len(contents), file.filename)
    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(contents)
        video_path = f.name
    logger.info("[API] Saved to temp: %s", video_path)

    asyncio.ensure_future(run_video_sell_pipeline(video_path, voice_text))
    return {"status": "video_pipeline_started", "bytes": len(contents)}


async def run_video_sell_pipeline(video_path: str, voice_text: str):
    """Full pipeline from video: intake → analyze → sell."""
    pipeline_state["status"] = "ingesting"
    pipeline_state["agent_log"] = []
    logger.info("=" * 60)
    logger.info("VIDEO PIPELINE START")
    logger.info("  video_path: %s", video_path)
    logger.info("  voice_text: %s", voice_text[:100])
    logger.info("=" * 60)

    log_event("SYSTEM", "Video received. Starting intake pipeline...")

    # Run intake (audio+frames+transcript) and carousel (angle spin) in
    # parallel — both consume the same video, neither blocks the other.
    t0 = time.time()
    try:
        logger.info("[INTAKE] Starting video processing + carousel in parallel...")
        intake_result, _carousel_done = await asyncio.gather(
            process_video(video_path),
            run_carousel_pipeline(video_path),
        )
    except Exception as e:
        logger.error("[INTAKE] FAILED: %s", e)
        logger.error(traceback.format_exc())
        log_event("SYSTEM", f"Video intake failed: {e}")
        Path(video_path).unlink(missing_ok=True)
        return
    finally:
        Path(video_path).unlink(missing_ok=True)

    intake_ms = int((time.time() - t0) * 1000)
    transcript = intake_result["transcript"]
    best_frames_b64 = intake_result["best_frames_b64"]
    timings = intake_result["timings"]

    log_event("SYSTEM", f"Intake complete ({intake_ms}ms)", {
        "frames_extracted": timings["frame_count"],
        "frames_kept": timings["filtered_frame_count"],
        "transcript_length": len(transcript),
    })
    logger.info("[INTAKE] Complete in %dms", intake_ms)
    logger.info("[INTAKE]   frames: %d raw → %d best", timings["frame_count"], timings["filtered_frame_count"])
    logger.info("[INTAKE]   transcript: %d chars", len(transcript))
    logger.info("[INTAKE]   timings: %s", json.dumps(timings))

    transcript_extract_hint = ""  # injected into Claude prompt below
    if transcript:
        log_event("EYES", f'Seller said: "{transcript[:200]}..."')
        await broadcast_to_dashboards({"type": "transcript", "text": transcript})
        # Race transcript_extract against a tight timeout. If Cactus on NPU
        # finishes in <1.5s we ground Claude's prompt on its hints. If it's
        # slower (CPU prefill on Mac dev = 10s), we fire Claude without
        # waiting and let extract land on the dashboard separately.
        try:
            from agents.transcript_extract import (
                extract_transcript_signals, hint_block_for_claude,
            )
            extract_task = asyncio.create_task(
                extract_transcript_signals(transcript)
            )
            try:
                extract = await asyncio.wait_for(asyncio.shield(extract_task), timeout=1.5)
                transcript_extract_hint = hint_block_for_claude(extract)
                pipeline_state["transcript_extract"] = extract
                await broadcast_to_dashboards({"type": "transcript_extract", "data": extract})
                log_event("EYES",
                          f"Transcript extract ready in time ({extract.get('latency_ms', 0)}ms)",
                          {"source": extract.get("source")})
            except asyncio.TimeoutError:
                # Extract is slow — let Claude run unblocked. The pending
                # task keeps running and reports when it lands.
                log_event("EYES", "Transcript extract slow (>1.5s), running unblocked")
                asyncio.ensure_future(_finish_transcript_extract(extract_task))
        except Exception as e:
            logger.warning("[TRANSCRIPT_EXTRACT] setup failed: %s", e)

    if best_frames_b64:
        combined_voice = f"{voice_text}. Seller's narration: {transcript}" if transcript else voice_text
        if transcript_extract_hint:
            # Append the on-device hint block. Claude treats it as additional
            # context and grounds the script accordingly.
            combined_voice = f"{combined_voice}\n\n{transcript_extract_hint}"
        pipeline_state["product_photo_b64"] = best_frames_b64[0]
        await broadcast_to_dashboards({"type": "phone_frame", "frame": best_frames_b64[0][:100] + "..."})
        await run_sell_pipeline(best_frames_b64[0], combined_voice)
    else:
        log_event("SYSTEM", "No usable frames extracted from video")


async def _finish_transcript_extract(task: asyncio.Task) -> None:
    """Helper: wait for a slow extract to finish, then broadcast it.
    Used when the timeout race in run_video_sell_pipeline punted."""
    try:
        extract = await task
    except Exception as e:
        logger.warning("[TRANSCRIPT_EXTRACT] late task failed: %s", e)
        return
    if not extract or extract.get("source") in (None, "empty"):
        return
    pipeline_state["transcript_extract"] = extract
    await broadcast_to_dashboards({"type": "transcript_extract", "data": extract})
    log_event("EYES", "Transcript extract late-arrived (post-Claude)", {
        "source": extract.get("source"),
        "latency_ms": extract.get("latency_ms"),
    })


# NOTE: the legacy run_transcript_extract task has been folded into
# run_video_sell_pipeline so the extract grounds Claude's prompt directly
# (via hint_block_for_claude). _finish_transcript_extract handles the
# slow-path case where extract didn't beat the 1.5s timeout.


@app.post("/api/comment")
async def api_comment(text: str = Form(...)):
    """Public REST entry-point for a viewer comment. Routes through the
    4-tool dispatcher (run_routed_comment) which picks between:
      • respond_locally — pre-rendered LatentSync MP4 from local_answers/
        (true lip-sync, sub-second response)
      • play_canned_clip — pre-rendered LatentSync bridge clip
      • block_comment — spam filter, no avatar response
      • escalate_to_cloud — api_respond_to_comment, audio + KaraokeCaptions
        + clean speaking-pose loop on Tier 1 (no Wav2Lip, pristine quality
        but no precise lip-sync on the novel response)
    Same path used by the voice agent and the WS simulate_comment message
    and the QR /comment audience form (POST /api/audience_comment)."""
    asyncio.ensure_future(run_routed_comment(text))
    return {"status": "processing"}


@app.get("/api/state")
async def api_state():
    return {
        "status": pipeline_state["status"],
        "product_data": pipeline_state["product_data"],
        "has_photo": pipeline_state["product_clean_b64"] is not None,
        "has_3d": pipeline_state["model_3d"] is not None,
        "log_count": len(pipeline_state["agent_log"]),
    }


# ── Audience-facing endpoints (QR-driven comment intake) ───────────────────
#
# /comment serves a tiny mobile-first HTML page that audience phones load
# after scanning the QR code on the intro slide. It posts to
# /api/audience_comment, which (a) broadcasts the comment to dashboards so
# it scrolls into the TikTokShopOverlay chat in real time AND (b) feeds it
# into the same run_routed_comment pipeline a typed comment uses, so the
# router + cost ticker + avatar response all fire identically.
#
# Tunnel-friendly: accepts whatever public hostname Cloudflare assigns
# (you'd run `cloudflared tunnel --url http://localhost:8000` then point
# the QR at https://<random>.trycloudflare.com/comment).

# The form is a single string so we don't drag in a Jinja template engine
# for a 60-line page. CSP-friendly: no external assets, no inline data
# URIs that need network fetches. Loads instantly even on weak hotspot.
_COMMENT_FORM_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover" />
  <title>Zo · Live Comment</title>
  <style>
    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    html, body { margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
                   "Segoe UI", Roboto, Inter, Arial, sans-serif;
      background: radial-gradient(ellipse at top, #1a0b2e 0%, #050507 60%);
      color: #fafafa;
      min-height: 100vh; min-height: 100dvh;
      display: flex; flex-direction: column; align-items: center;
      padding: 28px 20px 32px; gap: 20px;
    }
    .logo {
      font-weight: 900; letter-spacing: 6px; font-size: 28px;
      background: linear-gradient(135deg,#ec4899,#7c3aed,#3b82f6);
      -webkit-background-clip: text; background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .pill {
      font-size: 11px; font-weight: 800; letter-spacing: 2px;
      color: #ec4899; padding: 4px 10px; border-radius: 999px;
      background: rgba(236,72,153,0.12); border: 1px solid rgba(236,72,153,0.4);
      text-transform: uppercase;
    }
    h1 {
      margin: 6px 0 0; font-size: 22px; font-weight: 800;
      text-align: center; line-height: 1.25;
    }
    p.sub { margin: 0; color: #a1a1aa; font-size: 14px; text-align: center; }
    form {
      width: 100%; max-width: 420px; display: flex; flex-direction: column;
      gap: 12px; margin-top: 8px;
    }
    textarea {
      background: #18181b; color: #fafafa; border: 1px solid #3f3f46;
      border-radius: 14px; padding: 14px 16px; font-size: 17px;
      font-family: inherit; resize: none; min-height: 92px; outline: none;
      transition: border-color 200ms ease, box-shadow 200ms ease;
    }
    textarea:focus { border-color: #ec4899; box-shadow: 0 0 0 3px rgba(236,72,153,0.2); }
    button {
      background: linear-gradient(135deg,#ec4899,#f43f5e);
      color: #fff; border: none; border-radius: 14px;
      padding: 14px 18px; font-size: 17px; font-weight: 900;
      letter-spacing: 1.2px; cursor: pointer;
      box-shadow: 0 6px 18px rgba(244,63,94,0.45);
      transition: transform 120ms ease, opacity 200ms ease;
    }
    button:active { transform: scale(0.98); }
    button:disabled { opacity: 0.55; cursor: not-allowed; }
    .meta {
      display: flex; justify-content: space-between; align-items: center;
      color: #a1a1aa; font-size: 12px; padding: 0 4px;
    }
    .meta b { color: #fafafa; font-weight: 700; }
    .ack {
      color: #22c55e; font-weight: 800; font-size: 14px;
      text-align: center; min-height: 20px;
      transition: opacity 240ms ease;
    }
    .feed {
      width: 100%; max-width: 420px; display: flex; flex-direction: column;
      gap: 6px; margin-top: 4px;
    }
    .feed .row {
      background: rgba(24,24,27,0.7); border: 1px solid #27272a;
      border-radius: 10px; padding: 8px 10px;
      font-size: 13px; color: #d4d4d8;
    }
    .feed .row .you { color: #fbcfe8; font-weight: 800; margin-right: 6px; }
    .footer {
      margin-top: auto; color: #52525b; font-size: 11px; text-align: center;
      letter-spacing: 0.4px;
    }
  </style>
</head>
<body>
  <div class="logo">Zo</div>
  <span class="pill">Live · Ask anything</span>
  <h1>Drop a question for the seller.</h1>
  <p class="sub">Your comment goes straight to the live chat. The avatar replies in under a second.</p>
  <form id="f">
    <textarea id="t" placeholder="e.g. is it real leather? does it ship overseas?" maxlength="240" autofocus></textarea>
    <div class="meta">
      <span>Posting as <b id="u">@guest</b></span>
      <span id="cnt">0 / 240</span>
    </div>
    <button id="b" type="submit">SEND →</button>
    <div id="ack" class="ack" aria-live="polite"></div>
  </form>
  <div class="feed" id="feed"></div>
  <div class="footer">Be cool. We rate-limit at 5/min — the avatar has a queue to clear.</div>
  <script>
  (function(){
    var u = "guest_" + (1000 + Math.floor(Math.random()*9000));
    document.getElementById("u").textContent = "@" + u;
    var t = document.getElementById("t");
    var cnt = document.getElementById("cnt");
    var ack = document.getElementById("ack");
    var btn = document.getElementById("b");
    var feed = document.getElementById("feed");
    function updateCnt() { cnt.textContent = t.value.length + " / 240"; }
    t.addEventListener("input", updateCnt);
    document.getElementById("f").addEventListener("submit", function(e){
      e.preventDefault();
      var text = t.value.trim();
      if (!text) return;
      btn.disabled = true; ack.style.opacity = 0; ack.textContent = "";
      fetch("/api/audience_comment", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({username: u, text: text})
      }).then(function(r){
        if (r.status === 429) {
          ack.textContent = "Slow down a sec — try again in a moment.";
          ack.style.color = "#fbbf24"; ack.style.opacity = 1;
          return null;
        }
        if (!r.ok) {
          ack.textContent = "Couldn't send. Connection?";
          ack.style.color = "#ef4444"; ack.style.opacity = 1;
          return null;
        }
        ack.textContent = "Sent ✓ — ask another.";
        ack.style.color = "#22c55e"; ack.style.opacity = 1;
        var row = document.createElement("div"); row.className = "row";
        row.innerHTML = '<span class="you">@' + u + '</span>' +
          text.replace(/[<>&]/g, function(c){ return {"<":"&lt;",">":"&gt;","&":"&amp;"}[c]; });
        feed.prepend(row);
        while (feed.children.length > 6) feed.removeChild(feed.lastChild);
        t.value = ""; updateCnt(); t.focus();
        return null;
      }).catch(function(err){
        ack.textContent = "Network hiccup — try again.";
        ack.style.color = "#ef4444"; ack.style.opacity = 1;
      }).finally(function(){
        setTimeout(function(){ btn.disabled = false; }, 600);
      });
    });
    updateCnt();
  })();
  </script>
</body>
</html>"""


@app.get("/comment", response_class=HTMLResponse)
async def comment_form() -> HTMLResponse:
    """Public mobile-first comment form. Audience scans the QR on the
    intro slide, lands here, types a question, and the avatar responds
    on stage in <8s. Zero auth, no JS framework — loads on a weak hotspot."""
    return HTMLResponse(_COMMENT_FORM_HTML, headers={"Cache-Control": "no-store"})


@app.post("/api/audience_comment")
async def api_audience_comment(payload: dict, request: Request):
    """Audience-submitted comment from a phone. Two side effects:
      1. Broadcast `audience_comment` to all connected dashboards so the
         TikTokShopOverlay chat scroll renders the bubble immediately,
         attributed to @<username>.
      2. Hand the comment text to `run_routed_comment` so the same router
         + cost-ticker + avatar pipeline a typed comment uses fires for
         the audience input. The router emits routing_decision (drives
         the cost ticker) and eventually comment_response_video (drives
         the avatar response).
    """
    # Identify the client by best-available IP. Behind a Cloudflare tunnel
    # we'll get cf-connecting-ip; fall back to whatever uvicorn populated.
    client_ip = (
        request.headers.get("cf-connecting-ip")
        or request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        or (request.client.host if request.client else "unknown")
    )
    if not _audience_rate_check(client_ip):
        # Drop silently from the audience view — the form will show "slow
        # down" but the chat scroll never sees the comment.
        raise HTTPException(status_code=429, detail="rate_limited")

    text_raw = (payload.get("text") or "").strip()
    if not text_raw:
        raise HTTPException(status_code=400, detail="empty_text")
    # Trim to a sane upper bound so a runaway client can't broadcast
    # multi-kilobyte messages into the dashboard chat scroll.
    text = text_raw[:AUDIENCE_TEXT_MAX_CHARS]
    username_raw = (payload.get("username") or "guest").strip()
    # Same idea for the username — only allow sane chars and cap length.
    username = "".join(ch for ch in username_raw if ch.isalnum() or ch in "_-")[:24] or "guest"
    ts = int(time.time() * 1000)

    # Surface immediately to the overlay so the audience sees their comment
    # land in the chat scroll BEFORE the avatar response renders.
    await broadcast_to_dashboards({
        "type": "audience_comment",
        "username": username,
        "text": text,
        "ts": ts,
    })
    log_event("AUDIENCE", f"@{username}: {text[:80]}", {"ip": client_ip})

    # Route through the same dispatcher typed comments use. Fire-and-forget
    # so the HTTP response back to the phone returns instantly — we don't
    # want the form to hang while Wav2Lip renders a 5s response.
    async def _route():
        try:
            await run_routed_comment(text)
        except Exception as e:
            log_event("ROUTER", f"audience comment routing failed: {e}",
                      {"username": username})
            logger.exception("audience comment routing failed")

    asyncio.create_task(_route())
    return {"status": "queued", "username": username, "ts": ts}


@app.post("/api/dashboard_log")
async def api_dashboard_log(payload: dict):
    """Browser-side debug events mirrored into the backend logger so a
    `tail -f backend.log` shows audio/video/Tier 1 lifecycle events in the
    SAME terminal as the pipeline trace lines. Cheap fire-and-forget — the
    dashboard never blocks on this; we just write to logger and return.

    Expected payload shape: {"src": "tier1"|"audio"|"stage", "msg": "...",
                             "data": {<arbitrary kv pairs>}, "trace": "..."}
    `trace` is optional; when present it's prefixed inline so dashboard
    events show up under the same id as the pipeline that triggered them.
    """
    src = str(payload.get("src") or "dashboard")[:20]
    msg = str(payload.get("msg") or "")[:140]
    data = payload.get("data") or {}
    extras = " ".join(
        f"{k}={_dlog_fmt(v)}" for k, v in list(data.items())[:12]
    )
    tid = payload.get("trace") or "-------"
    logger.info("[dash:%s][trace %s] %s %s", src, tid, msg, extras)
    return {"ok": True}


def _dlog_fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)) or v is None:
        return str(v)
    s = str(v)
    if len(s) > 80:
        s = s[:77] + "..."
    return f'"{s}"' if isinstance(v, str) else s


@app.post("/api/go_live")
async def api_go_live():
    """Stage-view G-hotkey target. Plays a generic intro clip ("hey
    everyone, welcome back to the stream") via the Director so the
    avatar starts speaking instantly when the operator triggers Go Live.

    Falls back through the bridge_clips manifest chain:
      1. intro_arbitrary label (added in BRIDGE_SCRIPTS, populated by
         scripts/render_generic_clips.py)
      2. neutral fallback pool
      3. phase0 LatentSync library
      4. None — returns 503 so the operator knows nothing rendered

    Idempotent: spamming G plays back-to-back intros, which the Director's
    crossfade machinery handles cleanly. Useful for nervous demo restarts.
    """
    clip = pick_bridge_clip("intro_arbitrary")
    if not clip:
        # No intros rendered yet — fall through to a neutral acknowledgment
        # so the avatar at least says SOMETHING when the operator presses G.
        clip = pick_bridge_clip("neutral")
    if not clip:
        log_event("DIRECTOR", "go_live: no clips available — render bridges first")
        raise HTTPException(
            status_code=503,
            detail=("No intro/neutral clips on disk. Run "
                    "`python scripts/render_generic_clips.py` first."),
        )
    url = clip.get("url")
    script = clip.get("script", "")
    log_event("DIRECTOR", f"go_live: playing intro — {script[:60]}", {"url": url})

    if director:
        await director.play_response(url)
        # Probe the actual file duration when it's a renders/ MP4 so the
        # idle release fires when the avatar finishes. Bridge clips are
        # short (~2s) so a word-count fallback is tight enough.
        play_ms = (clip.get("ms") or 0)
        if not play_ms:
            words = max(4, len(script.split()))
            play_ms = int(words * 350)
        play_ms_with_tail = play_ms + 400

        async def _release_after(delay_ms: int):
            await asyncio.sleep(delay_ms / 1000)
            if director:
                await director.fade_to_idle()

        asyncio.create_task(_release_after(play_ms_with_tail))

    return {"status": "playing", "url": url, "script": script}


# ── Lip-sync endpoints (Phase 1 pipeline) ──────────────

from fastapi.staticfiles import StaticFiles
app.mount("/renders", StaticFiles(directory=str(RENDER_DIR)), name="renders")

# Static mount for the pre-rendered avatar clip library:
#   - phase0/assets/clips/         (LatentSync bridges/intros/responses)
#   - phase0/assets/clips/idle/    (Veo seamless idle + misc)
#   - phase0/assets/states/        (8s state-pose loops, used as Tier 0 fallback)
# All under one /clips URL so the Director and dashboard see one namespace.
CLIPS_DIR = Path(__file__).resolve().parent.parent / "phase0" / "assets" / "clips"
STATES_DIR = Path(__file__).resolve().parent.parent / "phase0" / "assets" / "states"
CLIPS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/clips", StaticFiles(directory=str(CLIPS_DIR)), name="clips")
app.mount("/states", StaticFiles(directory=str(STATES_DIR)), name="states")

# Pre-rendered local answers — the sub-300ms respond_locally path. Generated
# offline by scripts/render_local_answers.py; missing files fall back to
# escalate_to_cloud gracefully (see _run_respond_locally in this module).
LOCAL_ANSWERS_DIR = Path(__file__).resolve().parent / "local_answers"
LOCAL_ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/local_answers", StaticFiles(directory=str(LOCAL_ANSWERS_DIR)), name="local_answers")

# Audio-first playback (USE_AUDIO_FIRST). Each cloud-bound comment writes
# the TTS bytes here and broadcasts a `comment_response_audio` event the
# moment they're ready — the dashboard plays them immediately while
# Wav2Lip renders video in the background and crossfades in under the
# already-playing audio. Uses uuid filenames so concurrent requests can't
# collide. The dir is gitignored along with backend/renders.
RESPONSE_AUDIO_DIR = Path(__file__).resolve().parent / "response_audio"
RESPONSE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/response_audio", StaticFiles(directory=str(RESPONSE_AUDIO_DIR)), name="response_audio")

# Static assets the dashboard preloads at boot (e.g. silent_unlock.mp3 for
# StartDemoOverlay). Lives under backend/static so it can ship with the
# backend deploy and not need a separate CDN.
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _save_render(label: str, data: bytes) -> str:
    fname = f"{label}_{int(time.time())}.mp4"
    path = RENDER_DIR / fname
    path.write_bytes(data)
    return f"/renders/{fname}"


def _save_response_audio(audio_bytes: bytes) -> str:
    """Save TTS audio to /response_audio/<uuid>.mp3 and return the
    served URL. Caller is responsible for measuring duration via ffprobe
    before broadcasting if accurate timing is needed."""
    import uuid as _uuid
    fname = f"resp_{_uuid.uuid4().hex[:12]}.mp3"
    path = RESPONSE_AUDIO_DIR / fname
    path.write_bytes(audio_bytes)
    return f"/response_audio/{fname}"


def _probe_video_duration_ms(path: Path) -> int | None:
    """ffprobe a local mp4 for its video stream duration in ms.
    Returns None if probe fails; caller should fall back to an estimate."""
    import subprocess
    if not path.exists():
        return None
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=duration", "-of",
             "default=nokey=1:noprint_wrappers=1", str(path)],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode != 0:
            return None
        return int(float(r.stdout.strip()) * 1000)
    except Exception:
        return None


@app.post("/api/generate_pitch")
async def api_generate_pitch(
    text: str = Form(...),
    inference_steps: int = Form(10),
    out_height: int = Form(1080),
):
    """Render a high-quality pitch video via LatentSync (slow, one-time per pitch)."""
    log_event("SELLER", f"generate_pitch: TTS + LatentSync render ({len(text)} chars)")
    t0 = time.time()
    audio_bytes = await text_to_speech(text)
    tts_ms = int((time.time() - t0) * 1000)
    log_event("SELLER", f"TTS ready ({tts_ms}ms, {len(audio_bytes)}B)")

    t0 = time.time()
    video_bytes, headers = await render_pitch_latentsync(
        audio_bytes, inference_steps=inference_steps, out_height=out_height
    )
    render_ms = int((time.time() - t0) * 1000)
    url = _save_render("pitch", video_bytes)
    pipeline_state["pitch_video_url"] = url
    log_event("SELLER", f"LatentSync pitch rendered ({render_ms}ms)", {"url": url})

    await broadcast_to_dashboards({
        "type": "pitch_video",
        "url": url,
        "render_ms": render_ms,
        "tts_ms": tts_ms,
        "backend": "latentsync",
    })
    return {
        "url": url,
        "render_ms": render_ms,
        "tts_ms": tts_ms,
        "pipeline_seconds": headers.get("x-pipeline-seconds"),
    }


@app.post("/api/classify_comment")
async def api_classify_comment(comment: str = Form(...)):
    """Lightweight P1.5: classify a comment via on-device Gemma; if a generic
    pre-rendered clip exists for that label, return its URL so the client can
    play it instantly while the full Wav2Lip render proceeds in parallel."""
    t0 = time.time()
    try:
        result = await classify_comment_gemma(comment)
    except Exception as e:
        result = {"type": "question", "source": "fallback", "error": str(e)}
    label = (result.get("type") or "question").lower()
    elapsed_ms = int((time.time() - t0) * 1000)
    bridge = pick_bridge_clip(label)
    return {
        "comment": comment,
        "label": label,
        "classify_ms": elapsed_ms,
        "source": result.get("source"),
        "bridge_clip_url": bridge.get("url") if bridge else None,
        "bridge_clip_script": bridge.get("script") if bridge else None,
        "draft_response": result.get("draft_response"),
    }


@app.get("/api/bridges")
async def api_bridges():
    """Inspect what bridge clips are loaded (for debugging the manifest)."""
    return all_bridges()


@app.post("/api/respond_to_comment")
async def api_respond_to_comment(
    comment: str = Form(...),
    out_height: int = Form(1920),
):
    """LIVE comment response on the cloud-escalate path. Pristine quality
    over precise lip-sync (Wav2Lip's 96px patch + paste-back produces
    visible artifacts even with GFPGAN at max settings).

    Flow:
      1. _ensure_reading_chat_visible: reading_chat clip already on Tier 1
         from run_routed_comment, OR fired now as fallback for direct
         REST callers. Loops until step 6 fades it out.
      2. classify + LLM + Bedrock + ElevenLabs TTS (with real
         per-character word timings via /v1/text-to-speech/.../with-
         timestamps) — runs while reading_chat is visible.
      3. _release_reading_chat: fade Tier 1 → Tier 0 idle, 300ms beat.
         Honors READING_CHAT_MIN_VISIBLE_MS so a fast/cached pipeline
         doesn't blink past the audience.
      4. Broadcast comment_response_audio: standalone <audio> element
         on the dashboard plays the TTS, KaraokeCaptions reveals each
         word using ElevenLabs character alignments.
      5. Director.emit Tier 1 with intent="response", muted=True, looping
         /states/idle/idle_calm_speaking.mp4 for the duration of the
         audio. Speaking-pose substrate has continuous mouth motion (not
         word-precise but visually engaged) and ZERO Wav2Lip artifacts.
      6. Broadcast SYNTHETIC comment_response_video with url=None +
         audio_already_playing=True. Restores the 4 dashboard contracts
         the previous video event drove (pendingComments clear, voice
         state pill clear, ChatPanel new row, floating comment overlay).
      7. Schedule fade_to_idle for after the audio duration + 400ms tail.

    Pre-rendered qa_index entries take a different path
    (_run_respond_locally) and DO get true lip-sync via LatentSync — only
    this novel-question cloud-escalate path skips lip-sync.

    Total perceived: ~5-8s comment-arrival → first audio syllable
    (8s on Cactus CPU prefill classify is the dominant bottleneck;
    classify and TTS overlap with reading_chat visibility).
    """
    product_data = pipeline_state.get("product_data") or {}
    total_t0 = time.time()

    # If the caller didn't already mint a trace (e.g. direct REST hit on
    # /api/respond_to_comment without going through run_routed_comment),
    # mint one now so phase markers below have an id.
    if trace.get_id() == "-------":
        trace.new_trace("respond_to_comment")
        trace.phase("comment_received", text=comment)

    # 1) Reading visual — make sure it's on screen NOW. run_routed_comment
    #    fires it pre-classify; for direct REST callers we fire it here as
    #    a fallback. The clip loops, so it stays visible until we explicitly
    #    fade it out after the render is done (see step 7).
    reading_t0 = time.time()
    await _ensure_reading_chat_visible()

    # 2) Classify + LLM in parallel with the reading visual.
    class_t0 = time.time()
    classify_task = asyncio.create_task(classify_comment_gemma(comment))

    t0 = time.time()
    try:
        response_text = await generate_comment_response(comment, product_data, "question")
    except Exception:
        response_text = "Great question — let me get back to you on that."
    resp_ms = int((time.time() - t0) * 1000)
    trace.phase("llm_done", ms=resp_ms, words=len(response_text.split()),
                text=response_text)
    log_event("SELLER", f"response drafted ({resp_ms}ms)", {"response": response_text})
    pipeline_state["last_response_text"] = {"comment": comment, "response": response_text}

    # 3) Collect classify result (still useful for telemetry / future hooks).
    comment_type = "question"
    class_ms = int((time.time() - class_t0) * 1000)
    if classify_task.done():
        try:
            comment_type = (classify_task.result() or {}).get("type", "question")
        except Exception:
            pass
    else:
        classify_task.cancel()
    trace.phase("classify_collected", type=comment_type, ms=class_ms)
    log_event("EYES", f'Comment "{comment[:40]}" → {comment_type} ({class_ms}ms)')

    # 4) TTS — runs in parallel with the rest of the reading_chat hold.
    t0 = time.time()
    audio_bytes, word_timings = await text_to_speech(
        response_text, return_word_timings=True,
    )
    tts_ms = int((time.time() - t0) * 1000)
    trace.phase("tts_done", ms=tts_ms, audio_bytes=len(audio_bytes),
                words=len(word_timings))

    # 5) NO Wav2Lip render. Wav2Lip's 96px-patch + paste-back architecture
    #    produces visible mouth/face artifacts even with GFPGAN at maximum
    #    settings (weight=0.9, mouth-only, post-sharpen). The user's quality
    #    bar is "pristine, no compromises" — so we skip live lip-sync
    #    entirely on novel questions and instead play a clean speaking-pose
    #    idle loop on Tier 1 while the response audio + KaraokeCaptions
    #    carry the actual answer. Pre-rendered qa_index entries (LatentSync)
    #    still get true lip-sync via _run_respond_locally — only this
    #    cloud-escalate path skips it. Trade-off: novel responses don't
    #    show precise lip-sync but show ZERO Wav2Lip artifacts.
    audio_url: str | None = None
    audio_duration_ms: int | None = None

    # 7a) Audio-first + speaking-idle loop = pristine quality novel response.
    if audio_bytes:
        from agents.seller import _probe_audio_duration_ms
        audio_url = _save_response_audio(audio_bytes)
        audio_duration_ms = await asyncio.to_thread(
            _probe_audio_duration_ms, audio_bytes,
        )
        total_ms = int((time.time() - total_t0) * 1000)
        trace.phase("audio_only_path_chosen",
                    url=audio_url, audio_dur_ms=audio_duration_ms,
                    reason="wav2lip_skipped_for_quality")

        # Release reading → 300ms idle pause → speaking-idle + audio.
        await _release_reading_chat(reading_t0)

        # Dispatch standalone audio (powers KaraokeCaptions word reveal).
        await broadcast_to_dashboards({
            "type": "comment_response_audio",
            "comment": comment,
            "response": response_text,
            "url": audio_url,
            "word_timings": word_timings,
            "expected_duration_ms": audio_duration_ms,
            "intent": "response",
            "class_ms": class_ms,
            "llm_ms": resp_ms,
            "tts_ms": tts_ms,
            "ts": time.time_ns(),
        })
        trace.phase("audio_dispatched", url=audio_url,
                    audio_dur_ms=audio_duration_ms)

        # Loop a clean speaking-pose substrate on Tier 1 for the duration
        # of the audio. The substrate is a continuous "talking pose"
        # (mouth in motion, generic phonemes) — not synced to specific
        # words but visually engaged + ZERO Wav2Lip artifacts. The loop
        # naturally extends past the audio if needed; we fade it out
        # explicitly when audio ends.
        #
        # intent="response" (NOT "response_speaking_idle") so the dashboard's
        # LiveStage.overlayVisible flag activates and the floating comment
        # card with latency badge appears, exactly as it does for local
        # and canned-clip responses.
        SPEAKING_IDLE_URL = "/states/idle/idle_calm_speaking.mp4"
        if director:
            await director.emit(
                "tier1",
                "response",
                SPEAKING_IDLE_URL,
                loop=True,
                mode="crossfade",
                muted=True,  # audio comes from the standalone <audio>
                expected_duration_ms=audio_duration_ms,
                emitted_by="api_respond_to_comment_no_wav2lip",
            )
            trace.phase("speaking_idle_loop_playing", url=SPEAKING_IDLE_URL)

            release_ms = (audio_duration_ms or 0) + 400
            if release_ms < 2500:
                release_ms = 2500 + 400

            async def _release_after(delay_ms: int):
                await asyncio.sleep(delay_ms / 1000)
                if director:
                    await director.fade_to_idle()
            asyncio.create_task(_release_after(release_ms))

        # Synthetic comment_response_video event — restores the dashboard
        # contracts that the previous "comment_response_video always follows
        # comment_response_audio on the cloud path" assumption depended on:
        #   • setPendingComments filter (clears the pending chip)
        #   • setVoiceStateSafe(null) (drops the voice state pill)
        #   • setCommentResponse / setResponseVideo (powers ChatPanel + the
        #     floating comment overlay on /stage)
        # url=None tells consumers there's no video file to load — the
        # response visual is the speaking-idle loop emitted above via
        # play_clip, and the audio is the standalone <audio> element.
        # audio_already_playing=true matches the existing audio-first
        # contract; lip_synced=false marks this as the no-Wav2Lip path
        # for any downstream observer that wants to differentiate.
        await broadcast_to_dashboards({
            "type": "comment_response_video",
            "comment": comment,
            "response": response_text,
            "url": None,
            "total_ms": total_ms,
            "class_ms": class_ms,
            "llm_ms": resp_ms,
            "tts_ms": tts_ms,
            "lipsync_ms": 0,
            "audio_already_playing": True,
            "existing_audio_url": audio_url,
            "expected_duration_ms": audio_duration_ms,
            "lip_synced": False,
        })
        trace.phase("synthetic_response_video_dispatched", url=None)
        trace.summary("respond_to_comment.audio_with_speaking_idle",
                      total_ms=total_ms, audio=True, video=False,
                      karaoke=True, lip_synced=False,
                      reason="wav2lip_skipped_pristine_quality")
        return {
            "comment": comment,
            "response": response_text,
            "url": None,
            "total_ms": total_ms,
            "audio_url": audio_url,
            "audio_duration_ms": audio_duration_ms,
            "audio_first": True,
            "lip_synced": False,
            "breakdown": {
                "classify_ms": class_ms,
                "llm_ms": resp_ms,
                "tts_ms": tts_ms,
            },
        }

    # 7b) Last-resort: TTS itself failed (audio_bytes is empty/falsy).
    # Release reading + tell the dashboard the comment couldn't be
    # answered so it clears the pending chip; caller (run_routed_comment)
    # surfaces this as a failure.
    await _release_reading_chat(reading_t0)
    total_ms = int((time.time() - total_t0) * 1000)
    trace.summary("respond_to_comment.tts_failed",
                  total_ms=total_ms, audio=False, video=False,
                  reason="tts_returned_empty")
    await broadcast_to_dashboards({
        "type": "comment_failed",
        "comment": comment,
        "response": response_text,
        "reason": "tts_returned_empty",
    })
    return {
        "comment": comment,
        "response": response_text,
        "url": None,
        "total_ms": total_ms,
        "audio_first": False,
        "lip_synced": False,
        "failed": True,
    }


# ── Removed dead code (Apr 2026) ─────────────────────────────────────────
# The previous Wav2Lip-based comment-response pipeline lived here:
#   _render_response_video(audio_bytes, out_height)
#       Wav2Lip render + USE_LIVE_PAD audio/video alignment helper.
#   _render_and_broadcast_video(...)
#       Audio-first background-render coroutine that ran Wav2Lip in the
#       background after standalone audio dispatch.
# Both are deleted because api_respond_to_comment switched to a "no-Wav2Lip
# for novel cloud responses" model (audio + KaraokeCaptions over a clean
# speaking-pose loop) — the user's quality bar of "pristine, no compromises"
# rules out Wav2Lip's 96px-patch artifacts even with GFPGAN at max settings.
# _run_wav2lip_pitch is the only remaining live caller of the Wav2Lip server
# (legacy USE_PITCH_VEO=0 kill switch path); it calls
# render_comment_response_wav2lip directly so doesn't need _render_response_
# video. If a future caller wants a re-usable Wav2Lip helper, restore from
# git history at sha 2c98beb.


async def _fire_speculative_bridge(comment: str) -> None:
    """Play a short label-agnostic ack on Tier 1 the moment a transcript
    lands, BEFORE the router decides. Bought time fills the ~600-800ms
    gap between transcript broadcast and the response Tier 1 emit on
    cloud-escalate paths.

    Skipped when:
      • Director not yet ready (boot race)
      • Comment likely matches a local qa_index entry → respond_locally
        will fire in <100ms anyway and the bridge would only render
        ~200ms of itself before being preempted
      • No bridge clip available for the 'neutral' label
    """
    if director is None:
        return
    # Cheap pre-check: does this comment look like a local qa hit? Same
    # matcher the router uses, so we agree with the routing decision that's
    # about to fire 50-150ms from now.
    product = pipeline_state.get("product_data") or {}
    if _match_product_field(comment, product):
        logger.debug("[bridge] speculative skip — likely local match")
        return
    try:
        # 'neutral' pool is ~6 short utterances ("okay", "mhm", "right, so...")
        # that are safe to play before any routing decision is final.
        clip = await director.play_bridge("neutral")
        if clip:
            logger.info("[bridge] speculative bridge fired (clip=%s)",
                        Path(clip.get("url", "?")).name)
    except Exception:
        logger.exception("[bridge] speculative bridge failed (non-fatal)")


# ── Voice comment pipeline ─────────────────────────────────────────────────
#
# Hour 2-3 scaffolding: one new endpoint + a one-function stub router.
#
#   POST /api/voice_comment
#     accepts an audio blob (webm/opus from the dashboard MediaRecorder,
#     .wav from curl smoke tests), transcribes on-device via whisper-base on
#     Cactus (fallback: Gemini 2.5 Flash), broadcasts voice_transcript to
#     the dashboard, then hands the transcript to run_routed_comment().
#
#   run_routed_comment(comment)
#     STUB. Forwards every comment to the existing /api/respond_to_comment
#     cloud pipeline. Hour 4-5 replaces this with a FunctionGemma-driven
#     dispatcher that picks among four tools (respond_locally,
#     escalate_to_cloud, play_canned_clip, block_comment). Keeping the
#     signature stable means /api/voice_comment never needs to change.


async def run_routed_comment(comment: str) -> dict:
    """Route an incoming comment through one of four tools.

    Flow:
      0. Fire reading_chat IMMEDIATELY for sub-100ms visual feedback.
      1. Gemma 4 classify on device (already lock-serialized).
      2. router.decide → {tool, args, reason, ms, was_local, cost_saved_usd}.
      3. Broadcast routing_decision WS event for the RoutingPanel.
      4. Dispatch to the matching _run_* helper. On any downstream failure,
         the helpers fade Tier 1 back to idle and broadcast comment_failed
         with the best available fallback text (drafted by Claude before
         the failure, when applicable) so the dashboard never sticks."""
    # Mint a fresh trace id so this comment's full lifecycle (classify →
    # router → dispatch → reading_chat → TTS → wav2lip → response) shows up
    # under one greppable id in the backend log. Every helper called from
    # here inherits the trace id automatically via contextvars.
    trace.new_trace("comment")
    trace.phase("comment_received", text=comment)

    # 0. Fire reading_chat NOW — before classify/router/anything. The
    #    on-device Cactus classify can take 5-10s on CPU prefill; without
    #    this immediate emit, the user sees nothing happen on stage for
    #    that whole window, then a sudden burst of activity. By emitting
    #    the reading clip first, the avatar visibly "engages" within ~50ms
    #    of the comment arriving and stays in the reading pose for the
    #    natural duration of the rest of the pipeline. The clip loops so
    #    it just keeps showing until a downstream helper fades it out.
    if director is not None:
        try:
            await director.emit_reading_chat()
            _reading_already_fired.set(True)
            trace.phase("reading_chat_emitted_immediate")
        except Exception:
            logger.exception("[director] emit_reading_chat raised (non-fatal)")

    product = pipeline_state.get("product_data")

    # 1. Classify (on-device Gemma 4). Never raises — returns a dict with
    #    at least {type, source} even on fallback. Safe to await here.
    classify = await classify_comment_gemma(comment)
    trace.phase("classify_done",
                type=classify.get("type"), source=classify.get("source"))

    # 2. Decide.
    decision = await comment_router.decide(comment, classify, product)
    trace.phase("router_decided",
                tool=decision["tool"], reason=decision["reason"],
                local=decision["was_local"], ms=decision["ms"])

    # 3. Broadcast so RoutingPanel (Hour 7) can tick counters.
    await broadcast_to_dashboards({
        "type": "routing_decision",
        "comment": comment,
        "tool": decision["tool"],
        "reason": decision["reason"],
        "ms": decision["ms"],
        "was_local": decision["was_local"],
        "cost_saved_usd": decision["cost_saved_usd"],
    })
    log_event("ROUTER", f'{decision["tool"]} — {decision["reason"]} ({decision["ms"]}ms)',
              {"classify": classify.get("type"), "was_local": decision["was_local"]})

    # 4. Dispatch.
    tool = decision["tool"]
    args = decision["args"]
    if tool == "respond_locally":
        return await _run_respond_locally(comment, args, decision)
    if tool == "play_canned_clip":
        return await _run_play_canned_clip(comment, args, decision)
    if tool == "block_comment":
        return await _run_block_comment(comment, args, decision)
    # Default: escalate_to_cloud. Same pattern as the old stub — forward to
    # the existing api_respond_to_comment, and on failure fade + broadcast
    # comment_failed with any drafted text.
    #
    # Note: there's intentionally no pitch dispatch case here. Pitches
    # fire from the video-upload pipeline (run_sell_pipeline → the
    # _run_audio_first_pitch helper → Director.dispatch_audio_first_pitch).
    # Comments are audience reactions only.
    return await _run_escalate_to_cloud(comment, args, decision)


async def _run_escalate_to_cloud(comment: str, args: dict, decision: dict) -> dict:
    try:
        result = await api_respond_to_comment(comment=comment, out_height=1920)
        return {"dispatch": "escalate_to_cloud", "routing": decision, **result}
    except Exception as e:
        if director:
            try:
                await director.fade_to_idle()
            except Exception:
                logger.exception("fade_to_idle after failure also failed")
        last = pipeline_state.get("last_response_text") or {}
        fallback_text = last.get("response") if last.get("comment") == comment else None
        await broadcast_to_dashboards({
            "type": "comment_failed",
            "comment": comment,
            "response": fallback_text or "",
            "reason": str(e)[:200],
        })
        raise


# Per-task contextvar tracking whether reading_chat was already emitted by
# an upstream caller (run_routed_comment). Lets dispatch helpers skip a
# duplicate emit (which would re-crossfade the same clip and look like a
# visible flash). Defaults to False, so any helper called outside the
# routed-comment path (direct REST hit, voice, etc.) still fires the
# reading clip itself as a fallback.
_reading_already_fired: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "reading_already_fired", default=False
)


async def _ensure_reading_chat_visible() -> None:
    """Make sure the reading_chat clip is on Tier 1 right now. No-op if
    run_routed_comment already fired it; otherwise fires it (slow callers
    that bypass the routed path still get a reading visual).

    Used by dispatch helpers right before they spend time generating the
    response — guarantees the avatar is visibly engaged while we work."""
    if director is None:
        return
    if _reading_already_fired.get():
        return
    try:
        await director.emit_reading_chat()
        _reading_already_fired.set(True)
        trace.phase("reading_chat_emitted_late")
    except Exception:
        logger.exception("[director] emit_reading_chat raised (non-fatal)")


# Minimum visible duration of the reading_chat clip. If the entire response
# pipeline (LLM + TTS + Wav2Lip) finishes faster than this, we hold the
# reading visible for the remainder so the audience actually sees her read.
# 1.5s is enough to register the pose change without feeling artificial.
READING_CHAT_MIN_VISIBLE_MS = 1500


async def _release_reading_chat(reading_started_at: float) -> None:
    """Fade the reading_chat clip out and run the brief 'returning to
    listening' idle pause. Honors a minimum visible duration so a fast
    pipeline doesn't blink the reading state past the audience.

    Called by dispatch helpers AFTER they've finished generating the
    response (TTS done, Wav2Lip rendered, etc.) — at this point the
    response is ready to play and we just need to gracefully transition
    out of reading first."""
    if director is None:
        return
    elapsed_ms = int((time.time() - reading_started_at) * 1000)
    if elapsed_ms < READING_CHAT_MIN_VISIBLE_MS:
        wait_ms = READING_CHAT_MIN_VISIBLE_MS - elapsed_ms
        trace.phase("reading_chat_min_visible_pad", wait_ms=wait_ms,
                    elapsed_ms=elapsed_ms)
        await asyncio.sleep(wait_ms / 1000)
    try:
        await director.fade_to_idle()
    except Exception:
        logger.exception("[director] fade_to_idle raised (non-fatal)")
    trace.phase("faded_to_idle")
    await asyncio.sleep(0.3)
    trace.phase("idle_pause_done", ms=300)


async def _run_respond_locally(comment: str, args: dict, decision: dict) -> dict:
    """Play a pre-rendered answer clip via the Director. If the answer file
    isn't present on disk, fall back to escalate_to_cloud so the demo never
    shows dead air."""
    product = pipeline_state.get("product_data") or {}
    qa = product.get("qa_index") or {}
    entry = qa.get(args.get("answer_id", "")) or {}
    url = entry.get("url")
    text = entry.get("text") or ""

    if not url:
        logger.warning("[router] respond_locally with no URL (answer_id=%r) — escalating",
                       args.get("answer_id"))
        return await _run_escalate_to_cloud(comment, {"comment": comment}, decision)

    # Verify the file exists — pre-render script may not have shipped this
    # answer yet. Static mount is /local_answers → backend/local_answers/<slug>.mp4.
    if url.startswith("/local_answers/"):
        file_path = Path(__file__).resolve().parent / "local_answers" / Path(url).name
        if not file_path.exists():
            logger.warning("[router] respond_locally file missing: %s — escalating", file_path)
            return await _run_escalate_to_cloud(comment, {"comment": comment}, decision)

    # Make sure reading is visible (no-op if upstream already fired it),
    # then immediately release + play. The classify+router lag upstream
    # already gave the user a few seconds of reading visual.
    reading_t0 = time.time()
    await _ensure_reading_chat_visible()
    await _release_reading_chat(reading_t0)

    # Stage the response via the Director so Tier 0/Tier 1 layering stays
    # intact. Fade to idle after the clip plays — we don't have ffprobe
    # probing on pre-rendered clips, so use a word-count estimate with a
    # small tail. Clips are short (1-2 sentences) so word count is fine.
    if director:
        await director.play_response(url)
        trace.phase("local_clip_playing", url=url, words=len(text.split()))
        words = max(4, len(text.split()))
        play_ms = int(words * 350) + 400

        async def _release_after(delay_ms: int):
            await asyncio.sleep(delay_ms / 1000)
            await director.fade_to_idle()
        asyncio.create_task(_release_after(play_ms))

    await broadcast_to_dashboards({
        "type": "comment_response_video",
        "comment": comment,
        "response": text,
        "url": url,
        "total_ms": decision["ms"],   # near-zero: no render, no TTS
        "class_ms": 0,
        "llm_ms": 0,
        "tts_ms": 0,
        "lipsync_ms": 0,
        "local": True,
    })
    return {"dispatch": "respond_locally", "routing": decision,
            "comment": comment, "response": text, "url": url,
            "total_ms": decision["ms"]}


async def _run_play_canned_clip(comment: str, args: dict, decision: dict) -> dict:
    """Acknowledge with a pre-rendered bridge clip (compliment / objection /
    neutral). Uses pick_bridge_clip which is manifest-driven and shared with
    the escalate_to_cloud path's bridge."""
    label = args.get("label", "neutral")
    clip = pick_bridge_clip(label)
    if not clip:
        logger.warning("[router] no bridge for label=%s — escalating", label)
        return await _run_escalate_to_cloud(comment, {"comment": comment}, decision)

    url = clip.get("url")
    script = clip.get("script", "")

    # Same pattern as respond_locally — ensure reading visible, release.
    reading_t0 = time.time()
    await _ensure_reading_chat_visible()
    await _release_reading_chat(reading_t0)

    if director:
        await director.play_response(url)
        trace.phase("canned_clip_playing", label=label, url=url)
        # Bridge clips are short (~2s). Fade after clip + tail.
        play_ms = (clip.get("ms") or 2500) + 400

        async def _release_after(delay_ms: int):
            await asyncio.sleep(delay_ms / 1000)
            await director.fade_to_idle()
        asyncio.create_task(_release_after(play_ms))

    await broadcast_to_dashboards({
        "type": "comment_response_video",
        "comment": comment,
        "response": script,
        "url": url,
        "total_ms": decision["ms"],
        "class_ms": 0,
        "llm_ms": 0,
        "tts_ms": 0,
        "lipsync_ms": 0,
        "local": True,
    })
    return {"dispatch": "play_canned_clip", "routing": decision,
            "comment": comment, "label": label, "url": url,
            "response": script, "total_ms": decision["ms"]}


async def _run_block_comment(comment: str, args: dict, decision: dict) -> dict:
    """Spam/abuse — increment counter, broadcast for the panel, no visual."""
    pipeline_state.setdefault("blocked_count", 0)
    pipeline_state["blocked_count"] += 1
    await broadcast_to_dashboards({
        "type": "comment_blocked",
        "comment": comment,
        "reason": args.get("reason", "spam"),
        "blocked_count": pipeline_state["blocked_count"],
    })
    return {"dispatch": "block_comment", "routing": decision,
            "comment": comment, "reason": args.get("reason", "spam"),
            "blocked_count": pipeline_state["blocked_count"],
            "total_ms": decision["ms"]}


@app.post("/api/voice_comment")
async def api_voice_comment(audio: UploadFile = File(...)):
    """Voice-driven comment. Transcribes on-device, broadcasts the transcript
    immediately (so the dashboard can show what was heard with a latency
    chip), then routes the comment through the standard pipeline.

    Returns a single JSON with {transcript, dispatch, total_ms, ...response}.
    The dashboard also receives two WS events along the way: voice_transcript
    (after transcribe) and comment_response_video (after render)."""
    t0 = time.time()
    audio_bytes = await audio.read()
    if not audio_bytes:
        return {"error": "empty_audio", "total_ms": 0}

    # (1) Transcribe — local whisper first, Gemini fallback. transcribe_voice
    #     never raises; on total failure it returns source='transcription_failed'.
    trans = await transcribe_voice(audio_bytes)
    await broadcast_to_dashboards({
        "type": "voice_transcript",
        "text": trans.get("text", ""),
        "source": trans.get("source", "unknown"),
        "ms": trans.get("latency_ms", 0),
    })
    log_event("EYES", f'Voice: "{trans.get("text", "")}" '
              f'via {trans.get("source", "?")} ({trans.get("latency_ms", 0)}ms)')
    logger.info('[voice] "%s" via %s in %dms',
                trans.get("text", ""), trans.get("source", "?"),
                trans.get("latency_ms", 0))

    # Bail if we got nothing intelligible — no point firing the cloud path.
    text = (trans.get("text") or "").strip()
    if not text:
        return {
            "transcript": trans,
            "dispatch": "no_speech",
            "total_ms": int((time.time() - t0) * 1000),
        }

    # (1.5) Speculative bridge — fires the moment we have a transcript, BEFORE
    #       the router decides. Reads as "she heard you and is responding"
    #       which closes the gap between transcript landing and the actual
    #       response Tier 1 emit. Skipped on likely-local matches because
    #       respond_locally is already <100ms and the bridge would only get
    #       cut off mid-frame.
    if USE_SPECULATIVE_BRIDGE:
        asyncio.create_task(_fire_speculative_bridge(text))

    # (2) Route + render. Today = always escalate. Hour 4-5 adds real routing.
    try:
        routed = await run_routed_comment(text)
    except Exception as e:
        logger.exception("run_routed_comment failed")
        return {
            "transcript": trans,
            "dispatch": "error",
            "error": str(e),
            "total_ms": int((time.time() - t0) * 1000),
        }

    routed["transcript"] = trans
    routed["total_ms"] = int((time.time() - t0) * 1000)
    return routed


@app.post("/api/build_carousel")
async def api_build_carousel(
    file: UploadFile = File(...),
    n_frames: int = Form(36),
    out_size: int = Form(1024),
    clean_bg: bool = Form(True),
    rembg_model: str = Form("u2net"),
    stabilize: bool = Form(True),
    remove_skin: bool = Form(False),
    keep_central: bool = Form(True),
):
    """Build a 3D-spin carousel from an uploaded video. Tweaks exposed for
    local debugging — defaults match the production pipeline."""
    import tempfile
    contents = await file.read()
    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(contents)
        video_path = f.name
    try:
        view = await carousel_from_video(
            video_path, n_frames=n_frames, out_size=out_size,
            clean_bg=clean_bg, rembg_model=rembg_model, stabilize=stabilize,
            remove_skin=remove_skin, keep_central=keep_central,
        )
    finally:
        Path(video_path).unlink(missing_ok=True)
    pipeline_state["view_3d"] = view
    await broadcast_to_dashboards({"type": "view_3d", "data": view})
    return view


@app.get("/api/view_3d")
async def api_view_3d():
    return pipeline_state.get("view_3d") or {"kind": None}


@app.get("/api/photo")
async def api_photo():
    from fastapi.responses import Response
    b64 = pipeline_state.get("product_clean_b64", "")
    if not b64:
        return {"error": "no photo"}
    return Response(content=base64.b64decode(b64), media_type="image/png")


# ── Run ────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=True)
