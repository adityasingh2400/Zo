import asyncio
import base64
import json
import logging
import os
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("empire")

from agents import brain
from agents import router as comment_router
from agents import translator
from agents.hands import Hands
from agents.avatar_director import Director
from agents.bridge_clips import all_bridges, pick_bridge_clip
from agents.creator import build_all as creator_build_all
from agents.creator import generate_3d_model, remove_background
from agents.eyes import (
    analyze_and_script_claude,
    analyze_with_claude,
    classify_comment_gemma,
    transcribe_voice,
)
from agents.intake import process_video
from agents.router import _match_product_field  # used by speculative bridge
from agents.seller import (
    generate_comment_response,
    make_avatar_speak,
    render_comment_response_wav2lip,
    render_pitch_latentsync,
    set_livetalking_session,
    text_to_speech,
)
from agents.threed import carousel_from_video
from config import (
    BACKEND_HOST,
    BACKEND_PORT,
    LIPSYNC_PROVIDER,
    USE_AUDIO_FIRST,
    USE_BACKCHANNEL,
    USE_KARAOKE,
    USE_PITCH_VEO,
    USE_SPECULATIVE_BRIDGE,
)

logger.info(
    "[flags] USE_AUDIO_FIRST=%s USE_KARAOKE=%s USE_PITCH_VEO=%s "
    "USE_BACKCHANNEL=%s USE_SPECULATIVE_BRIDGE=%s LIPSYNC_PROVIDER=%s",
    USE_AUDIO_FIRST, USE_KARAOKE, USE_PITCH_VEO,
    USE_BACKCHANNEL, USE_SPECULATIVE_BRIDGE, LIPSYNC_PROVIDER,
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
hands: "Hands | None" = None        # set in app startup (Item 5)


PRODUCTS_PATH = Path(__file__).resolve().parent / "data" / "products.json"
AVATARS_PATH = Path(__file__).resolve().parent / "data" / "avatars.json"


def _load_products() -> None:
    """Read backend/data/products.json into pipeline_state["products_catalog"]
    (full dict) + set the initial active product. The active product lives in
    pipeline_state["product_data"] (kept in sync with active_product_id) so
    the router's respond_locally path can match against its qa_index without
    requiring a /api/sell upload first.

    Selection order for initial active:
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

    pipeline_state["products_catalog"] = products
    initial = os.getenv("ACTIVE_PRODUCT_ID") or next(iter(products.keys()))
    if initial not in products:
        logger.warning("ACTIVE_PRODUCT_ID=%s not in products.json — falling back", initial)
        initial = next(iter(products.keys()))
    _set_active_product(initial)
    logger.info("[products] Loaded %d products: %s", len(products), list(products.keys()))


def _load_avatars() -> None:
    """Load backend/data/avatars.json into pipeline_state['avatars_catalog'].
    Analogous to _load_products. Missing file / bad JSON: log + skip; the
    system falls back to the single-avatar path (Maya's state videos are
    the current hardcoded defaults in avatar_director.py)."""
    if not AVATARS_PATH.exists():
        logger.info("No avatars.json at %s — skipping pre-load", AVATARS_PATH)
        return
    try:
        with AVATARS_PATH.open() as f:
            avatars = json.load(f)
    except Exception as e:
        logger.warning("Failed to read avatars.json: %s", e)
        return
    if not isinstance(avatars, dict) or not avatars:
        return

    pipeline_state["avatars_catalog"] = avatars
    initial = os.getenv("ACTIVE_AVATAR_ID") or next(iter(avatars.keys()))
    if initial not in avatars:
        initial = next(iter(avatars.keys()))
    pipeline_state["active_avatar_id"] = initial
    logger.info("[avatars] Loaded %d avatars: %s (active=%s)",
                len(avatars), list(avatars.keys()), initial)


def _active_avatar() -> dict:
    """Resolve the currently-active avatar dict. Returns {} if the catalog
    is empty (pre-lifespan or no avatars.json), signaling callers to fall
    back to their legacy hardcoded paths."""
    catalog = pipeline_state.get("avatars_catalog") or {}
    active_id = pipeline_state.get("active_avatar_id")
    if active_id and active_id in catalog:
        return catalog[active_id]
    return {}


def _set_active_product(product_id: str) -> dict | None:
    """Switch the currently-active product. Updates active_product_id +
    keeps the legacy product_data accessor in sync so existing routes
    (router, /api/state, agents that read product_data) don't need to change.
    Returns the product dict on success, None if product_id isn't loaded."""
    catalog = pipeline_state.get("products_catalog") or {}
    product = catalog.get(product_id)
    if not isinstance(product, dict) or not product:
        logger.warning("[products] _set_active_product: %r not in catalog", product_id)
        return None
    pipeline_state["active_product_id"] = product_id
    pipeline_state["product_data"] = product
    qa_count = len(product.get("qa_index") or {})
    logger.info('[products] Active product → "%s" (id=%s, %d Q/A entries)',
                product.get("name", "?"), product_id, qa_count)
    return product


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


# ── App ────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global director, hands
    print(f"EMPIRE backend running on {BACKEND_HOST}:{BACKEND_PORT}")
    # Bring up the Hands agent (Item 5). Idempotent — reuses the same
    # broadcast_to_dashboards callable the Director uses for WS emits.
    hands = Hands(broadcast=broadcast_to_dashboards)
    logger.info("[hands] initialized with platforms: %s", list(hands.adapters.keys()))
    # Bring up the Avatar Director early so the first dashboard connect can
    # immediately receive a Tier 0 idle clip.
    director = Director(broadcast_to_dashboards)
    director.start_idle_rotation()
    logger.info("Avatar Director instantiated; idle rotation running.")
    # Pre-load Cactus models in background threads so the first request isn't
    # slow. Gemma 4 (vision + classify + script) and whisper-base (voice
    # transcription) live on separate Cactus handles; we load them
    # sequentially to avoid any re-entrant SDK init on startup.
    from agents.eyes import CACTUS_AVAILABLE, _get_cactus_model, _get_cactus_whisper_model
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
        import asyncio as _aio

        from agents.threed import prewarm_rembg
        _aio.create_task(prewarm_rembg("u2net"))
    except Exception as e:
        logger.warning("rembg prewarm scheduling failed: %s", e)

    # Load demo products + pre-select an active one for respond_locally. The
    # Hour 5-6 scope: the router can answer routine questions without any
    # prior /api/sell call. ACTIVE_PRODUCT_ID env var overrides the first-
    # key default if you want to swap between demo objects without editing
    # code on stage.
    _load_products()
    _load_avatars()
    yield

app = FastAPI(title="EMPIRE", lifespan=lifespan)

# CORS: default to localhost:5173 (dev dashboard). Override via FRONTEND_ORIGIN
# env (comma-separated for multiple origins, e.g. dev + cloudflare tunnel).
# Replaces the prior allow_origins=["*"] which let any origin call the API.
_FRONTEND_ORIGINS = [
    o.strip() for o in os.getenv("FRONTEND_ORIGIN", "http://localhost:5173").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("[cors] allow_origins=%s", _FRONTEND_ORIGINS)


async def _ws_auth_check(ws: WebSocket) -> bool:
    """Validate ?token=<secret> against WS_SHARED_SECRET env. Returns True
    when OK; closes the socket with 1008 (policy violation) on mismatch.

    Default-off: when WS_SHARED_SECRET is unset, all connections are allowed
    (matches the prior unauth behavior so this change doesn't break a fresh
    clone). Set WS_SHARED_SECRET in .env + VITE_WS_TOKEN in the dashboard env
    to enable. Required before exposing the backend on a public network."""
    expected = os.getenv("WS_SHARED_SECRET", "").strip()
    if not expected:
        return True
    provided = ws.query_params.get("token", "")
    if provided != expected:
        await ws.close(code=1008, reason="invalid or missing token")
        logger.warning("[ws] rejected %s connection: bad token", ws.url.path)
        return False
    return True


# ── WebSocket: Phone ───────────────────────────────────

@app.websocket("/ws/phone")
async def phone_ws(ws: WebSocket):
    if not await _ws_auth_check(ws):
        return
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


# ── WebSocket: Dashboard ──────────────────────────────

@app.websocket("/ws/dashboard")
async def dashboard_ws(ws: WebSocket):
    if not await _ws_auth_check(ws):
        return
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
                # resolve in <300ms; cloud tool forwards to the existing
                # api_respond_to_comment Wav2Lip pipeline.
                comment_text = msg.get("text", "")
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

    except WebSocketDisconnect:
        dashboard_clients.remove(ws)


# ── Sell Pipeline ──────────────────────────────────────

async def run_sell_pipeline(frame_b64: str, voice_text: str,
                            request_id: str | None = None):
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
    await _emit_pipeline_step(request_id, "claude", "active")
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
    await _emit_pipeline_step(request_id, "claude", "done", ms=phase1_ms)
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

    # PHASE 2: TTS + Wav2Lip render — real video lipsync via the same fast
    # path the comment responses use. The legacy LiveTalking WebRTC path was
    # removed because it required a session handshake that the dashboard no
    # longer initiates, and its audio-only fallback played MP3 over a silent
    # idle face — read as "gibberish in a random language" because the mouth
    # never moved.
    #
    # New flow:
    #   1. Voice pill → RESPONDING so the audience knows we're rendering.
    #   2. ElevenLabs TTS the Claude-generated script (English-locked).
    #   3. Wav2Lip render against whichever speaking-substrate the Director
    #      thinks matches the visible Tier 0 idle (calm by default).
    #   4. Director crossfades the rendered mp4 onto Tier 1 → settles back
    #      to the silent idle layer when the video duration elapses.
    pipeline_state["status"] = "selling"
    log_event("SELLER", "Avatar going live...")
    if director:
        await director.set_voice_state("responding")

    pitch_t0 = time.time()
    try:
        # 2a. TTS. If the active avatar declares a voice_id, use it;
        # otherwise text_to_speech falls back to ELEVENLABS_VOICE_ID.
        # Item 6: if active_language is non-English, translate the script
        # first + pass language_code to ElevenLabs. Cache hits are cheap;
        # misses incur one Claude Haiku call per unique pitch per language.
        await _emit_pipeline_step(request_id, "eleven", "active")
        t0 = time.time()
        active_voice = _active_avatar().get("voice_id") or None
        active_lang = pipeline_state.get("active_language", "en")
        tts_script = script
        if active_lang != "en":
            tts_script = await translator.translate(script, active_lang)
            if tts_script != script:
                log_event("SELLER", f"Script translated to {active_lang} "
                                    f"({len(script)} → {len(tts_script)} chars)")
        audio_bytes = await text_to_speech(
            tts_script,
            voice=active_voice,
            language_code=active_lang,
        )
        tts_ms = int((time.time() - t0) * 1000)
        if not audio_bytes:
            await _emit_pipeline_step(request_id, "eleven", "failed",
                                       detail="empty audio")
            raise RuntimeError("TTS returned empty audio")
        log_event("SELLER", f"TTS ready ({tts_ms}ms, {len(audio_bytes)}B)")
        await _emit_pipeline_step(request_id, "eleven", "done", ms=tts_ms)

        # 2b. Wav2Lip render against the substrate the Director picked for
        #     whatever Tier 0 is visible. Falls back to default substrate if
        #     the configured one isn't on the pod (the Director caches a
        #     "missing" mark so we don't repeatedly retry the bad path).
        substrate = director.current_substrate_pod_path() if director else None
        await _emit_pipeline_step(request_id, "wav2lip", "active")
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
        await _emit_pipeline_step(request_id, "wav2lip", "done", ms=lipsync_ms)

        # 2c. Save + broadcast through Director so the carousel crossfade
        #     machinery and idle-release timing work the same way as comment
        #     responses (single source of truth for stage state).
        url = _save_render("pitch", video_bytes)
        pipeline_state["pitch_video_url"] = url
        await _emit_pipeline_step(request_id, "going_live", "done",
                                   ms=lipsync_ms + tts_ms + phase1_ms)

        if director:
            await director.play_response(url)
            # Probe rendered duration so the idle release fires precisely as
            # she finishes the pitch. Falls back to a word-count estimate.
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
    except Exception as e:
        logger.exception("[pitch] render failed")
        log_event("SELLER", f"Pitch render failed: {e}")
        # Last-ditch: TTS-only fallback so at least audio plays. This is the
        # legacy "no video" path, kept only for catastrophic Wav2Lip failures.
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
    log_event("SYSTEM", f"EMPIRE is LIVE. Total pipeline: {total_ms}ms")
    logger.info("=" * 60)
    logger.info("SELL PIPELINE COMPLETE — %dms total", total_ms)
    logger.info("=" * 60)
    await broadcast_to_dashboards({"type": "status", "status": "live"})


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


async def _emit_pipeline_step(request_id: str | None, step: str, status: str,
                              ms: int | None = None, detail: str | None = None):
    """Broadcast a pipeline_step event scoped to a request_id so the iPhone's
    PipelineProgressView can render the 'Building your avatar' rail. Called
    from run_video_sell_pipeline + run_sell_pipeline at each major boundary.

    When request_id is None (legacy callers that didn't generate one), this
    is a no-op — backward-compatible with the pre-Phase-1 flow."""
    if not request_id:
        return
    payload = {"type": "pipeline_step", "request_id": request_id,
               "step": step, "status": status}
    if ms is not None:
        payload["ms"] = ms
    if detail is not None:
        payload["detail"] = detail
    await broadcast_to_dashboards(payload)


@app.post("/api/sell-video")
async def api_sell_video(file: UploadFile = File(...), voice_text: str = Form("sell this")):
    """Upload a product video. Extracts frames + transcript, runs full pipeline.

    Returns a request_id so the caller (iPhone SellerCaptureView) can scope
    its pipeline_step subscription on the WS bus and render per-stage progress."""
    import tempfile
    request_id = uuid.uuid4().hex[:12]
    logger.info("[API] /api/sell-video called — file: %s, voice: %s, request_id: %s",
                file.filename, voice_text[:50], request_id)
    contents = await file.read()
    logger.info("[API] Video received: %d bytes (%s)", len(contents), file.filename)
    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(contents)
        video_path = f.name
    logger.info("[API] Saved to temp: %s", video_path)

    # Fire the "uploaded" pipeline_step immediately so the iPhone sees the
    # first checkmark light up without waiting for intake to start.
    asyncio.ensure_future(_emit_pipeline_step(request_id, "uploaded", "done",
                                               ms=0, detail=f"{len(contents)}B"))
    asyncio.ensure_future(run_video_sell_pipeline(video_path, voice_text,
                                                   request_id=request_id))
    return {"status": "video_pipeline_started", "bytes": len(contents),
            "request_id": request_id}


async def run_video_sell_pipeline(video_path: str, voice_text: str,
                                   request_id: str | None = None):
    """Full pipeline from video: intake → analyze → sell."""
    pipeline_state["status"] = "ingesting"
    pipeline_state["agent_log"] = []
    logger.info("=" * 60)
    logger.info("VIDEO PIPELINE START")
    logger.info("  video_path: %s", video_path)
    logger.info("  voice_text: %s", voice_text[:100])
    logger.info("=" * 60)

    log_event("SYSTEM", "Video received. Starting intake pipeline...")
    await _emit_pipeline_step(request_id, "deepgram", "active")

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
        await _emit_pipeline_step(request_id, "deepgram", "failed",
                                   detail=str(e)[:120])
        Path(video_path).unlink(missing_ok=True)
        return
    finally:
        Path(video_path).unlink(missing_ok=True)

    intake_ms = int((time.time() - t0) * 1000)
    await _emit_pipeline_step(request_id, "deepgram", "done", ms=intake_ms)
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
                extract_transcript_signals,
                hint_block_for_claude,
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
            except TimeoutError:
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
        await run_sell_pipeline(best_frames_b64[0], combined_voice,
                                 request_id=request_id)
    else:
        log_event("SYSTEM", "No usable frames extracted from video")
        await _emit_pipeline_step(request_id, "claude", "failed",
                                   detail="no usable frames")


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
    """Public REST entry-point for a viewer comment. Routes through Cody's
    4-tool dispatcher (run_routed_comment) which picks between local
    pre-rendered answers, the LIVE_LIPSYNC path (api_respond_to_comment →
    reading_chat → bridge → Wav2Lip → Director crossfade), or other tools
    based on classifier output. Same path used by the voice agent and the
    WS simulate_comment message."""
    asyncio.ensure_future(run_routed_comment(text))
    return {"status": "processing"}


# ── Director transport — used by the dashboard's Control Room layout ──────
#
# POST /api/director/force_phase — manually advance the stage machine.
#   Body (form): phase = INTRO | BRIDGE | PITCH | LIVE
# POST /api/director/on_air — pause/resume broadcasts.
#   Body (form): on = true | false
#
# These are operator affordances: during rehearsal or between takes we want
# to drop out of LIVE without tearing down the backend. For v1, on_air is a
# soft flag that the dashboard reads but doesn't gate the response pipeline —
# Item 5 (Hands agent) will tighten this when real distribution fanout
# can be paused.

_PHASE_TO_STATUS = {
    "INTRO": "idle",
    "BRIDGE": "creating",  # useEmpireSocket derives 'BRIDGE' only from
                           # comment_response_video; we broadcast an explicit
                           # force_phase event so the dashboard lands correctly.
    "PITCH": "selling",
    "LIVE": "live",
}
pipeline_state["on_air"] = True  # default ON so live flow works without toggling
pipeline_state["active_language"] = "en"  # Item 6 — live-time translation target


@app.post("/api/director/force_phase")
async def api_director_force_phase(phase: str = Form(...)):
    """Manually advance the stage machine. Operator use."""
    p = phase.upper().strip()
    if p not in _PHASE_TO_STATUS:
        raise HTTPException(400, f"unknown phase {phase!r}, expected one of "
                                 f"{list(_PHASE_TO_STATUS)}")
    new_status = _PHASE_TO_STATUS[p]
    pipeline_state["status"] = new_status
    log_event("DIRECTOR", f"Phase forced → {p}")
    await broadcast_to_dashboards({
        "type": "force_phase",
        "phase": p,
        "status": new_status,
    })
    # Also broadcast a status event so the dashboard's useEffect-driven
    # liveStage derivation updates even without the explicit force_phase
    # handler (backward-compat; handler ships in Item 2).
    await broadcast_to_dashboards({"type": "status", "status": new_status})
    return {"status": "ok", "phase": p, "backend_status": new_status}


# ── Hands agent (Item 5) ──────────────────────────────────────────────────
#
# GET  /api/hands/state        — platforms + enabled + last publish
# POST /api/hands/toggle       — toggle a single platform on/off
# POST /api/hands/publish      — fan out the current product to all enabled
#
# DistributionToggles (dashboard) hydrates from /api/hands/state on mount
# and POSTs /api/hands/toggle on each toggle. MetricsStrip subscribes to
# hands_published events (broadcast by Hands.publish_all) to animate the
# BASKETS counter in real time.

# ── Multi-language (Item 6) ─────────────────────────────────────────────

@app.get("/api/live/language")
async def api_get_language():
    return {
        "active_language": pipeline_state.get("active_language", "en"),
        "supported": translator.SUPPORTED,
        "cache_stats": translator.stats(),
    }


@app.post("/api/live/language")
async def api_set_language(lang: str = Form(...)):
    lang = lang.strip().lower()
    if lang not in translator.SUPPORTED:
        raise HTTPException(400, f"unsupported language {lang!r}, expected one of "
                                 f"{list(translator.SUPPORTED)}")
    pipeline_state["active_language"] = lang
    log_event("SELLER", f"Live language → {translator.SUPPORTED[lang]['name']} ({lang})")
    await broadcast_to_dashboards({
        "type": "language_changed",
        "lang": lang,
        "name": translator.SUPPORTED[lang]["name"],
    })
    return {"active_language": lang}


@app.get("/api/hands/state")
async def api_hands_state():
    if hands is None:
        return {"platforms": {}}
    return hands.get_state()


@app.post("/api/hands/toggle")
async def api_hands_toggle(platform: str = Form(...),
                            enabled: str = Form(...)):
    if hands is None:
        raise HTTPException(503, "hands not initialized yet")
    flag = enabled.strip().lower() in ("true", "1", "yes", "on")
    try:
        hands.set_enabled(platform, flag)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    await broadcast_to_dashboards({
        "type": "hands_toggle",
        "platform": platform,
        "enabled": flag,
    })
    return {"ok": True, "platform": platform, "enabled": flag}


@app.post("/api/hands/publish")
async def api_hands_publish():
    """Fan out the currently-active product to every enabled platform.
    Returns per-platform results. Fires hands_published events on each."""
    if hands is None:
        raise HTTPException(503, "hands not initialized yet")
    product = pipeline_state.get("product_data")
    if not product:
        raise HTTPException(400, "no active product — POST /api/sell first")
    results = await hands.publish_all(product)
    return {
        "product_name": product.get("name"),
        "results": {
            p: {
                "ok": r.ok,
                "url": r.url,
                "listing_id": r.listing_id,
                "basket_impressions": r.basket_impressions,
                "latency_ms": r.latency_ms,
                "error": r.error,
            }
            for p, r in results.items()
        },
    }


@app.get("/api/best_frames")
async def api_best_frames():
    """Return the most recent intake's best frames as base64 JPEG strings.
    Used by the iPhone StreamView's "best 6 frames" rotator — the frames
    themselves are already in pipeline_state['best_frames_b64'] after intake
    completes; we just surface them via HTTP for the phone to fetch on demand
    (rather than pushing them over WS, which would bloat the broadcast)."""
    frames = pipeline_state.get("best_frames_b64") or []
    # pipeline_state may also store a single "product_photo_b64" from older
    # paths; include it as index 0 when best_frames_b64 is empty.
    if not frames:
        single = pipeline_state.get("product_photo_b64")
        if single:
            frames = [single]
    return {
        "count": len(frames),
        "frames": frames,  # each string is base64-encoded JPEG
        "product_name": (pipeline_state.get("product_data") or {}).get("name"),
    }


@app.get("/api/avatars")
async def api_avatars():
    """Return the avatar catalog + currently active id. Used by the
    dashboard's AvatarRail (Item 2) to render the rail + light up the
    selected card."""
    catalog = pipeline_state.get("avatars_catalog") or {}
    return {
        "active_avatar_id": pipeline_state.get("active_avatar_id"),
        "avatars": [
            {
                "id": aid,
                "name": a.get("name", aid),
                "language_tags": a.get("language_tags", []),
                "voice_id": a.get("voice_id", ""),
            }
            for aid, a in catalog.items()
        ],
    }


@app.post("/api/avatars/active")
async def api_set_active_avatar(avatar_id: str = Form(...)):
    """Switch the active avatar. Seller's TTS and (future) Director's
    state-video lookups route through `pipeline_state['active_avatar_id']`
    via `_active_avatar()` so the swap is live after this call.

    v1: state_videos are identical across avatars until Veo 3.1 renders
    land, so the visible impact today is the ElevenLabs voice swap (when
    the avatar has a non-empty voice_id)."""
    catalog = pipeline_state.get("avatars_catalog") or {}
    if avatar_id not in catalog:
        raise HTTPException(404, f"avatar_id {avatar_id!r} not in catalog")
    pipeline_state["active_avatar_id"] = avatar_id
    avatar = catalog[avatar_id]
    log_event("DIRECTOR", f"Avatar switched → {avatar.get('name', avatar_id)}")
    await broadcast_to_dashboards({
        "type": "avatar_changed",
        "avatar_id": avatar_id,
        "avatar_name": avatar.get("name", avatar_id),
    })
    return {
        "active_avatar_id": avatar_id,
        "avatar_name": avatar.get("name", avatar_id),
        "voice_id": avatar.get("voice_id", ""),
    }


@app.post("/api/director/on_air")
async def api_director_on_air(on: str = Form(...)):
    """Toggle the on-air flag. Soft v1 — doesn't gate the pipeline yet."""
    flag = on.strip().lower() in ("true", "1", "yes", "on")
    pipeline_state["on_air"] = flag
    log_event("DIRECTOR", f"On Air → {flag}")
    await broadcast_to_dashboards({"type": "on_air", "on": flag})
    return {"status": "ok", "on_air": flag}


@app.get("/api/state")
async def api_state():
    catalog = pipeline_state.get("products_catalog") or {}
    return {
        "status": pipeline_state["status"],
        "product_data": pipeline_state["product_data"],
        "active_product_id": pipeline_state.get("active_product_id"),
        "active_avatar_id": pipeline_state.get("active_avatar_id"),
        # Lightweight catalog summary — id + name only, not full product_data.
        # Dashboard renders the dropdown from this; full data is available via
        # /api/state on demand for the active product.
        "products": [
            {"id": pid, "name": (p.get("name") or pid),
             "qa_count": len(p.get("qa_index") or {})}
            for pid, p in catalog.items()
        ],
        "has_photo": pipeline_state["product_clean_b64"] is not None,
        "has_3d": pipeline_state["model_3d"] is not None,
        "log_count": len(pipeline_state["agent_log"]),
        "on_air": pipeline_state.get("on_air", True),
    }


@app.post("/api/state/active_product")
async def api_set_active_product(product_id: str = Form(...)):
    """Switch the active product. Dashboard's product selector posts here.
    Broadcasts a state_sync event so all dashboard clients re-pull /api/state
    and reflect the change without a hard refresh."""
    product = _set_active_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail=f"product_id {product_id!r} not in catalog")
    await broadcast_to_dashboards({
        "type": "active_product_changed",
        "product_id": product_id,
        "product_name": product.get("name", product_id),
    })
    return {
        "active_product_id": product_id,
        "product_name": product.get("name", product_id),
        "qa_count": len(product.get("qa_index") or {}),
    }


@app.get("/api/brain/stats")
async def api_brain_stats(stream_id: str | None = None, since_seconds: float | None = None):
    """Aggregate stats for the BRAIN dashboard panel. Defaults to all streams,
    all time. Pass `?since_seconds=3600` for the last hour, `?stream_id=foo`
    to scope to one stream."""
    return brain.get_stats(stream_id=stream_id, since_seconds=since_seconds)


@app.post("/api/creator/build")
async def api_creator_build(
    file: UploadFile = File(...),
    include_3d: bool = Form(True),
):
    """CREATOR v0: generate 3 marketplace photos + 1 promo video + optional
    3D model from one input photo + the currently-active product's metadata.
    Returns URLs to the generated assets under /renders/creator/<request_id>/.

    `include_3d=false` skips the TripoSR call (saves ~15-30s when the pod
    isn't running TripoSR — Wav2Lip is on :8010, TripoSR would be on :8020)."""
    product = pipeline_state.get("product_data")
    if not product:
        raise HTTPException(
            status_code=400,
            detail="no active product loaded — POST /api/state/active_product first",
        )
    photo_bytes = await file.read()
    if not photo_bytes:
        raise HTTPException(status_code=400, detail="empty file upload")
    photo_b64 = base64.b64encode(photo_bytes).decode()
    log_event("CREATOR", f"build_all (include_3d={include_3d})",
              {"product": product.get("name"), "input_bytes": len(photo_bytes)})
    try:
        result = await creator_build_all(photo_b64, product, include_3d=include_3d)
    except Exception as e:
        logger.exception("[creator] build_all failed")
        raise HTTPException(status_code=500, detail=f"creator failed: {e}") from e
    log_event("CREATOR", f"built {len(result['photos'])} photos + promo "
              f"({result['timing_ms']['total']}ms)",
              {"request_id": result["request_id"]})
    return result


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
  <title>EMPIRE · Live Comment</title>
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
  <div class="logo">EMPIRE</div>
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


def _save_render(label: str, data: bytes) -> str:
    fname = f"{label}_{int(time.time())}.mp4"
    path = RENDER_DIR / fname
    path.write_bytes(data)
    return f"/renders/{fname}"


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


# Legacy stub — kept for backwards compat. Real bridge selection now lives
# in agents/bridge_clips.py (manifest-driven, populated by `python -m
# agents.bridge_clips render`). This dict is unused at runtime.
CLIP_LIBRARY: dict[str, list[str]] = {
    "question":   [],
    "compliment": [],
    "objection":  [],
    "spam":       [],
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
    """LIVE comment response with seamless avatar choreography.

    Flow:
      1. reading_chat clip fades in instantly (avatar 'reads the comment')
      2. classify + LLM + TTS run in parallel, classify result picks the bridge
      3. bridge clip crossfades over reading_chat the moment we know the label
      4. Wav2Lip renders the response over the same source
      5. response crossfades over the bridge when ready
      6. fade_to_idle releases Tier 1 back to the always-on Tier 0 idle layer

    Target: sub-8s end-to-end (warm pod, warm face cache).
    """
    product_data = pipeline_state.get("product_data") or {}
    total_t0 = time.time()

    # 1) Reading the chat — instant visual feedback. Director holds it briefly
    #    before the bridge takes over, so the viewer registers the moment.
    if director:
        asyncio.create_task(director.reading_chat())

    # 2) Classify + LLM in parallel. Classify drives bridge selection; LLM gets
    #    "question" as a safe default if classify is slow.
    class_t0 = time.time()
    classify_task = asyncio.create_task(classify_comment_gemma(comment))

    t0 = time.time()
    try:
        response_text = await generate_comment_response(comment, product_data, "question")
    except Exception:
        response_text = "Great question — let me get back to you on that."
    resp_ms = int((time.time() - t0) * 1000)
    log_event("SELLER", f"response drafted ({resp_ms}ms)", {"response": response_text})
    # Stash for degraded-path fallback: if TTS or Wav2Lip fails below, the
    # wrapper in run_routed_comment reads this so we can still surface the
    # text answer to the dashboard instead of showing nothing.
    pipeline_state["last_response_text"] = {"comment": comment, "response": response_text}

    # 3) Collect classify result NOW — needed to pick the right bridge label.
    #    Don't wait if it hasn't finished; fall back to "neutral" pool.
    comment_type = "question"
    class_ms = int((time.time() - class_t0) * 1000)
    if classify_task.done():
        try:
            comment_type = (classify_task.result() or {}).get("type", "question")
        except Exception:
            pass
    else:
        classify_task.cancel()
    log_event("EYES", f'Comment "{comment[:40]}" → {comment_type} ({class_ms}ms)')

    # 4) Fire TTS and bridge in parallel. Bridge plays an audible
    #    acknowledgment over the reading_chat pose while TTS + Wav2Lip cook.
    #    Without parallelism the bridge would only get the Wav2Lip render
    #    window (~3.5s) to play; with parallelism it gets the full TTS +
    #    Wav2Lip window (~4s) so it doesn't end before the response arrives.
    bridge_task = None
    if director:
        bridge_task = asyncio.create_task(director.play_bridge(comment_type))

    t0 = time.time()
    audio_bytes = await text_to_speech(response_text)
    tts_ms = int((time.time() - t0) * 1000)

    # Best-effort: collect bridge result without blocking. If the manifest
    # is empty / the call errored we just keep reading_chat showing.
    if bridge_task and bridge_task.done():
        try:
            bridge_task.result()
        except Exception:
            logger.exception("[director] play_bridge failed (non-fatal)")

    # 5) Wav2Lip render — use the substrate of whichever Tier 0 idle is
    #    currently visible so the response inherits the same body language.
    #    Eliminates the "different person leaning forward" jump-cut when
    #    Tier 1 fades in over the calm idle layer.
    substrate = director.current_substrate_pod_path() if director else None
    t0 = time.time()
    try:
        if substrate:
            video_bytes, headers = await render_comment_response_wav2lip(
                audio_bytes, source_path_on_pod=substrate, out_height=out_height,
            )
        else:
            video_bytes, headers = await render_comment_response_wav2lip(
                audio_bytes, out_height=out_height,
            )
    except Exception as e:
        # If the configured substrate doesn't exist on the pod (400 / 404),
        # mark it unavailable so future calls skip it, then retry with the
        # default speaking-pose substrate. This keeps the live path alive
        # when speaking variants haven't shipped yet.
        if substrate and director:
            err_str = str(e).lower()
            if "404" in err_str or "400" in err_str or "not found" in err_str:
                logger.warning("[lipsync] substrate %s unavailable, falling back: %s",
                               substrate, e)
                director.mark_substrate_status(substrate, False)
                video_bytes, headers = await render_comment_response_wav2lip(
                    audio_bytes, out_height=out_height,
                )
            else:
                raise
        else:
            raise
    lipsync_ms = int((time.time() - t0) * 1000)

    url = _save_render("resp", video_bytes)
    pipeline_state["last_response_video_url"] = url
    total_ms = int((time.time() - total_t0) * 1000)

    log_event("SELLER", f"comment response ready in {total_ms}ms", {
        "url": url, "classify_ms": class_ms, "llm_ms": resp_ms,
        "tts_ms": tts_ms, "lipsync_ms": lipsync_ms,
    })

    # 6) Crossfade in the live response, then release back to idle when done.
    if director:
        await director.play_response(url)
        # Probe the actual rendered video duration so the idle release fires
        # precisely as the avatar finishes speaking. Falls back to a word-count
        # estimate if ffprobe fails. Adds a small tail (400ms) so the response
        # gets a beat of post-speech facial relaxation before crossfade.
        rendered_path = RENDER_DIR / Path(url).name
        play_ms = _probe_video_duration_ms(rendered_path)
        if play_ms is None:
            word_count = len(response_text.split())
            play_ms = int(max(2500, word_count * 350))
        play_ms_with_tail = play_ms + 400

        async def _release_after(delay_ms: int):
            await asyncio.sleep(delay_ms / 1000)
            await director.fade_to_idle()
        asyncio.create_task(_release_after(play_ms_with_tail))

    await broadcast_to_dashboards({
        "type": "comment_response_video",
        "comment": comment,
        "response": response_text,
        "url": url,
        "total_ms": total_ms,
        "class_ms": class_ms,
        "llm_ms": resp_ms,
        "tts_ms": tts_ms,
        "lipsync_ms": lipsync_ms,
    })
    return {
        "comment": comment,
        "response": response_text,
        "url": url,
        "total_ms": total_ms,
        "breakdown": {
            "classify_ms": class_ms,
            "llm_ms": resp_ms,
            "tts_ms": tts_ms,
            "lipsync_ms": lipsync_ms,
        },
        "wav2lip": {
            "total_sec": headers.get("x-total-sec"),
            "detect_sec": headers.get("x-detect-sec"),
            "predict_sec": headers.get("x-predict-sec"),
        },
    }


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
      1. Gemma 4 classify on device (already lock-serialized).
      2. router.decide → {tool, args, reason, ms, was_local, cost_saved_usd}.
      3. Broadcast routing_decision WS event for the RoutingPanel.
      4. Dispatch to the matching _run_* helper. On any downstream failure,
         the helpers fade Tier 1 back to idle and broadcast comment_failed
         with the best available fallback text (drafted by Claude before
         the failure, when applicable) so the dashboard never sticks."""
    product = pipeline_state.get("product_data")

    # 1. Classify (on-device Gemma 4). Never raises — returns a dict with
    #    at least {type, source} even on fallback. Safe to await here.
    classify = await classify_comment_gemma(comment)

    # 2. Decide.
    decision = await comment_router.decide(comment, classify, product)

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

    # 3b. Persist for BRAIN aggregation. Hardcoded stream_id="default" until
    #     multi-stream isolation lands (Sprint 5+); active_product_id mirrors
    #     pipeline_state which the dashboard sets via /api/state/active_product
    #     once multi-product (S1.3) ships.
    brain.record_event(
        stream_id="default",
        product_id=pipeline_state.get("active_product_id") or "unknown",
        comment=comment,
        classify=classify,
        decision=decision,
    )

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

    # Stage the response via the Director so Tier 0/Tier 1 layering stays
    # intact. Fade to idle after the clip plays — we don't have ffprobe
    # probing on pre-rendered clips, so use a word-count estimate with a
    # small tail. Clips are short (1-2 sentences) so word count is fine.
    if director:
        await director.play_response(url)
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
    if director:
        await director.play_response(url)
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
