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
from agents.bridge_clips import all_bridges, pick_bridge_clip, pick_intent_substrate
from agents.creator import build_all as creator_build_all
from agents.creator import generate_3d_model, remove_background
from agents.eyes import (
    analyze_and_script_claude,
    analyze_and_script_gemma,
    analyze_and_script_text_only,
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
    POD_SPEAKING_1080P,
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

    # Load demo products + pre-select an active one for respond_locally so
    # the router can answer routine questions without any prior /api/sell
    # call. ACTIVE_PRODUCT_ID env var overrides the first-key default if
    # you want to swap between demo objects without editing code on stage.
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

    # NOTE: upload-bridge (walk_off → processing.mp4 chain) + voice_state
    # are now fired by the route handler via _play_upload_bridge(), not
    # here. Calling once at the route layer means the chain fires exactly
    # once per upload regardless of which pipeline runs (run_sell_pipeline
    # is also called from run_video_sell_pipeline and we previously got
    # double walk-offs). See Option A in the avatar-director refactor.

    # PHASE 1: Product analysis + pitch script + background removal in parallel.
    #
    # PRODUCT_ANALYSIS_MODEL env var (defaults to "auto") picks the vision/
    # script engine:
    #   "gemma"  — Cactus Gemma 4 on-device only, raise if unavailable
    #   "claude" — Claude Haiku on Bedrock only (legacy cloud path)
    #   "auto"   — try Gemma first, fall back to Claude if it fails
    #
    # Cody wants the seller flow to stay local: the iPhone films → Mac
    # runs Gemma on the seller's narration + product frame → pitch script
    # comes out. No cloud roundtrip for understanding.
    pam = os.getenv("PRODUCT_ANALYSIS_MODEL", "auto").lower()
    log_event("EYES", f"Analyzing product + writing script ({pam} path)...")
    await _emit_pipeline_step(request_id, "claude", "active",
                               detail=f"model={pam}")
    t0 = time.time()

    # Detect if Deepgram already extracted a usable seller narration during
    # the intake phase. run_video_sell_pipeline injects it as
    # "Seller's narration: <transcript>" appended to voice_text. When
    # present and at least 20 chars long, we skip Gemma's vision pass
    # entirely and go straight to a text-only call (~2-3s vs ~18s).
    # Mirrors swarmsell's intake recipe (Deepgram → text → SLM → output).
    NARRATION_MARKER = "Seller's narration:"
    has_narration = False
    if NARRATION_MARKER in voice_text:
        narration = voice_text.split(NARRATION_MARKER, 1)[1].strip()
        has_narration = len(narration) >= 20

    async def _analyze_combined():
        """Cascade: text-only Gemma (fast) → vision Gemma (fallback) →
        Claude cloud (last resort). All return the same `{product,
        script}` shape so downstream code is agnostic.

        Text-only path requires a Deepgram-derived narration in
        voice_text (set by run_video_sell_pipeline during intake).
        Photo-only uploads (/api/sell with no audio) skip the text-only
        path because there's nothing to read — they go straight to
        vision Gemma."""
        if pam in ("gemma", "auto"):
            # Fast path — text-only when Deepgram narration exists.
            if has_narration:
                try:
                    result = await analyze_and_script_text_only(voice_text)
                    src = result.get("source", "")
                    if src == "cactus_on_device":
                        return result
                    logger.info("[EYES] text-only Gemma failed (%s), trying vision",
                                result.get("reason", "unknown"))
                except Exception as e:
                    logger.warning("[EYES] text-only Gemma errored, trying vision: %s", e)

            # Vision fallback — slower but pixel-grounded for cases where
            # text alone wasn't enough (or the photo-only upload path
            # where we never had narration).
            try:
                result = await analyze_and_script_gemma(frame_b64, voice_text)
                src = result.get("source", "")
                if src == "cactus_on_device":
                    return result
                if pam == "gemma":
                    # Strict mode — return the failure as-is so the caller
                    # can surface it; don't fall through to Claude.
                    return result
                logger.info("[EYES] Gemma path failed (%s), falling back to Claude",
                            result.get("reason", "unknown"))
            except Exception as e:
                logger.warning("[EYES] Gemma call errored, falling back to Claude: %s", e)
        try:
            return await analyze_and_script_claude(frame_b64, voice_text)
        except Exception as e:
            logger.error("[EYES] Claude combined error: %s", e)
            return {"error": str(e), "source": "claude_error"}

    async def _bg_removal():
        try:
            return await remove_background(frame_b64)
        except Exception as e:
            logger.error("Background removal error: %s", e)
            return None

    claude_result, clean_b64 = await asyncio.gather(
        _analyze_combined(), _bg_removal()
    )
    phase1_ms = int((time.time() - t0) * 1000)

    # Extract product data and script from combined result. Preserve the
    # analyzer's own `source` tag (cactus_on_device | claude_cloud |
    # gemma_failed | claude_error) so the verify curl + dashboards can
    # tell which engine actually wrote this product — the pipeline shell
    # shouldn't overwrite truth with a hardcoded label.
    product_data = claude_result.get("product", claude_result)
    product_data["source"] = claude_result.get("source", "claude_cloud")
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
            # Probe rendered duration BEFORE the emit so play_response can
            # carry expected_duration_ms — without this the Director's
            # busy_until horizon defaulted to 8 s and autonomous Tier 1
            # interjections (sip / walk_off / glance) would fire over the
            # pitch the moment busy expired (~+10 s into a 20 s pitch).
            rendered_path = RENDER_DIR / Path(url).name
            play_ms = _probe_video_duration_ms(rendered_path)
            if play_ms is None:
                word_count = len(script.split())
                play_ms = int(max(2500, word_count * 350))
            play_ms_with_tail = play_ms + 400

            # PITCH LOCK — hard gate that prevents ANY autonomous Tier 1
            # emit from firing for the duration of the pitch, regardless
            # of timer drift. Cleared either by:
            #   1. The dashboard's pitch_audio_end event (truth source —
            #      handled in director.observe). Fires the moment the
            #      <audio> element's ended event ticks.
            #   2. The fallback _release_pitch_to_idle task below, in
            #      case the dashboard's WS drops mid-pitch and the
            #      pitch_audio_end never lands.
            director.lock_tier1_for_pitch()
            await director.play_response(url, expected_duration_ms=play_ms)

            async def _release_pitch_to_idle(delay_ms: int):
                await asyncio.sleep(delay_ms / 1000)
                if director:
                    # Fallback unlock — pitch_audio_end usually beats us
                    # to the punch but if the dashboard WS dropped or
                    # the dashboard's <audio> errored, this guarantees
                    # the lock releases. Idempotent: if already unlocked
                    # by observe(), this is a no-op (settle_until just
                    # gets refreshed).
                    director.unlock_tier1_with_settle(settle_seconds=2.5)
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


async def _play_upload_bridge() -> None:
    """Fire the upload-bridge cover (walk_off → processing chain) and flip
    voice-state to 'thinking' so Spin3D rim-light + voice pill move in
    lockstep with the visual. Bridge ownership lives at the route layer
    (Option A) — the pipelines no longer emit it, which means the chain
    fires exactly once per upload regardless of which pipeline runs.
    Wrapped non-fatal: a Director-broadcast failure should never block
    the pipeline kickoff.

    Also clears the previous upload's `product_data` so the dashboard
    doesn't flash the prior product's name (e.g. "Minimal Leather
    Wallet" — the seeded catalog default) for the 7-12 s between drop
    and Gemma's analyze landing. Carousel updates faster than analyze
    (~7 s vs ~12 s), so without this clear the operator sees the new
    product's frames + the old product's name simultaneously.
    """
    pipeline_state["product_data"] = None
    pipeline_state["pitch_video_url"] = None
    pipeline_state["product_clean_b64"] = None
    pipeline_state["last_response_text"] = None
    try:
        await broadcast_to_dashboards({"type": "product_data", "data": None})
    except Exception:
        logger.exception("[route] product_data clear broadcast failed (non-fatal)")
    if not director:
        return
    try:
        await director.play_processing()
        await director.set_voice_state("thinking")
    except Exception:
        logger.exception("[route] play_processing emit failed (non-fatal)")


@app.post("/api/sell")
async def api_sell(file: UploadFile = File(...), voice_text: str = Form("sell this")):
    contents = await file.read()
    frame_b64 = base64.b64encode(contents).decode()
    await _play_upload_bridge()
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
    # Bridge BEFORE pipeline kickoff so the avatar starts walking off
    # within ~50 ms of the POST landing — closes the pre-intake dead-air
    # gap (~3-5 s of Deepgram + frame extract) on top of the existing
    # in-pipeline coverage. Awaited (not background-scheduled) so the
    # broadcast actually flushes before the heavy work starts.
    await _play_upload_bridge()
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

    # NOTE: upload-bridge (walk_off → processing.mp4) is fired by the
    # /api/sell-video route handler via _play_upload_bridge(), not here.
    # Single call site = no chain restart on the run_video → run_sell
    # cascade. See Option A in the avatar-director refactor.

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

    if transcript:
        log_event("EYES", f'Seller said: "{transcript[:200]}..."')
        # Stash for comment dispatch — classify_comment_gemma reads this
        # so the on-device Gemma draft can quote what the seller actually
        # said in the pitch ("you mentioned same-day shipping…") instead
        # of falling back to a generic "we're not sure what you're
        # asking" reply when the comment is terse like "how much?".
        pipeline_state["seller_transcript"] = transcript
        await broadcast_to_dashboards({"type": "transcript", "text": transcript})

    # NOTE: transcript_extract was firing here in parallel — a separate
    # Cactus call to extract structured pitch hints ({title, key_features,
    # tone}) for grounding Claude's prompt. With the new text-only Gemma
    # path (analyze_and_script_text_only), Gemma already has the full
    # transcript via voice_text and produces both product fields + script
    # in one call. The extract was now redundant AND was blocking the
    # main Cactus device — running both in parallel competed for ANE/CPU
    # cycles, dragging the text-only Gemma call from ~3s to ~9s. Skipping
    # it cuts the Gemma stage in half without losing any signal (the
    # transcript is already in voice_text, Gemma reads it directly).
    # Re-enable by setting EMPIRE_TRANSCRIPT_EXTRACT=1 if you ever want
    # the structured hint block back as a Claude-fallback grounding.
    if transcript and os.getenv("EMPIRE_TRANSCRIPT_EXTRACT") == "1":
        try:
            from agents.transcript_extract import (
                extract_transcript_signals,
            )
            extract_task = asyncio.create_task(
                extract_transcript_signals(transcript)
            )
            asyncio.ensure_future(_finish_transcript_extract(extract_task))
            log_event("EYES", "Transcript extract opted-in (background only)")
        except Exception as e:
            logger.warning("[TRANSCRIPT_EXTRACT] setup failed: %s", e)

    if best_frames_b64:
        combined_voice = f"{voice_text}. Seller's narration: {transcript}" if transcript else voice_text
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
# Veo bridge clip library — welcome.mp4, pitch_chained.mp4, intent-based
# bridges (question/objection/compliment/neutral/intro), and the new
# processing/paper-reading clip. The simulator + Director both reference
# these via /bridges/<intent>/<clip>.mp4. Without this mount the URLs 404
# (the welcome.mp4 fallback in TIER1_INTERJECTIONS was silently failing).
BRIDGES_DIR = Path(__file__).resolve().parent.parent / "phase0" / "assets" / "bridges"
CLIPS_DIR.mkdir(parents=True, exist_ok=True)
BRIDGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/clips", StaticFiles(directory=str(CLIPS_DIR)), name="clips")
app.mount("/states", StaticFiles(directory=str(STATES_DIR)), name="states")
app.mount("/bridges", StaticFiles(directory=str(BRIDGES_DIR)), name="bridges")

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

    # Item 6 — match the pitch path: if active_language is non-English,
    # translate the response_text first and pass language_code to
    # ElevenLabs flash_v2_5 (multilingual). The translator caches every
    # (text_hash, lang) tuple in sqlite so repeat answers are free; only
    # the first time we ever speak a particular response in a new
    # language costs one Claude Haiku call. Failure modes (Bedrock error,
    # unknown lang) fall through to the original English text — see
    # translator.translate() for the fallback contract.
    active_lang = pipeline_state.get("active_language", "en")
    tts_text = response_text
    if active_lang != "en":
        try:
            tts_text = await translator.translate(response_text, active_lang)
            if tts_text != response_text:
                log_event("SELLER", f"Response translated to {active_lang} "
                                    f"({len(response_text)} → {len(tts_text)} chars)")
        except Exception as e:
            logger.warning("[lang] response translate failed (%s) — falling back to English", e)
            tts_text = response_text

    t0 = time.time()
    audio_bytes = await text_to_speech(tts_text, language_code=active_lang)
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
# Two entry points:
#
#   POST /api/voice_comment
#     accepts an audio blob (webm/opus from the dashboard MediaRecorder,
#     .wav from curl smoke tests), transcribes on-device via whisper-base on
#     Cactus (fallback: Gemini 2.5 Flash), broadcasts voice_transcript to
#     the dashboard, then hands the transcript to run_routed_comment().
#
#   run_routed_comment(comment)
#     The four-tool dispatcher: rule-based router fed by Gemma's classify
#     output picks among (respond_locally,
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
    # Pass product + transcript context so Gemma's draft is grounded in
    # what the seller actually said + what the product actually is.
    # Without these the prompt is just the comment string in isolation
    # and Gemma defaults to generic fallbacks for anything it doesn't
    # recognise. Both are pulled from pipeline_state — set by the sell
    # video pipeline and the catalog default loader respectively.
    classify = await classify_comment_gemma(
        comment,
        product=pipeline_state.get("product_data"),
        transcript=pipeline_state.get("seller_transcript"),
    )

    # 2. Decide.
    decision = await comment_router.decide(comment, classify, product)

    # 3. Broadcast so RoutingPanel + the Gemma-decision HUD on
    #    the operator stage can render the early signal — fires within
    #    ~300 ms of the comment landing, ~5-7 s before the rendered
    #    comment_response_video shows up. `intent` is the raw classifier
    #    bucket (compliment / objection / question / spam); `intent_hint`
    #    in args is what the dispatcher actually uses to pick the bridge
    #    substrate (resolved from cue lists when classify is "question"
    #    but the comment scans as a compliment, etc.).
    await broadcast_to_dashboards({
        "type": "routing_decision",
        "comment": comment,
        "tool": decision["tool"],
        "reason": decision["reason"],
        "ms": decision["ms"],
        "was_local": decision["was_local"],
        "cost_saved_usd": decision["cost_saved_usd"],
        "intent": classify.get("type"),
        "intent_hint": decision.get("args", {}).get("intent_hint"),
        "classify_ms": int(classify.get("latency_ms", 0)),
        "draft_response": (classify.get("draft_response") or "")[:140],
    })
    log_event("ROUTER", f'{decision["tool"]} — {decision["reason"]} ({decision["ms"]}ms)',
              {"classify": classify.get("type"), "was_local": decision["was_local"]})

    # 3b. Persist for BRAIN aggregation. Hardcoded stream_id="default" until
    #     multi-stream isolation lands (roadmap); active_product_id mirrors
    #     pipeline_state which the dashboard sets via /api/state/active_product.
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
    # Default (escalate_to_cloud bucket): the new bridge+wav2lip path.
    # Uses Gemma's draft_response (already on-device, no extra LLM call)
    # and lip-syncs onto the intent-specific bridge clip as substrate.
    # Falls back to the legacy _run_escalate_to_cloud (Claude + default
    # substrate) on any failure so the demo never shows dead air.
    return await _run_bridge_with_wav2lip(comment, classify, decision)


async def _run_bridge_with_wav2lip(comment: str, classify: dict, decision: dict) -> dict:
    """The intent-aware comment dispatch path.

    Architecture (per the user's spec):
      1. Gemma already classified upstream (in run_routed_comment) and
         returned `{type, draft_response}` — both fields used directly
         here. NO additional LLM call.
      2. reading_chat fires immediately as the visual mask while we work.
      3. Pick a random raw bridge clip from /workspace/bridges/<intent>/
         on the pod (intent ∈ {compliment, objection, question}).
         Falls back to the default speaking-pose substrate when the
         intent has no clips (or classifier returned an unknown type).
      4. ElevenLabs TTS the draft_response.
      5. Wav2Lip /lipsync_fast renders the audio onto the bridge clip
         as substrate. Bridge clips are 8 s — comfortably longer than
         a typical 5 s response. First render per substrate is COLD
         (~12 s, builds face-detect cache); subsequent are warm (~5-6 s).
      6. Director.play_response crossfades from reading_chat to the
         rendered output. fade_to_idle releases Tier 1 after the audio
         finishes.

    The avatar visibly does an intent-appropriate gesture (warm smile
    for compliments, thoughtful nod for questions, "actually..." beat
    for objections) WHILE speaking the response. Mouth alignment isn't
    perfect (Wav2Lip on a non-purpose-built substrate warbles a bit)
    but the body language coherence is the point.

    Failure mode: if Wav2Lip errors (404 substrate, pod down, bad
    audio), fall back to _run_escalate_to_cloud which uses the default
    speaking-pose substrate via Claude. This guarantees a response
    even when the bridge upload is stale or the pod's face cache is
    inaccessible.
    """
    intent = (classify.get("type") or "").lower().strip() or "question"
    draft = (classify.get("draft_response") or "").strip()

    # 1. Pick substrate. None ⇒ default speaking pose (safe path).
    substrate = pick_intent_substrate(intent)
    pod_path = substrate["pod_path"] if substrate else POD_SPEAKING_1080P
    substrate_label = substrate["url"] if substrate else "default"
    log_event("ROUTER", f'bridge+wav2lip intent={intent} substrate={substrate_label}',
              {"pod_path": pod_path, "draft": draft[:80] if draft else None})

    # Broadcast the substrate pick IMMEDIATELY so the operator HUD shows
    # which bridge clip Gemma's intent mapped to — visible within ~300 ms
    # of the comment landing, ~5-15 s before the rendered comment_response_video
    # closes the loop. Without this the substrate is invisible until the
    # full Wav2Lip render lands, and the operator can't tell why a
    # particular bridge gesture is about to play.
    await broadcast_to_dashboards({
        "type": "comment_substrate_picked",
        "comment": comment,
        "intent": intent,
        "substrate": substrate_label,
        "draft_response": (draft or "")[:140],
    })

    # 2. Reading-chat as the visual mask while we render. Background
    #    task — emit_reading_chat returns immediately after the WS
    #    broadcast; the Tier 1 busy_until horizon set by emit() keeps
    #    the idle rotation suppressed for the loop's full ttl.
    if director:
        asyncio.create_task(director.emit_reading_chat())

    # 3. Pick the response text. Gemma's draft is the light path;
    #    fall back to Claude only when the draft is empty/too short.
    text_t0 = time.time()
    if draft and len(draft) >= 6:
        response_text = draft
        text_source = "gemma_draft"
        text_ms = 0
    else:
        try:
            response_text = await generate_comment_response(
                comment, pipeline_state.get("product_data") or {}, intent)
            text_source = "claude_fallback"
        except Exception as e:
            logger.warning("[bridge_wav2lip] Claude fallback failed: %s — using stock", e)
            response_text = "Let me come back to that one."
            text_source = "stock"
        text_ms = int((time.time() - text_t0) * 1000)
    pipeline_state["last_response_text"] = {"comment": comment, "response": response_text}

    # 4. TTS — translate first if active language ≠ en (mirrors pitch path).
    active_lang = pipeline_state.get("active_language", "en")
    tts_text = response_text
    if active_lang != "en":
        try:
            tts_text = await translator.translate(response_text, active_lang)
        except Exception as e:
            logger.warning("[bridge_wav2lip] translate failed: %s — using English", e)
    tts_t0 = time.time()
    audio_bytes = await text_to_speech(tts_text, language_code=active_lang)
    tts_ms = int((time.time() - tts_t0) * 1000)
    if not audio_bytes:
        logger.error("[bridge_wav2lip] empty TTS — escalating to legacy cloud path")
        return await _run_escalate_to_cloud(comment, {"comment": comment}, decision)

    # 5. Wav2Lip render. First call per substrate is cold (~12 s);
    #    subsequent calls hit the pod's face-detect cache and run
    #    in ~5-6 s. On HARD failure (404 / pod down / exhausted retries)
    #    fall back to the default substrate via the legacy path so the
    #    audience always sees something.
    lipsync_t0 = time.time()
    try:
        video_bytes, _hdrs = await render_comment_response_wav2lip(
            audio_bytes, source_path_on_pod=pod_path, out_height=1920)
    except Exception as e:
        logger.warning("[bridge_wav2lip] Wav2Lip on substrate %s failed (%s) — "
                       "retrying with default substrate", pod_path, e)
        try:
            video_bytes, _hdrs = await render_comment_response_wav2lip(
                audio_bytes, source_path_on_pod=POD_SPEAKING_1080P, out_height=1920)
            substrate_label = f"{substrate_label} → default (fallback)"
        except Exception as e2:
            logger.error("[bridge_wav2lip] default substrate fallback also failed: %s", e2)
            if director:
                try:
                    await director.fade_to_idle()
                except Exception:
                    pass
            await broadcast_to_dashboards({
                "type": "comment_failed",
                "comment": comment, "response": response_text,
                "reason": f"wav2lip: {str(e2)[:120]}",
            })
            raise
    lipsync_ms = int((time.time() - lipsync_t0) * 1000)

    # 6. Persist + emit via Director. Probe the rendered duration so
    #    fade-to-idle fires at the right moment (Wav2Lip output length
    #    matches the audio length, not the substrate length).
    rid = uuid.uuid4().hex[:12]
    out_path = RENDER_DIR / f"comment_{rid}.mp4"
    out_path.write_bytes(video_bytes)
    url = f"/renders/{out_path.name}"
    audio_dur_ms = _probe_video_duration_ms(out_path) or \
                   int(max(2500, len(response_text.split()) * 350))

    if director:
        await director.play_response(url, expected_duration_ms=audio_dur_ms)

        async def _release_after_audio():
            await asyncio.sleep(audio_dur_ms / 1000 + 0.4)
            await director.fade_to_idle()
        asyncio.create_task(_release_after_audio())

    total_ms = decision["ms"] + text_ms + tts_ms + lipsync_ms
    await broadcast_to_dashboards({
        "type": "comment_response_video",
        "comment": comment,
        "response": response_text,
        "url": url,
        "intent": intent,
        "substrate": substrate_label,
        "text_source": text_source,
        "total_ms": total_ms,
        "class_ms": int(classify.get("latency_ms", 0)),
        "text_ms": text_ms,
        "tts_ms": tts_ms,
        "lipsync_ms": lipsync_ms,
    })
    log_event("LIPSYNC", f"comment response rendered ({lipsync_ms}ms, "
              f"intent={intent}, substrate={substrate_label})")
    return {
        "dispatch": "bridge_wav2lip",
        "routing": decision,
        "comment": comment,
        "response": response_text,
        "url": url,
        "intent": intent,
        "substrate": substrate_label,
        "text_source": text_source,
        "total_ms": total_ms,
        "class_ms": int(classify.get("latency_ms", 0)),
        "text_ms": text_ms,
        "tts_ms": tts_ms,
        "lipsync_ms": lipsync_ms,
    }


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

    # (2) Route + render through the rule-based + Gemma-classify dispatcher.
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



# Visit /dev/transitions in a browser to watch the same idle-rotation +
# ambient-interjection loop the live Director runs on stage, but with
# adjustable speed and a live transition log so visual issues
# (crossfade artefacts, pose-mismatch pops, audio glitches at swap
# boundaries, etc.) are easy to spot and reproduce.
#
# Implements the same machinery LiveStage uses: 4 stacked <video>
# elements (Tier 0 A/B ping-pong, Tier 1 A/B ping-pong), prepareFirstFrame
# seek-to-t=0 before the opacity ramp, weighted random clip selection
# from the Director's TIER0_LIBRARY + TIER1_INTERJECTIONS tables,
# 35% chance per rotation tick that a Tier 1 interjection fires instead
# of a Tier 0 swap.
@app.get("/dev/transitions", response_class=HTMLResponse)
async def dev_transitions() -> HTMLResponse:
    body = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8" />
<title>Zo · transition simulator</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  html, body { margin: 0; background: #050507; color: #fafafa;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
                 "Segoe UI", Roboto, Inter, Arial, sans-serif; }
  body { display: grid; grid-template-columns: 1fr 360px;
         gap: 18px; padding: 18px; min-height: 100vh;
         /* Make sizes available to children's calc() — see .stage-wrap */
         --stage-h: calc(100vh - 36px);
         --stage-w: calc((100vh - 36px) * 9 / 16); }
  .stage-col { display: flex; flex-direction: column; align-items: center;
               gap: 12px; min-width: 0; }
  /* Explicit width + height keeps the 9:16 frame intact across browsers
     that flake on `aspect-ratio` + `max-height` inside a flex column
     (which collapses to 0×0 in Safari and some Chromium versions, leaving
     the video AND the in-stage badges invisible — that was the bug). */
  .stage-wrap { background: #000; border-radius: 14px; overflow: hidden;
                position: relative;
                width: var(--stage-w); height: var(--stage-h);
                max-width: 100%; }
  .stage-wrap video { position: absolute; inset: 0;
                       width: 100%; height: 100%; object-fit: contain;
                       background: #000; display: block;
                       opacity: 0; }
  /* Wrapper around ONLY the four <video> elements (siblings of .badges).
     Carrier for the blur trick — a short CSS filter:blur() pulse during
     each crossfade destroys the high-frequency facial edges (mouth, eyes,
     jawline) that betray "two faces overlapping" while the opacity ramp
     is mid-flight. Filter applied to .stage-videos (not .stage-wrap) so
     the badges + control overlays stay sharp. will-change keeps the GPU
     layer ready so the first blur tick doesn't jank. */
  .stage-videos { position: absolute; inset: 0; will-change: filter; }
  /* In-stage state badges: Tier 0 always shown; Tier 1 only when active.
     Both pulse for 600ms when the state changes so the eye locks onto the
     exact transition frame. Designed to be readable at-a-glance from
     several feet away — same readability target as the live stage. */
  .badges { position: absolute; top: 14px; left: 14px; right: 14px;
            display: flex; flex-direction: column; gap: 8px;
            pointer-events: none; z-index: 100; }
  .badge { display: inline-flex; align-items: center; gap: 10px;
           font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
           padding: 8px 14px; border-radius: 8px;
           background: rgba(9,9,11,0.85); backdrop-filter: blur(8px);
           border: 1px solid #27272a;
           transition: border-color 220ms ease, box-shadow 220ms ease,
                       transform 220ms ease;
           align-self: flex-start; max-width: 100%; }
  .badge .tier-tag { font-size: 10px; font-weight: 900;
                     letter-spacing: 2px; padding: 3px 8px;
                     border-radius: 4px; color: #09090b; }
  .badge.t0 .tier-tag { background: #38bdf8; }
  .badge.t1 .tier-tag { background: #fbbf24; }
  .badge .state-name { font-size: 16px; font-weight: 800;
                       letter-spacing: 0.5px; color: #fafafa; }
  .badge .elapsed { font-size: 10px; color: #71717a; font-weight: 600;
                    letter-spacing: 1px; }
  .badge.flash { transform: scale(1.04); }
  .badge.t0.flash { border-color: #38bdf8;
                    box-shadow: 0 0 24px rgba(56,189,248,0.55); }
  .badge.t1.flash { border-color: #fbbf24;
                    box-shadow: 0 0 24px rgba(251,191,36,0.55); }
  .badge.hidden { display: none; }
  .panel { background: #0f0f12; border: 1px solid #27272a;
           border-radius: 10px; padding: 16px;
           font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  .panel h2 { font-size: 11px; letter-spacing: 1.5px; margin: 0 0 12px;
              color: #a78bfa; text-transform: uppercase; font-weight: 800; }
  .row { display: flex; justify-content: space-between; align-items: baseline;
         padding: 4px 0; font-size: 12px; gap: 10px; }
  .row .k { color: #71717a; text-transform: uppercase; letter-spacing: 1px;
            font-size: 10px; }
  .row .v { color: #fafafa; font-weight: 700; }
  .row .v.big { font-size: 16px; color: #22c55e; }
  .row .v.muted { color: #52525b; }
  button { background: #27272a; color: #fafafa; border: 1px solid #3f3f46;
           border-radius: 6px; padding: 6px 12px; font-size: 11px;
           font-weight: 700; cursor: pointer;
           font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
           letter-spacing: 1px; text-transform: uppercase; }
  button:hover { border-color: #7c3aed; }
  button.on { background: linear-gradient(135deg,#ec4899,#7c3aed);
              border-color: transparent; color: #fff; }
  .controls { display: flex; flex-wrap: wrap; gap: 6px; }
  .log { font-size: 11px; color: #d4d4d8; max-height: 280px;
         overflow-y: auto; line-height: 1.5; }
  .log .ts { color: #52525b; margin-right: 6px; }
  .log .lvl-tier0 { color: #38bdf8; }
  .log .lvl-tier1 { color: #fbbf24; }
  .log .lvl-skip { color: #52525b; font-style: italic; }
  .weights { display: grid; grid-template-columns: 1fr auto auto; gap: 6px 12px;
             font-size: 11px; align-items: center; }
  .weights label { color: #d4d4d8; }
  .weights input[type=number] { width: 56px; background: #050507;
                                  color: #fafafa; border: 1px solid #27272a;
                                  border-radius: 4px; padding: 3px 6px;
                                  font-family: inherit; font-size: 11px; }
  .weights .pct { color: #52525b; min-width: 36px; text-align: right; }
  hr { border: 0; border-top: 1px solid #18181b; margin: 14px 0; }
</style>
</head><body>
<div class="stage-col">
  <div class="stage-wrap">
    <div class="stage-videos" id="stage-videos">
      <video id="t0a" playsinline muted loop></video>
      <video id="t0b" playsinline muted loop></video>
      <video id="t1a" playsinline muted></video>
      <video id="t1b" playsinline muted></video>
    </div>
    <div class="badges">
      <div class="badge t0" id="badge-t0">
        <span class="tier-tag">TIER 0</span>
        <span class="state-name" id="badge-t0-name">—</span>
        <span class="elapsed" id="badge-t0-elapsed">0.0s</span>
      </div>
      <div class="badge t1 hidden" id="badge-t1">
        <span class="tier-tag">TIER 1</span>
        <span class="state-name" id="badge-t1-name">—</span>
        <span class="elapsed" id="badge-t1-elapsed">0.0s</span>
      </div>
    </div>
  </div>
</div>

<div>
  <div class="panel">
    <h2>now playing</h2>
    <div class="row"><span class="k">tier 0</span>
      <span class="v big" id="now-t0">—</span></div>
    <div class="row"><span class="k">tier 1</span>
      <span class="v" id="now-t1">(idle)</span></div>
    <div class="row"><span class="k">next rotation</span>
      <span class="v" id="next-rot">—</span></div>
    <hr />
    <div class="controls">
      <button id="btn-pause">pause</button>
      <button id="btn-skip">skip to next rotation</button>
      <button id="btn-force-t1">force interjection now</button>
    </div>
    <hr />
    <div class="row"><span class="k">speed</span>
      <span class="controls">
        <button class="speed" data-mult="1">1×</button>
        <button class="speed on" data-mult="2">2×</button>
        <button class="speed" data-mult="5">5×</button>
        <button class="speed" data-mult="10">10×</button>
      </span></div>
    <div class="row"><span class="k">interjection p</span>
      <span><input type="number" id="prob-input"
        min="0" max="1" step="0.05" value="0.35" /></span></div>
    <div class="row"><span class="k">rotation min/max (s)</span>
      <span>
        <input type="number" id="rot-min" min="1" max="60" step="1" value="8" />
        –
        <input type="number" id="rot-max" min="1" max="60" step="1" value="18" />
      </span></div>
    <hr />
    <div class="row"><span class="k">blur peak (px)</span>
      <span>
        <input type="range" id="blur-peak" min="0" max="20" step="0.5" value="8"
               style="vertical-align: middle; width: 140px;" />
        <span id="blur-peak-val" class="v" style="display:inline-block; min-width: 30px; text-align: right;">8.0</span>
      </span></div>
    <div class="row"><span class="k">blur max (ms)</span>
      <span>
        <input type="range" id="blur-max" min="60" max="1000" step="20" value="500"
               style="vertical-align: middle; width: 140px;" />
        <span id="blur-max-val" class="v" style="display:inline-block; min-width: 36px; text-align: right;">500</span>
      </span></div>
  </div>

  <div class="panel" style="margin-top:12px">
    <h2>tier 0 weights</h2>
    <div class="weights" id="t0-weights"></div>
  </div>

  <div class="panel" style="margin-top:12px">
    <h2>tier 1 weights</h2>
    <div class="weights" id="t1-weights"></div>
  </div>

  <div class="panel" style="margin-top:12px">
    <h2>transition log</h2>
    <div class="log" id="log"></div>
  </div>
</div>

<script>
// Mirrors backend/agents/avatar_director.py constants. Edit there + here
// in lockstep when the live Director numbers change.
const TIER0_CROSSFADE_MS = 600;
const TIER1_CROSSFADE_MS = 120;
const TIER1_FADEOUT_MS = 500;
const PREPARE_TIMEOUT_MS = 4000;

// Mirror of TIER0_LIBRARY in avatar_director.py
const T0 = [
  { intent: 'idle_calm',       url: '/states/idle/idle_calm.mp4',       weight: 0.75 },
  { intent: 'idle_thinking',   url: '/states/idle/idle_thinking.mp4',   weight: 0.10 },
  { intent: 'misc_hair_touch', url: '/states/idle/misc_hair_touch.mp4', weight: 0.15 },
];

// Mirror of TIER1_INTERJECTIONS + the new welcome (low probability per
// the new spec). Editable from the right-side controls.
const T1 = [
  { intent: 'misc_sip_drink',       url: '/states/idle/misc_sip_drink.mp4',            weight: 0.40 },
  { intent: 'misc_walk_off_return', url: '/states/idle/misc_walk_off_return.mp4',      weight: 0.20 },
  { intent: 'misc_glance_aside',    url: '/states/idle/misc_glance_aside_silent.mp4', weight: 0.25 },
  { intent: 'welcome',              url: '/bridges/welcome/welcome.mp4',               weight: 0.15 },
];

let interjectionProbability = 0.35;
let rotMinS = 8, rotMaxS = 18;
let speed = 2;
let paused = false;
// Blur trick — symmetric pulse 0 → peak → 0 of CSS filter:blur() on
// the .stage-videos wrapper during every crossfade. Destroys high-
// frequency facial edges (mouth, eyes, jawline) so two faces
// overlapping at intermediate opacity can't be picked apart by the
// eye. Curve = sin(t * π) → smooth bell, peak at midpoint.
//   blurPeakPx = 0   disables the trick (A/B baseline against today's
//                    pure-opacity behaviour)
//   blurMaxMs        hard cap so long crossfades (Tier 0 600ms) never
//                    sit blurred — the pulse ends well before the
//                    opacity ramp does. Short stab only, just enough
//                    to hide the seam.
let blurPeakPx = 8;
let blurMaxMs  = 500;
let activeBlurRaf = null;

const els = {
  t0a: document.getElementById('t0a'),
  t0b: document.getElementById('t0b'),
  t1a: document.getElementById('t1a'),
  t1b: document.getElementById('t1b'),
  nowT0: document.getElementById('now-t0'),
  nowT1: document.getElementById('now-t1'),
  nextRot: document.getElementById('next-rot'),
  badgeT0: document.getElementById('badge-t0'),
  badgeT0Name: document.getElementById('badge-t0-name'),
  badgeT0Elapsed: document.getElementById('badge-t0-elapsed'),
  badgeT1: document.getElementById('badge-t1'),
  badgeT1Name: document.getElementById('badge-t1-name'),
  badgeT1Elapsed: document.getElementById('badge-t1-elapsed'),
  log: document.getElementById('log'),
};

// Track when each tier last changed so we can show "elapsed on this state"
// counters and trigger the flash animation right at the swap moment.
let t0StartedAt = 0;
let t1StartedAt = 0;

function flashBadge(badgeEl) {
  badgeEl.classList.add('flash');
  setTimeout(() => badgeEl.classList.remove('flash'), 600);
}

let t0ActiveIsA = true;
let t1ActiveIsA = true;
let currentT0 = null;
let currentT1 = null;
let nextRotationAt = 0;
let rotationTimer = null;

function logEvt(level, msg) {
  const ts = new Date().toLocaleTimeString('en-US', { hour12: false });
  const div = document.createElement('div');
  div.innerHTML = `<span class="ts">${ts}</span><span class="lvl-${level}">${msg}</span>`;
  els.log.prepend(div);
  while (els.log.children.length > 80) {
    els.log.removeChild(els.log.lastChild);
  }
}

function weightedPick(pool) {
  const total = pool.reduce((s, p) => s + p.weight, 0);
  let r = Math.random() * total;
  for (const p of pool) {
    r -= p.weight;
    if (r <= 0) return p;
  }
  return pool[pool.length - 1];
}

// Blur pulse — call at the start of every crossfade. Duration scales
// with the crossfade itself: Tier 1 in (120ms) → 120ms pulse; Tier 0
// rotation (600ms) → clipped to blurMaxMs so the blur is OUT well
// before the opacity ramp finishes. We force a 60ms floor so even
// the 120ms Tier 1 crossfade gets a perceptible blur arc instead of
// flickering on/off in two frames. Cancels any in-flight pulse before
// starting a new one so rapid back-to-back transitions don't stack.
function blurStage(crossfadeMs) {
  if (blurPeakPx <= 0) return;
  const stage = document.getElementById('stage-videos');
  if (!stage) return;
  if (activeBlurRaf) cancelAnimationFrame(activeBlurRaf);
  const dur = Math.max(60, Math.min(crossfadeMs, blurMaxMs));
  const start = performance.now();
  function tick(now) {
    const t = Math.max(0, Math.min(1, (now - start) / dur));
    const px = blurPeakPx * Math.sin(t * Math.PI);
    stage.style.filter = px > 0.05 ? `blur(${px.toFixed(2)}px)` : 'none';
    if (t < 1) {
      activeBlurRaf = requestAnimationFrame(tick);
    } else {
      stage.style.filter = 'none';
      activeBlurRaf = null;
    }
  }
  activeBlurRaf = requestAnimationFrame(tick);
}

// Mirror of the LiveStage prepareFirstFrame helper — load the new src,
// seek to t=0, await the seeked event so the FIRST frame is decoded
// before the opacity ramp begins. Without this the crossfade can show
// frames 2-5 of the new clip mid-fade (visible head jump).
function prepareFirstFrame(el, src) {
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      el.removeEventListener('loadedmetadata', onLoadedMeta);
      el.removeEventListener('seeked', onSeeked);
      el.removeEventListener('error', onError);
      clearTimeout(timer);
    };
    function onError() { cleanup(); reject(el.error || 'video_error'); }
    function trySeek() {
      if (el.currentTime > 0.001) el.currentTime = 0;
      else Promise.resolve().then(onSeeked);
    }
    function onLoadedMeta() {
      el.removeEventListener('loadedmetadata', onLoadedMeta);
      trySeek();
    }
    function onSeeked() {
      if (el.readyState >= 2) { cleanup(); resolve(); }
      else el.addEventListener('canplay', () => { cleanup(); resolve(); }, { once: true });
    }
    el.addEventListener('error', onError, { once: true });
    el.addEventListener('seeked', onSeeked);
    const timer = setTimeout(() => { cleanup(); reject(new Error('prepare_timeout')); }, PREPARE_TIMEOUT_MS);
    if (el.src === src && el.readyState >= 2 && el.currentTime <= 0.01) {
      cleanup(); resolve(); return;
    }
    if (el.src !== src) {
      el.src = src;
      el.addEventListener('loadedmetadata', onLoadedMeta, { once: true });
      try { el.load(); } catch {}
    } else {
      trySeek();
    }
  });
}

async function swapTier0(pick) {
  const incoming = t0ActiveIsA ? els.t0b : els.t0a;
  const outgoing = t0ActiveIsA ? els.t0a : els.t0b;
  incoming.muted = true;
  incoming.loop = true;
  try {
    await prepareFirstFrame(incoming, pick.url);
    await incoming.play();
    blurStage(TIER0_CROSSFADE_MS);
    incoming.style.transition = `opacity ${TIER0_CROSSFADE_MS}ms ease`;
    outgoing.style.transition = `opacity ${TIER0_CROSSFADE_MS}ms ease`;
    incoming.style.opacity = 1;
    outgoing.style.opacity = 0;
    t0ActiveIsA = !t0ActiveIsA;
    currentT0 = pick;
    t0StartedAt = Date.now();
    els.nowT0.textContent = pick.intent;
    els.badgeT0Name.textContent = pick.intent;
    flashBadge(els.badgeT0);
    setTimeout(() => { try { outgoing.pause(); } catch {} }, TIER0_CROSSFADE_MS + 50);
    logEvt('tier0', `tier 0 → ${pick.intent} (crossfade ${TIER0_CROSSFADE_MS}ms)`);
  } catch (e) {
    logEvt('skip', `tier 0 → ${pick.intent} FAILED: ${e?.message || e}`);
  }
}

async function fireTier1(pick) {
  const incoming = t1ActiveIsA ? els.t1b : els.t1a;
  const outgoing = t1ActiveIsA ? els.t1a : els.t1b;
  incoming.muted = true;
  incoming.loop = false;
  try {
    await prepareFirstFrame(incoming, pick.url);
    await incoming.play();
    blurStage(TIER1_CROSSFADE_MS);
    incoming.style.transition = `opacity ${TIER1_CROSSFADE_MS}ms ease`;
    outgoing.style.transition = `opacity ${TIER1_FADEOUT_MS}ms ease`;
    incoming.style.opacity = 1;
    if (currentT1) outgoing.style.opacity = 0;
    t1ActiveIsA = !t1ActiveIsA;
    currentT1 = pick;
    t1StartedAt = Date.now();
    els.nowT1.textContent = pick.intent;
    els.badgeT1Name.textContent = pick.intent;
    els.badgeT1.classList.remove('hidden');
    flashBadge(els.badgeT1);
    logEvt('tier1', `tier 1 → ${pick.intent} (crossfade ${TIER1_CROSSFADE_MS}ms; auto fade-out on natural end)`);
    incoming.addEventListener('ended', () => {
      blurStage(TIER1_FADEOUT_MS);
      incoming.style.transition = `opacity ${TIER1_FADEOUT_MS}ms ease`;
      incoming.style.opacity = 0;
      const ended = pick;
      currentT1 = null;
      els.nowT1.textContent = '(idle)';
      // Hide the Tier 1 badge after the fade-out settles so it disappears
      // in lockstep with the visual fade — flash Tier 0 to redirect the
      // eye back to "what's underneath."
      setTimeout(() => {
        els.badgeT1.classList.add('hidden');
        flashBadge(els.badgeT0);
      }, TIER1_FADEOUT_MS);
      logEvt('tier1', `tier 1 ← ${ended.intent} ended (fade-out ${TIER1_FADEOUT_MS}ms)`);
    }, { once: true });
  } catch (e) {
    logEvt('skip', `tier 1 → ${pick.intent} FAILED: ${e?.message || e}`);
  }
}

function rotationDelay() {
  const min = rotMinS * 1000, max = rotMaxS * 1000;
  return Math.round((min + Math.random() * (max - min)) / Math.max(speed, 0.1));
}

async function rotationTick() {
  if (paused) { scheduleNext(); return; }
  // 35% (configurable) chance to fire a tier 1 interjection instead of
  // swapping tier 0. Mirrors the live Director.
  if (Math.random() < interjectionProbability && !currentT1) {
    const pick = weightedPick(T1);
    await fireTier1(pick);
  } else {
    let pick;
    do { pick = weightedPick(T0); } while (currentT0 && pick.intent === currentT0.intent && T0.length > 1);
    await swapTier0(pick);
  }
  scheduleNext();
}

function scheduleNext() {
  if (rotationTimer) clearTimeout(rotationTimer);
  const ms = rotationDelay();
  nextRotationAt = Date.now() + ms;
  rotationTimer = setTimeout(rotationTick, ms);
}

setInterval(() => {
  // While paused, freeze the displayed elapsed at pausedAt (not Date.now())
  // so the badges stop ticking when the videos are stopped.
  const now = paused ? pausedAt : Date.now();
  if (!paused) {
    const remain = Math.max(0, nextRotationAt - Date.now());
    els.nextRot.textContent = `${(remain / 1000).toFixed(1)}s`;
  } else {
    els.nextRot.textContent = '(paused)';
  }
  if (currentT0) {
    els.badgeT0Elapsed.textContent = `${((now - t0StartedAt) / 1000).toFixed(1)}s`;
  }
  if (currentT1) {
    els.badgeT1Elapsed.textContent = `${((now - t1StartedAt) / 1000).toFixed(1)}s`;
  }
}, 100);

// Controls
document.querySelectorAll('button.speed').forEach(b => {
  b.addEventListener('click', () => {
    document.querySelectorAll('button.speed').forEach(x => x.classList.remove('on'));
    b.classList.add('on');
    speed = parseFloat(b.dataset.mult);
    logEvt('skip', `speed → ${speed}×`);
  });
});
// Pause does THREE things: stop the rotation scheduler, pause whichever
// video elements are currently playing, and freeze the elapsed counters.
// Resume reverses all three. Without the video.pause() the playback kept
// going while the state machine was idle — confusing because "paused" only
// stopped FUTURE transitions, not the current one.
let pausedAt = 0;
function pausedVideos() {
  const out = [];
  if (currentT0) out.push(t0ActiveIsA ? els.t0a : els.t0b);
  if (currentT1) out.push(t1ActiveIsA ? els.t1a : els.t1b);
  return out;
}
document.getElementById('btn-pause').addEventListener('click', (e) => {
  paused = !paused;
  e.target.textContent = paused ? 'resume' : 'pause';
  e.target.classList.toggle('on', paused);
  if (paused) {
    pausedAt = Date.now();
    if (rotationTimer) { clearTimeout(rotationTimer); rotationTimer = null; }
    pausedVideos().forEach(v => { try { v.pause(); } catch {} });
    logEvt('skip', 'paused (videos + rotation frozen)');
  } else {
    // Shift the startedAt timestamps forward by the pause duration so
    // the elapsed counters resume from where they were, not from zero.
    const dt = Date.now() - pausedAt;
    if (currentT0) t0StartedAt += dt;
    if (currentT1) t1StartedAt += dt;
    pausedAt = 0;
    pausedVideos().forEach(v => { v.play().catch(() => {}); });
    scheduleNext();
    logEvt('skip', `resumed (after ${(dt / 1000).toFixed(1)}s pause)`);
  }
});
document.getElementById('btn-skip').addEventListener('click', () => {
  if (rotationTimer) clearTimeout(rotationTimer);
  rotationTick();
});
document.getElementById('btn-force-t1').addEventListener('click', async () => {
  if (currentT1) { logEvt('skip', 'force-t1 skipped (already on)'); return; }
  await fireTier1(weightedPick(T1));
});
document.getElementById('prob-input').addEventListener('change', (e) => {
  interjectionProbability = parseFloat(e.target.value);
  logEvt('skip', `interjection p → ${interjectionProbability}`);
});
document.getElementById('rot-min').addEventListener('change', (e) => {
  rotMinS = parseInt(e.target.value, 10);
});
document.getElementById('rot-max').addEventListener('change', (e) => {
  rotMaxS = parseInt(e.target.value, 10);
});
// Live blur tuning. `input` (not `change`) so dragging the slider
// updates instantly — pair with btn-skip to fire a transition with
// each new peak value and dial in the right "tiny" amount visually.
const blurPeakInput = document.getElementById('blur-peak');
const blurPeakLabel = document.getElementById('blur-peak-val');
blurPeakInput.addEventListener('input', (e) => {
  blurPeakPx = parseFloat(e.target.value);
  blurPeakLabel.textContent = blurPeakPx.toFixed(1);
});
const blurMaxInput = document.getElementById('blur-max');
const blurMaxLabel = document.getElementById('blur-max-val');
blurMaxInput.addEventListener('input', (e) => {
  blurMaxMs = parseInt(e.target.value, 10);
  blurMaxLabel.textContent = `${blurMaxMs}`;
});

// Render the editable weight tables
function renderWeights(pool, target) {
  const total = pool.reduce((s, p) => s + p.weight, 0);
  target.innerHTML = '';
  pool.forEach((p, i) => {
    const lab = document.createElement('label'); lab.textContent = p.intent;
    const inp = document.createElement('input');
    inp.type = 'number'; inp.min = 0; inp.max = 1; inp.step = 0.05;
    inp.value = p.weight.toFixed(2);
    inp.addEventListener('change', () => {
      p.weight = parseFloat(inp.value);
      renderWeights(pool, target);
    });
    const pct = document.createElement('span'); pct.className = 'pct';
    pct.textContent = total > 0 ? `${Math.round(100 * p.weight / total)}%` : '—';
    target.append(lab, inp, pct);
  });
}
renderWeights(T0, document.getElementById('t0-weights'));
renderWeights(T1, document.getElementById('t1-weights'));

// Boot — pick a starting tier 0 then start rotating
(async () => {
  await swapTier0(weightedPick(T0));
  scheduleNext();
})();
</script>
</body></html>"""
    return HTMLResponse(body, headers={"Cache-Control": "no-store"})


# ── Run ────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=True)
