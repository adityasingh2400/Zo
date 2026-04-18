import asyncio
import json
import time
import base64
import logging
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("empire")

from config import BACKEND_HOST, BACKEND_PORT
from agents.eyes import analyze_with_claude, analyze_and_script_claude, classify_comment_gemma
from agents.creator import remove_background, generate_3d_model
from agents.seller import (
    generate_comment_response,
    make_avatar_speak,
    text_to_speech,
    set_livetalking_session,
    render_comment_response_wav2lip,
    render_pitch_latentsync,
)
from agents.intake import process_video
from agents.threed import carousel_from_video, glb_from_image
from agents.bridge_clips import pick_bridge_clip, all_bridges
from agents.avatar_director import Director

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

# Single Director instance owns all play_clip emission. Bound to the
# dashboard broadcast helper so it can talk to every connected client.
director: "Director | None" = None  # set in app startup, see below


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
    for ws in dead:
        dashboard_clients.remove(ws)


# ── App ────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global director
    print(f"EMPIRE backend running on {BACKEND_HOST}:{BACKEND_PORT}")
    # Bring up the Avatar Director early so the first dashboard connect can
    # immediately receive a Tier 0 idle clip.
    director = Director(broadcast_to_dashboards)
    director.start_idle_rotation()
    logger.info("Avatar Director instantiated; idle rotation running.")
    # Pre-load Cactus model in background thread so first request isn't slow
    from agents.eyes import _get_cactus_model, CACTUS_AVAILABLE
    if CACTUS_AVAILABLE:
        logger.info("Pre-loading Cactus/Gemma 4 model (background thread)...")
        import asyncio
        await asyncio.to_thread(_get_cactus_model)
        logger.info("Cactus/Gemma 4 model ready.")
    yield

app = FastAPI(title="EMPIRE", lifespan=lifespan)
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
                # Use the fast Wav2Lip path (sub-8s) instead of the legacy LiveTalking path.
                # The endpoint also broadcasts `comment_response_video` to dashboards.
                comment_text = msg.get("text", "")
                async def _run():
                    try:
                        await api_respond_to_comment(comment=comment_text, out_height=1920)
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

    # PHASE 2: Avatar speak / TTS
    pipeline_state["status"] = "selling"
    log_event("SELLER", "Avatar going live...")
    t0 = time.time()
    lt_result = await make_avatar_speak(script)
    lt_ms = int((time.time() - t0) * 1000)
    if "error" in lt_result:
        log_event("SELLER", f"LiveTalking: {lt_result['error']} ({lt_ms}ms)")
        log_event("SELLER", "Falling back to TTS audio only...")
        try:
            audio_bytes = await text_to_speech(script)
            if audio_bytes:
                await broadcast_to_dashboards({
                    "type": "tts_audio",
                    "audio": base64.b64encode(audio_bytes).decode(),
                    "format": "mp3",
                })
        except Exception as e:
            log_event("SELLER", f"TTS fallback failed: {e}")
    else:
        log_event("SELLER", f"Avatar speaking! ({lt_ms}ms, lip-synced via WebRTC)")

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
        view = await carousel_from_video(video_path, n_frames=12, out_size=512, clean_bg=True)
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

    if transcript:
        log_event("EYES", f'Seller said: "{transcript[:200]}..."')
        await broadcast_to_dashboards({"type": "transcript", "text": transcript})
        # Fire on-device transcript extract in the background. It's CPU-slow
        # (~10s on Mac) but runs in parallel with Claude vision. When it
        # lands, we broadcast structured signals to the dashboard.
        asyncio.ensure_future(run_transcript_extract(transcript))

    if best_frames_b64:
        combined_voice = f"{voice_text}. Seller's narration: {transcript}" if transcript else voice_text
        pipeline_state["product_photo_b64"] = best_frames_b64[0]
        await broadcast_to_dashboards({"type": "phone_frame", "frame": best_frames_b64[0][:100] + "..."})
        await run_sell_pipeline(best_frames_b64[0], combined_voice)
    else:
        log_event("SYSTEM", "No usable frames extracted from video")


async def run_transcript_extract(transcript: str):
    """Background: pull structured product hints from the seller's pitch.

    Runs concurrently with Claude vision. Doesn't gate anything; arrives
    when it arrives and enriches the dashboard. ~500ms (Cactus on NPU)
    to ~15s (Cactus CPU prefill on Mac dev).
    """
    from agents.transcript_extract import extract_transcript_signals
    try:
        extract = await extract_transcript_signals(transcript)
    except Exception as e:
        logger.warning("[TRANSCRIPT_EXTRACT] failed: %s", e)
        return

    if not extract or extract.get("source") in (None, "empty"):
        return

    pipeline_state["transcript_extract"] = extract
    await broadcast_to_dashboards({"type": "transcript_extract", "data": extract})
    log_event("EYES", "On-device transcript extract ready", {
        "source": extract.get("source"),
        "latency_ms": extract.get("latency_ms"),
        "name_hint": extract.get("name_hint"),
        "claims_count": len(extract.get("claims") or []),
        "selling_points_count": len(extract.get("selling_points") or []),
    })


@app.post("/api/comment")
async def api_comment(text: str = Form(...)):
    asyncio.ensure_future(run_comment_pipeline(text))
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


def _save_render(label: str, data: bytes) -> str:
    fname = f"{label}_{int(time.time())}.mp4"
    path = RENDER_DIR / fname
    path.write_bytes(data)
    return f"/renders/{fname}"


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
        # Schedule the idle release after the response has had time to play.
        # Wav2Lip output duration ≈ TTS audio duration ≈ ~3-5s for short replies.
        # Estimate 1s per 3 words and add a safety buffer.
        word_count = len(response_text.split())
        estimated_play_ms = int(max(2500, word_count * 350) + 600)

        async def _release_after(delay_ms: int):
            await asyncio.sleep(delay_ms / 1000)
            await director.fade_to_idle()
        asyncio.create_task(_release_after(estimated_play_ms))

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


@app.post("/api/build_carousel")
async def api_build_carousel(
    file: UploadFile = File(...),
    n_frames: int = Form(12),
    out_size: int = Form(512),
    clean_bg: bool = Form(True),
):
    """Build a 3D-spin carousel from an uploaded video. For local testing
    without going through the full sell pipeline."""
    import tempfile
    contents = await file.read()
    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(contents)
        video_path = f.name
    try:
        view = await carousel_from_video(
            video_path, n_frames=n_frames, out_size=out_size, clean_bg=clean_bg,
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
