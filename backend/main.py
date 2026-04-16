import asyncio
import json
import time
import base64
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import BACKEND_HOST, BACKEND_PORT
from agents.eyes import analyze_with_claude, analyze_with_gemma, classify_comment_gemma, parse_voice_intent_gemma
from agents.creator import remove_background, generate_3d_model
from agents.seller import (
    generate_sales_script,
    generate_comment_response,
    text_to_speech,
    send_audio_to_livetalking,
)

# ── State ──────────────────────────────────────────────

pipeline_state: dict[str, Any] = {
    "status": "idle",
    "product_data": None,
    "product_photo_b64": None,
    "product_clean_b64": None,
    "model_3d": None,
    "sales_script": None,
    "agent_log": [],
}

dashboard_clients: list[WebSocket] = []
phone_clients: list[WebSocket] = []


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
    print(f"EMPIRE backend running on {BACKEND_HOST}:{BACKEND_PORT}")
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
            "agent_log": pipeline_state["agent_log"][-50:],
        },
    })

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "simulate_comment":
                asyncio.ensure_future(run_comment_pipeline(msg.get("text", "")))

            elif msg.get("type") == "simulate_sell":
                frame = pipeline_state.get("product_photo_b64", "")
                asyncio.ensure_future(run_sell_pipeline(
                    frame_b64=frame,
                    voice_text=msg.get("voice_text", "sell this"),
                ))
    except WebSocketDisconnect:
        dashboard_clients.remove(ws)


# ── Sell Pipeline ──────────────────────────────────────

async def run_sell_pipeline(frame_b64: str, voice_text: str):
    pipeline_state["status"] = "analyzing"
    pipeline_state["agent_log"] = []

    # Step 0: Parse voice intent (Gemma 4, on-device)
    log_event("EYES", "Parsing voice command (Gemma 4, on-device)...")
    t0 = time.time()
    intent = await parse_voice_intent_gemma(voice_text)
    intent_ms = int((time.time() - t0) * 1000)
    log_event("EYES", f"Intent parsed ({intent_ms}ms, FREE)", intent)
    await broadcast_to_dashboards({"type": "voice_intent", "data": intent})

    # Step 1: EYES (Gemma 4 fast analysis)
    log_event("EYES", "Analyzing product on-device (Gemma 4)...")
    t0 = time.time()
    gemma_result = await analyze_with_gemma(frame_b64, voice_text)
    gemma_ms = int((time.time() - t0) * 1000)
    log_event("EYES", f"On-device analysis complete ({gemma_ms}ms, FREE)", gemma_result)
    await broadcast_to_dashboards({"type": "gemma_analysis", "data": gemma_result})

    # Step 2: EYES (Claude rich analysis)
    log_event("EYES", "Deep analysis via Claude Vision (cloud)...")
    t0 = time.time()
    try:
        claude_result = await analyze_with_claude(frame_b64, voice_text)
    except Exception as e:
        claude_result = {"error": str(e)}
        log_event("EYES", f"Claude Vision error: {e}")
    claude_ms = int((time.time() - t0) * 1000)
    log_event("EYES", f"Deep analysis complete ({claude_ms}ms)", claude_result)

    product_data = claude_result if "error" not in claude_result else gemma_result
    pipeline_state["product_data"] = product_data
    await broadcast_to_dashboards({"type": "product_data", "data": product_data})

    # Step 3: CREATOR (background removal)
    pipeline_state["status"] = "creating"
    log_event("CREATOR", "Removing background...")
    t0 = time.time()
    try:
        clean_b64 = await remove_background(frame_b64)
        pipeline_state["product_clean_b64"] = clean_b64
        creator_ms = int((time.time() - t0) * 1000)
        log_event("CREATOR", f"Clean product photo ready ({creator_ms}ms)")
        await broadcast_to_dashboards({"type": "product_photo", "photo": clean_b64})
    except Exception as e:
        log_event("CREATOR", f"Background removal failed: {e}")

    # Step 3b: CREATOR (3D model, non-blocking)
    asyncio.ensure_future(run_3d_generation(frame_b64))

    # Step 4: SELLER (generate script)
    pipeline_state["status"] = "selling"
    log_event("SELLER", "Writing sales pitch...")
    t0 = time.time()
    try:
        script = await generate_sales_script(product_data, voice_text)
    except Exception as e:
        script = f"Check out this amazing product! {product_data.get('name', 'item')}."
        log_event("SELLER", f"Script generation error, using fallback: {e}")
    script_ms = int((time.time() - t0) * 1000)
    pipeline_state["sales_script"] = script
    log_event("SELLER", f"Sales pitch ready ({script_ms}ms)", {"script": script})
    await broadcast_to_dashboards({"type": "sales_script", "script": script})

    # Step 5: SELLER (TTS)
    log_event("SELLER", "Generating voice...")
    t0 = time.time()
    try:
        audio_bytes = await text_to_speech(script)
    except Exception as e:
        audio_bytes = b""
        log_event("SELLER", f"TTS error: {e}")
    tts_ms = int((time.time() - t0) * 1000)
    log_event("SELLER", f"Voice generated ({tts_ms}ms, {len(audio_bytes)} bytes)")

    if audio_bytes:
        audio_b64 = base64.b64encode(audio_bytes).decode()
        await broadcast_to_dashboards({
            "type": "tts_audio",
            "audio": audio_b64,
            "format": "mp3",
        })

    # Step 6: SELLER (send to LiveTalking for lip sync)
    if audio_bytes:
        log_event("SELLER", "Sending to avatar for lip sync...")
        t0 = time.time()
        lt_result = await send_audio_to_livetalking(audio_bytes)
        lt_ms = int((time.time() - t0) * 1000)
        if "error" in lt_result:
            log_event("SELLER", f"LiveTalking: {lt_result['error']} ({lt_ms}ms)")
        else:
            log_event("SELLER", f"Avatar lip-syncing! ({lt_ms}ms)")

    pipeline_state["status"] = "live"
    log_event("SYSTEM", "EMPIRE is LIVE. Avatar selling your product.")
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


# ── Comment Pipeline ──────────────────────────────────

async def run_comment_pipeline(comment: str):
    product_data = pipeline_state.get("product_data", {})

    # Step 1: Gemma 4 classifies + drafts response (on-device, FREE)
    log_event("EYES", f'Comment: "{comment}" — classifying on-device...')
    t0 = time.time()
    classification = await classify_comment_gemma(comment)
    class_ms = int((time.time() - t0) * 1000)
    comment_type = classification.get("type", "question")
    gemma_draft = classification.get("draft_response", "")
    log_event("EYES", f"Classified as {comment_type} ({class_ms}ms, FREE)", classification)

    # Step 2: Claude refines the response with full product context
    log_event("SELLER", "Refining response with product context (Claude)...")
    t0 = time.time()
    try:
        response_text = await generate_comment_response(comment, product_data)
    except Exception as e:
        response_text = "Thanks for the question! Let me look into that."
        log_event("SELLER", f"Response gen error: {e}")
    resp_ms = int((time.time() - t0) * 1000)
    log_event("SELLER", f"Response generated ({resp_ms}ms)", {"response": response_text})

    # TTS
    t0 = time.time()
    try:
        audio_bytes = await text_to_speech(response_text)
    except Exception as e:
        audio_bytes = b""
        log_event("SELLER", f"TTS error: {e}")
    tts_ms = int((time.time() - t0) * 1000)
    log_event("SELLER", f"Voice ready ({tts_ms}ms)")

    if audio_bytes:
        audio_b64 = base64.b64encode(audio_bytes).decode()
        await broadcast_to_dashboards({
            "type": "comment_response",
            "comment": comment,
            "response": response_text,
            "audio": audio_b64,
            "format": "mp3",
        })

        lt_result = await send_audio_to_livetalking(audio_bytes)
        if "error" not in lt_result:
            log_event("SELLER", "Avatar responding with lip sync!")
        else:
            log_event("SELLER", f"LiveTalking: {lt_result.get('error')}")


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
