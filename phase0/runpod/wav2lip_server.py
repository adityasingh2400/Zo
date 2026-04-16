"""Wav2Lip FastAPI lip-sync server — runs on the RunPod pod.

Accepts a source video + speech audio, runs Wav2Lip inference, returns mp4.

Endpoints:
  GET  /health        -> {status, gpu}
  POST /lipsync       -> form-data: video (file), audio (file), return mp4
  POST /lipsync/json  -> JSON: {video_url, audio_url} -> mp4

Latency targets (10s audio, 720p source video):
  RTX 5090: ~1-3s wall clock
  A100:     ~2-5s
  A40:      ~3-6s
"""
from __future__ import annotations
import os, sys, time, uuid, subprocess, tempfile, shutil, pathlib, logging
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import httpx
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("wav2lip-server")

WAV2LIP_ROOT = pathlib.Path("/workspace/Wav2Lip")
WORK_ROOT = pathlib.Path("/workspace/work")
WORK_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="EMPIRE Lip-Sync (Wav2Lip)")


def _info():
    cuda = torch.cuda.is_available()
    return {
        "status": "ok",
        "cuda": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "compute_capability": list(torch.cuda.get_device_capability(0)) if cuda else None,
        "wav2lip_checkpoint": (WAV2LIP_ROOT / "checkpoints" / "wav2lip_gan.pth").exists(),
    }


@app.get("/")
def root():
    return {"service": "empire-lipsync", **_info()}


@app.get("/health")
def health():
    return _info()


def _run_wav2lip(video_path: pathlib.Path, audio_path: pathlib.Path, out_path: pathlib.Path) -> float:
    """Invoke Wav2Lip's inference.py. Returns wall-clock seconds."""
    cmd = [
        str(WAV2LIP_ROOT / "venv" / "bin" / "python"),
        str(WAV2LIP_ROOT / "inference.py"),
        "--checkpoint_path", str(WAV2LIP_ROOT / "checkpoints" / "wav2lip_gan.pth"),
        "--face", str(video_path),
        "--audio", str(audio_path),
        "--outfile", str(out_path),
        "--nosmooth",  # faster
        "--pads", "0", "20", "0", "0",
        "--wav2lip_batch_size", "128",  # big batch on 5090/A100
    ]
    log.info("running: %s", " ".join(cmd))
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=WAV2LIP_ROOT, capture_output=True, text=True, timeout=180)
    dt = time.perf_counter() - t0
    log.info("wav2lip done in %.2fs, rc=%d", dt, proc.returncode)
    if proc.returncode != 0:
        log.error("stdout: %s", proc.stdout[-2000:])
        log.error("stderr: %s", proc.stderr[-2000:])
        raise RuntimeError(f"Wav2Lip failed ({proc.returncode}): {proc.stderr[-500:]}")
    return dt


@app.post("/lipsync")
async def lipsync(video: UploadFile = File(...), audio: UploadFile = File(...)):
    req_id = uuid.uuid4().hex[:8]
    work = WORK_ROOT / req_id
    work.mkdir(parents=True, exist_ok=True)

    video_path = work / f"src{pathlib.Path(video.filename or 'x.mp4').suffix or '.mp4'}"
    audio_path = work / f"audio{pathlib.Path(audio.filename or 'a.mp3').suffix or '.mp3'}"
    out_path = work / "out.mp4"

    try:
        with video_path.open("wb") as f:
            shutil.copyfileobj(video.file, f)
        with audio_path.open("wb") as f:
            shutil.copyfileobj(audio.file, f)

        t_render = _run_wav2lip(video_path, audio_path, out_path)
        return FileResponse(
            path=out_path,
            media_type="video/mp4",
            headers={"X-Render-Seconds": f"{t_render:.3f}", "X-Request-Id": req_id},
        )
    except Exception as e:
        raise HTTPException(500, str(e))


class _JsonReq:
    video_url: str
    audio_url: str


@app.post("/lipsync/json")
async def lipsync_json(req: dict):
    req_id = uuid.uuid4().hex[:8]
    work = WORK_ROOT / req_id
    work.mkdir(parents=True, exist_ok=True)

    video_url = req.get("video_url")
    audio_url = req.get("audio_url")
    if not video_url or not audio_url:
        raise HTTPException(400, "video_url and audio_url required")

    video_path = work / "src.mp4"
    audio_path = work / "audio.mp3"
    out_path = work / "out.mp4"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            for url, path in [(video_url, video_path), (audio_url, audio_path)]:
                r = await client.get(url)
                r.raise_for_status()
                path.write_bytes(r.content)

        t_render = _run_wav2lip(video_path, audio_path, out_path)
        return FileResponse(
            path=out_path, media_type="video/mp4",
            headers={"X-Render-Seconds": f"{t_render:.3f}", "X-Request-Id": req_id},
        )
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
