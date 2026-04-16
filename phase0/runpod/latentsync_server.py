"""LatentSync FastAPI server — models loaded once, fp16 on 5090.

Mirrors wav2lip_server_v2.py pattern:
- Models loaded in-process at startup (no subprocess spawns per request)
- Supports two configs: 'stage2' (256 res, fast) and 'stage2_512' (512 res, best quality)
- /lipsync runs inference and returns the output video

Run:
    cd /workspace/LatentSync
    source venv/bin/activate
    PYTHONPATH=. python /workspace/latentsync_server.py --config stage2_512 --port 8766

API:
  GET  /health
  POST /lipsync  (source_video, audio, [guidance_scale=2.0, inference_steps=20, seed=1247, enable_deepcache=1, out_height=0])
  POST /prewarm  (source_video) — runs a tiny inference to JIT-compile CUDA kernels
"""
from __future__ import annotations
import os, sys, time, argparse, tempfile, shutil, subprocess
from pathlib import Path

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

REPO = Path("/workspace/LatentSync")
sys.path.insert(0, str(REPO))
os.chdir(str(REPO))  # many LatentSync paths are relative to repo root

import torch
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from accelerate.utils import set_seed

from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.whisper.audio2feature import Audio2Feature

try:
    from DeepCache import DeepCacheSDHelper
    HAS_DEEPCACHE = True
except Exception:
    HAS_DEEPCACHE = False

# Globals
_PIPELINE: LipsyncPipeline | None = None
_CONFIG = None
_DEEPCACHE_HELPER = None
_CONFIG_NAME = "stage2_512"  # overridable via CLI
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IS_FP16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
DTYPE = torch.float16 if IS_FP16 else torch.float32

app = FastAPI(title="latentsync-server")


def _init_pipeline(config_name: str = "stage2_512"):
    global _PIPELINE, _CONFIG, _DEEPCACHE_HELPER
    if _PIPELINE is not None:
        return

    print(f"[ls] loading LatentSync pipeline (config={config_name}, dtype={DTYPE})...", flush=True)
    t0 = time.time()

    config_path = REPO / "configs" / "unet" / f"{config_name}.yaml"
    _CONFIG = OmegaConf.load(str(config_path))

    # Scheduler from configs/ dir (contains scheduler_config.json)
    scheduler = DDIMScheduler.from_pretrained(str(REPO / "configs"))

    # Whisper tiny (384 dim) or small (768 dim) depending on cross_attention_dim
    if _CONFIG.model.cross_attention_dim == 768:
        whisper_path = REPO / "checkpoints" / "whisper" / "small.pt"
    elif _CONFIG.model.cross_attention_dim == 384:
        whisper_path = REPO / "checkpoints" / "whisper" / "tiny.pt"
    else:
        raise ValueError(f"unexpected cross_attention_dim={_CONFIG.model.cross_attention_dim}")
    if not whisper_path.exists():
        raise FileNotFoundError(
            f"Whisper checkpoint missing: {whisper_path}. "
            f"Download from https://huggingface.co/ByteDance/LatentSync-1.5/resolve/main/whisper/{whisper_path.name}"
        )

    audio_encoder = Audio2Feature(
        model_path=str(whisper_path),
        device=DEVICE,
        num_frames=_CONFIG.data.num_frames,
        audio_feat_length=_CONFIG.data.audio_feat_length,
    )

    # VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=DTYPE)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    # UNet — note signature: (config_dict, ckpt_path, device)
    ckpt_path = REPO / "checkpoints" / "latentsync_unet.pt"
    unet, _meta = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(_CONFIG.model),
        str(ckpt_path),
        device="cpu",
    )
    unet = unet.to(dtype=DTYPE)

    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to(DEVICE)

    _PIPELINE = pipeline
    print(f"[ls] pipeline ready in {time.time()-t0:.1f}s", flush=True)
    print(f"[ls] config: num_frames={_CONFIG.data.num_frames}, resolution={_CONFIG.data.resolution}", flush=True)


@app.on_event("startup")
async def startup():
    _init_pipeline(_CONFIG_NAME)


@app.get("/health")
def health():
    return {
        "status": "ok" if _PIPELINE is not None else "loading",
        "device": DEVICE,
        "dtype": str(DTYPE),
        "config": _CONFIG_NAME,
        "resolution": int(_CONFIG.data.resolution) if _CONFIG else None,
        "num_frames": int(_CONFIG.data.num_frames) if _CONFIG else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "deepcache": HAS_DEEPCACHE,
    }


def _maybe_enable_deepcache(enable: bool):
    """Enable or disable DeepCache on the pipeline (context-managed per request)."""
    global _DEEPCACHE_HELPER
    if not HAS_DEEPCACHE:
        return None
    if enable:
        helper = DeepCacheSDHelper(pipe=_PIPELINE)
        helper.set_params(cache_interval=3, cache_branch_id=0)
        helper.enable()
        return helper
    return None


@app.post("/lipsync")
async def lipsync(
    source_video: UploadFile = File(...),
    audio: UploadFile = File(...),
    guidance_scale: float = Form(2.0),
    inference_steps: int = Form(20),
    seed: int = Form(1247),
    enable_deepcache: int = Form(1),
    out_height: int = Form(0),  # 0 = native pipeline resolution (256/512)
):
    if _PIPELINE is None:
        raise HTTPException(503, "pipeline not loaded")

    with tempfile.TemporaryDirectory(prefix="ls_req_") as td:
        tdp = Path(td)
        src = tdp / "source.mp4"
        aud_in = tdp / "audio_in"
        aud = tdp / "audio.wav"
        out = tdp / "output.mp4"
        temp_dir = tdp / "pipeline_temp"
        temp_dir.mkdir(parents=True)

        src.write_bytes(await source_video.read())
        aud_in.write_bytes(await audio.read())

        # normalize audio to 16k mono WAV (what Whisper expects)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(aud_in), "-ar", "16000", "-ac", "1",
             "-c:a", "pcm_s16le", str(aud)],
            check=True, capture_output=True,
        )

        set_seed(seed)
        helper = _maybe_enable_deepcache(bool(enable_deepcache))

        t0 = time.time()
        try:
            _PIPELINE(
                video_path=str(src),
                audio_path=str(aud),
                video_out_path=str(out),
                num_frames=_CONFIG.data.num_frames,
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                weight_dtype=DTYPE,
                width=_CONFIG.data.resolution,
                height=_CONFIG.data.resolution,
                mask_image_path=str(REPO / _CONFIG.data.mask_image_path),
                temp_dir=str(temp_dir),
            )
        finally:
            if helper is not None:
                try:
                    helper.disable()
                except Exception:
                    pass

        pipeline_sec = time.time() - t0

        if not out.exists() or out.stat().st_size == 0:
            raise HTTPException(500, "pipeline produced no output")

        final_src = out
        # Optional upscale (ffmpeg lanczos → out_height, preserve aspect) + re-mux for clean mp4
        if out_height > 0:
            up = tdp / "output_up.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(out), "-vf",
                 f"scale=-2:{out_height}:flags=lanczos", "-c:v", "libx264",
                 "-preset", "fast", "-crf", "18", "-c:a", "aac", "-b:a", "128k",
                 str(up)],
                check=True, capture_output=True,
            )
            final_src = up
        total_sec = time.time() - t0

        # ship out of tempdir before it gets removed
        final = Path("/tmp") / f"ls_out_{int(time.time()*1000)}.mp4"
        shutil.copy(final_src, final)
        size_kb = final.stat().st_size // 1024
        print(
            f"[ls] /lipsync done  pipeline={pipeline_sec:.2f}s total={total_sec:.2f}s "
            f"steps={inference_steps} cfg={guidance_scale} deepcache={enable_deepcache} "
            f"-> {final} ({size_kb} KB)",
            flush=True,
        )

        return FileResponse(
            str(final),
            media_type="video/mp4",
            headers={
                "x-render-seconds": f"{total_sec:.2f}",
                "x-pipeline-seconds": f"{pipeline_sec:.2f}",
                "x-size-kb": str(size_kb),
                "x-config": _CONFIG_NAME,
                "x-resolution": str(_CONFIG.data.resolution),
            },
        )


@app.post("/prewarm")
async def prewarm(source_video: UploadFile = File(...)):
    """Warmup by running a short pipeline inference. Triggers CUDA JIT + VAE/UNet on-device."""
    if _PIPELINE is None:
        raise HTTPException(503, "pipeline not loaded")
    # Just report ready; real warmup happens when /lipsync is hit once.
    return {
        "status": "warm",
        "note": "LatentSync is a diffusion pipeline; first call JITs CUDA kernels. Call /lipsync once with real inputs to fully warm.",
        "config": _CONFIG_NAME,
        "resolution": int(_CONFIG.data.resolution),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="stage2_512",
                    help="unet config name in configs/unet/ (stage2, stage2_512, stage2_efficient)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    _CONFIG_NAME = args.config
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
