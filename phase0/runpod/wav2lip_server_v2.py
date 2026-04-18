"""Wav2Lip FastAPI server — in-process model loading, face-detection cache.

Key optimizations vs v1:
  1. Load Wav2Lip + RetinaFace ONCE at server startup, keep in GPU memory.
  2. Cache face-detection bounding boxes per source video (hashed by sha256).
     State videos never change → face detection runs exactly once per state video.
  3. Direct function calls, no subprocess fork per request.
  4. Configurable out_height (720 default for vertical 9:16 presentation).

Expected latency (10s audio, 720p source, RTX 5090):
  - Cold first run: 40-60s (model load + face detect)
  - Warm + uncached source: 8-15s
  - Warm + cached source: 3-6s  <-- this is what we want at demo time

Endpoints:
  GET  /health            -> status + gpu + warmup
  POST /lipsync           -> form-data video+audio, returns mp4
  POST /prewarm           -> {source_path_on_pod} -> pre-detects face + caches

The state videos can be uploaded once to the pod, then prewarmed at startup
so the very first real render hits the fast path.
"""
from __future__ import annotations
import os, sys, time, uuid, tempfile, shutil, pathlib, hashlib, pickle, threading, logging, io, contextlib, subprocess

# wav2lip import prep — must be before FastAPI app so module loads cleanly
WAV2LIP_ROOT = pathlib.Path("/workspace/Wav2Lip")
WORK_ROOT = pathlib.Path("/workspace/work")
CACHE_ROOT = pathlib.Path("/workspace/facecache")
WORK_ROOT.mkdir(parents=True, exist_ok=True)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

os.chdir(WAV2LIP_ROOT)
sys.path.insert(0, str(WAV2LIP_ROOT))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import httpx
import torch
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("w2l-v2")

# ─── Import & warm up Wav2Lip ────────────────────────────────────────────────
import audio as w2l_audio
from models import Wav2Lip
from batch_face import RetinaFace

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEL_STEP = 16
IMG_SIZE = 96
FACE_BATCH = 512
WAV2LIP_BATCH = 128


def _load_ckpt(path: str):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    return ck["state_dict"] if "state_dict" in ck else ck


def _load_wav2lip(path: str):
    model = Wav2Lip()
    sd = _load_ckpt(path)
    # strip 'module.' prefix if present
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model.to(DEVICE).eval()


log.info("loading wav2lip weights...")
t0 = time.perf_counter()
MODEL = _load_wav2lip(str(WAV2LIP_ROOT / "checkpoints" / "wav2lip_gan.pth"))
log.info("  wav2lip loaded in %.2fs", time.perf_counter() - t0)

log.info("loading retinaface...")
t0 = time.perf_counter()
DETECTOR = RetinaFace(gpu_id=0, model_path=str(WAV2LIP_ROOT / "checkpoints" / "mobilenet.pth"), network="mobilenet")
log.info("  retinaface loaded in %.2fs", time.perf_counter() - t0)

RENDER_LOCK = threading.Lock()   # single-GPU, serialize
CACHE_LOCK = threading.Lock()
FACE_CACHE: dict[str, tuple] = {}   # {sha256: (boxes, smoothed_boxes, fps, frames_count)}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_smoothed(boxes, T=5):
    boxes = np.array(boxes)
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


# In-process cache of decoded frames keyed by (sha256, out_height).
# Without this we re-decode the entire 8s 1080p source video on every call
# (~700MB raw decode = 0.5-1.5s wasted per request). With it, a warm call
# skips all video I/O and the cache hit is just a dict lookup.
# Memory cost: ~250MB per cached source × 11 sources = ~2.7GB. Well within
# the 5090's host RAM budget.
_FRAME_MEMCACHE: dict[str, tuple[list, float]] = {}


def _face_detect_cached(source_path: pathlib.Path, out_height: int):
    """Return (frames_rgb_full, smoothed_boxes, fps). Caches bbox on disk by
    source sha256, AND caches decoded frames in-memory keyed by the same."""
    key = f"{_sha256(source_path)}_h{out_height}"
    cache_file = CACHE_ROOT / f"{key}.pkl"

    cached = _FRAME_MEMCACHE.get(key)
    if cached is not None:
        frames, fps = cached
    else:
        t0_decode = time.perf_counter()
        vs = cv2.VideoCapture(str(source_path))
        fps = vs.get(cv2.CAP_PROP_FPS) or 24.0
        frames = []
        while True:
            ok, frame = vs.read()
            if not ok:
                break
            aspect = frame.shape[1] / frame.shape[0]
            frame = cv2.resize(frame, (int(out_height * aspect), out_height))
            frames.append(frame)
        vs.release()
        _FRAME_MEMCACHE[key] = (frames, fps)
        log.info("frame memcache MISS (key=%s) decode=%.3fs frames=%d",
                 key[:12], time.perf_counter() - t0_decode, len(frames))

    with CACHE_LOCK:
        if cache_file.exists():
            t0 = time.perf_counter()
            with cache_file.open("rb") as f:
                smoothed = pickle.load(f)
            log.info("face cache HIT  (key=%s)  load=%.3fs  frames=%d", key[:12], time.perf_counter()-t0, len(frames))
            return frames, smoothed, fps

    # cache miss: detect
    t0 = time.perf_counter()
    boxes = []
    # Larger pad gives the predicted mouth more vertical area to land on a
    # closed-mouth substrate, fixing the "tiny pasted mouth" look on idle
    # poses. pad_top adds forehead room (helps the face crop frame the chin
    # better), pad_bottom adds chin room (where new mouth pixels actually go).
    pad_top, pad_bottom, pad_left, pad_right = 10, 30, 5, 5

    nb = (len(frames) + FACE_BATCH - 1) // FACE_BATCH
    last_box = None
    for b in range(nb):
        batch = frames[b * FACE_BATCH:(b + 1) * FACE_BATCH]
        faces = DETECTOR(batch)
        for i, face_list in enumerate(faces):
            if not face_list:
                if last_box is None:
                    found = None
                    for jj in range(i + 1, len(faces)):
                        if faces[jj]:
                            bb = faces[jj][0][0]
                            y1_, y2_, x1_, x2_ = max(0, int(bb[1]) - pad_top), int(bb[3]) + pad_bottom, max(0, int(bb[0]) - pad_left), int(bb[2]) + pad_right
                            found = [y1_, y2_, x1_, x2_]
                            break
                    if found is None:
                        raise RuntimeError(f"no face detected anywhere starting at frame {b*FACE_BATCH+i}")
                    last_box = found
                log.warning("face miss on frame %d -> reusing last box", b * FACE_BATCH + i)
                boxes.append(list(last_box))
                continue
            bbox = face_list[0][0]
            y1, y2, x1, x2 = max(0, int(bbox[1]) - pad_top), int(bbox[3]) + pad_bottom, max(0, int(bbox[0]) - pad_left), int(bbox[2]) + pad_right
            last_box = [y1, y2, x1, x2]
            boxes.append(last_box)

    smoothed = _get_smoothed(boxes, T=5)
    with CACHE_LOCK:
        with cache_file.open("wb") as f:
            pickle.dump(smoothed, f)
    log.info("face cache MISS (key=%s)  detect=%.2fs  frames=%d", key[:12], time.perf_counter()-t0, len(frames))
    return frames, smoothed, fps


def _run_lipsync(source_path: pathlib.Path, audio_path: pathlib.Path, out_path: pathlib.Path, out_height: int = 720) -> dict:
    """In-process Wav2Lip render. Returns {total_sec, detect_sec, predict_sec, mux_sec}."""
    t_all = time.perf_counter()
    timings = {}

    # 1) faces (cached per source)
    t0 = time.perf_counter()
    frames, smoothed_boxes, fps = _face_detect_cached(source_path, out_height)
    timings["detect_sec"] = round(time.perf_counter() - t0, 3)

    # 2) audio → mel
    t0 = time.perf_counter()
    # ensure wav
    wav_path = audio_path
    if audio_path.suffix.lower() != ".wav":
        wav_tmp = out_path.with_suffix(".wav")
        subprocess.run(["ffmpeg", "-y", "-i", str(audio_path), "-ar", "16000", "-ac", "1", str(wav_tmp),
                        "-loglevel", "error"], check=True)
        wav_path = wav_tmp
    wav = w2l_audio.load_wav(str(wav_path), 16000)
    mel = w2l_audio.melspectrogram(wav)
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise RuntimeError("mel contains NaN")
    mel_idx_mult = 80.0 / fps
    mel_chunks = []
    i = 0
    while True:
        s = int(i * mel_idx_mult)
        if s + MEL_STEP > mel.shape[1]:
            mel_chunks.append(mel[:, mel.shape[1] - MEL_STEP:])
            break
        mel_chunks.append(mel[:, s:s + MEL_STEP])
        i += 1

    # clip frames to mel length (if audio shorter than video, cut; if longer, loop)
    n_mels = len(mel_chunks)
    if n_mels <= len(frames):
        frames = frames[:n_mels]
    else:
        # loop frames to match audio length
        reps = (n_mels + len(frames) - 1) // len(frames)
        frames = (frames * reps)[:n_mels]
        # extend smoothed boxes too
        sb = list(smoothed_boxes)
        smoothed_boxes = np.array((sb * reps)[:n_mels])
    timings["audio_sec"] = round(time.perf_counter() - t0, 3)

    # 3) wav2lip predict
    t0 = time.perf_counter()
    temp_avi = out_path.with_suffix(".avi")
    fw, fh = frames[0].shape[1], frames[0].shape[0]
    writer = cv2.VideoWriter(str(temp_avi), cv2.VideoWriter_fourcc(*"DIVX"), fps, (fw, fh))

    def datagen():
        img_batch, mel_batch, frm_batch, coord_batch = [], [], [], []
        for idx, m in enumerate(mel_chunks):
            f = frames[idx].copy()
            y1, y2, x1, x2 = smoothed_boxes[idx].astype(int)
            face = f[y1:y2, x1:x2]
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            img_batch.append(face)
            mel_batch.append(m)
            frm_batch.append(f)
            coord_batch.append((y1, y2, x1, x2))
            if len(img_batch) == WAV2LIP_BATCH:
                yield _prep(img_batch, mel_batch), frm_batch, coord_batch
                img_batch, mel_batch, frm_batch, coord_batch = [], [], [], []
        if img_batch:
            yield _prep(img_batch, mel_batch), frm_batch, coord_batch

    def _prep(imgs, mels):
        imgs = np.asarray(imgs)
        mels = np.asarray(mels)
        img_masked = imgs.copy()
        img_masked[:, IMG_SIZE // 2:] = 0
        img_concat = np.concatenate((img_masked, imgs), axis=3) / 255.0
        mels = np.reshape(mels, [len(mels), mels.shape[1], mels.shape[2], 1])
        img_t = torch.FloatTensor(np.transpose(img_concat, (0, 3, 1, 2))).to(DEVICE)
        mel_t = torch.FloatTensor(np.transpose(mels, (0, 3, 1, 2))).to(DEVICE)
        return img_t, mel_t

    # Soft-edge mask for paste-back: feather the patch boundary so the seam
    # between predicted mouth pixels and original face pixels is invisible.
    # Built once per frame size since the box dims are stable across the take.
    _mask_cache: dict[tuple[int, int], np.ndarray] = {}

    def _soft_mask(h: int, w: int) -> np.ndarray:
        """3-channel float mask, 1.0 in the center, fading to 0 at edges."""
        key = (h, w)
        if key not in _mask_cache:
            mask = np.ones((h, w), dtype=np.float32)
            # Feather width: ~10% of the smaller dimension, capped at 24px
            # so we don't over-blend on tight crops.
            feather = max(4, min(24, int(min(h, w) * 0.10)))
            for i in range(feather):
                a = (i + 1) / (feather + 1)
                mask[i, :] *= a
                mask[h - 1 - i, :] *= a
                mask[:, i] *= a
                mask[:, w - 1 - i] *= a
            _mask_cache[key] = np.dstack([mask, mask, mask])
        return _mask_cache[key]

    def _sharpen(img: np.ndarray) -> np.ndarray:
        """Unsharp mask: subtract a blurred copy from the original to boost
        edge contrast. Compensates for the soft Wav2Lip 96px upscale.
        ~3ms per face crop, no model needed."""
        # 1.4 sigma gaussian + 1.5 amount sharpen — calibrated to lift
        # tooth edges and lip definition without ringing artifacts.
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.4)
        sharp = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
        return sharp

    with torch.no_grad():
        for (img_t, mel_t), frm_batch, coord_batch in datagen():
            pred = MODEL(mel_t, img_t).cpu().numpy().transpose(0, 2, 3, 1) * 255.0
            for p, f, (y1, y2, x1, x2) in zip(pred, frm_batch, coord_batch):
                ph, pw = y2 - y1, x2 - x1
                p = cv2.resize(p.astype(np.uint8), (pw, ph), interpolation=cv2.INTER_LANCZOS4)
                # Sharpen the predicted patch BEFORE blending so the boost
                # only affects new pixels (mouth + chin), not the original
                # face skin which is already crisp.
                p = _sharpen(p)
                # Alpha-blend the predicted patch over the original frame
                # using the feathered mask. Hides the rectangular seam.
                mask = _soft_mask(ph, pw)
                original = f[y1:y2, x1:x2].astype(np.float32)
                blended = original * (1.0 - mask) + p.astype(np.float32) * mask
                f[y1:y2, x1:x2] = blended.astype(np.uint8)
                writer.write(f)
    writer.release()
    timings["predict_sec"] = round(time.perf_counter() - t0, 3)

    # 4) mux with original audio. veryfast preset is ~10% slower than
    # ultrafast but produces noticeably crisper video at the same crf.
    # crf 17 (was 18) gives slight quality boost — still well under
    # the network bandwidth cost of streaming back from the pod.
    # yuv420p kept for browser compat. -threads 0 uses all cores.
    t0 = time.perf_counter()
    subprocess.run([
        "ffmpeg", "-y", "-i", str(temp_avi), "-i", str(audio_path),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p",
        "-threads", "0",
        "-c:a", "aac", "-shortest", "-loglevel", "error", str(out_path),
    ], check=True)
    temp_avi.unlink(missing_ok=True)
    timings["mux_sec"] = round(time.perf_counter() - t0, 3)

    timings["total_sec"] = round(time.perf_counter() - t_all, 3)
    log.info("render timings: %s", timings)
    return timings


# ─── FastAPI ─────────────────────────────────────────────────────────────────

app = FastAPI(title="EMPIRE Wav2Lip v2")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda": torch.cuda.is_available(),
        "torch": torch.__version__,
        "models_loaded": True,
        "face_cache_size": len(list(CACHE_ROOT.glob("*.pkl"))),
    }


@app.post("/lipsync_fast")
async def lipsync_fast(source_path: str = Form(...), audio: UploadFile = File(...), out_height: int = Form(1080)):
    """FAST path: source video already on the pod. Saves upload overhead per call."""
    req_id = uuid.uuid4().hex[:8]
    work = WORK_ROOT / req_id
    work.mkdir(parents=True, exist_ok=True)
    src = pathlib.Path(source_path)
    if not src.exists():
        raise HTTPException(400, f"source_path not found on pod: {source_path}")
    audio_path = work / ("audio" + (pathlib.Path(audio.filename or "a.mp3").suffix or ".mp3"))
    out_path = work / "out.mp4"
    try:
        with audio_path.open("wb") as f:
            shutil.copyfileobj(audio.file, f)
        with RENDER_LOCK:
            timings = _run_lipsync(src, audio_path, out_path, out_height=out_height)
        return FileResponse(
            path=out_path, media_type="video/mp4",
            headers={
                "X-Request-Id": req_id,
                "X-Total-Sec": str(timings["total_sec"]),
                "X-Detect-Sec": str(timings["detect_sec"]),
                "X-Predict-Sec": str(timings["predict_sec"]),
                "X-Mux-Sec": str(timings["mux_sec"]),
            },
        )
    except Exception as e:
        log.exception("fast render failed")
        raise HTTPException(500, str(e))


@app.post("/lipsync")
async def lipsync(video: UploadFile = File(...), audio: UploadFile = File(...), out_height: int = Form(720)):
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

        with RENDER_LOCK:
            timings = _run_lipsync(video_path, audio_path, out_path, out_height=out_height)

        return FileResponse(
            path=out_path,
            media_type="video/mp4",
            headers={
                "X-Request-Id": req_id,
                "X-Total-Sec": str(timings["total_sec"]),
                "X-Detect-Sec": str(timings["detect_sec"]),
                "X-Predict-Sec": str(timings["predict_sec"]),
                "X-Mux-Sec": str(timings["mux_sec"]),
            },
        )
    except Exception as e:
        log.exception("render failed")
        raise HTTPException(500, str(e))


@app.post("/prewarm")
async def prewarm(body: dict):
    """Precompute face cache for a source video on the pod."""
    path = pathlib.Path(body["path"])
    out_height = int(body.get("out_height", 720))
    if not path.exists():
        raise HTTPException(404, f"not on pod: {path}")
    with RENDER_LOCK:
        frames, boxes, fps = _face_detect_cached(path, out_height)
    return {"frames": len(frames), "fps": fps, "box_count": len(boxes), "path": str(path), "out_height": out_height}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010, log_level="info")
