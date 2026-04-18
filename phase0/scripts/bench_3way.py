"""3-way lip sync benchmark: LatentSync vs Wav2Lip vs (optional) Sonic.

Runs against local servers on pod via SSH tunnel. Inputs:
  source video: state_pitching_pose_speaking_1080p.mp4 (1080x1920, 8s, 24fps)
  audio:        pitch_10s.mp3 (~10s TTS)

Protocol (per model):
  1. POST /prewarm (face cache)
  2. Discard first warm-up render
  3. Fire 3 timed renders, record p50 latency + output resolution + file size
  4. Save first output to bench/results/3way/<model>.mp4

Output:
  bench/results/3way/summary.json with full numbers
  bench/results/3way/<model>.mp4 for visual side-by-side

Usage:
  # Ensure SSH tunnel is open:
  #   ssh -N -L 8010:localhost:8010 -L 8766:localhost:8766 root@<pod>
  # Then:
  python phase0/scripts/bench_3way.py
"""
from __future__ import annotations
import os, sys, time, json, pathlib, statistics, argparse
import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_1080 = ROOT / "assets" / "states" / "state_pitching_pose_speaking_1080p.mp4"
SRC_720  = ROOT / "assets" / "states" / "state_pitching_pose_speaking.mp4"
AUDIO    = ROOT / "bench" / "audio" / "pitch_10s.mp3"

OUT_DIR = ROOT / "bench" / "results" / "3way"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    # wav2lip: field name 'video', prewarm is a JSON path on the pod (not an upload)
    "wav2lip": {
        "url": "http://localhost:8010",
        "source_field": "video",
        "params": {"out_height": "720"},
        "source": SRC_720,
        "prewarm_style": "pod_path",  # body={"path": "/workspace/state_pitching_pose_speaking.mp4"}
        "prewarm_pod_path": "/workspace/state_pitching_pose_speaking.mp4",
    },
    # latentsync: field name 'source_video', prewarm takes the file
    "latentsync": {
        "url": "http://localhost:8766",
        "source_field": "source_video",
        "params": {"inference_steps": "20", "guidance_scale": "2.0", "enable_deepcache": "1", "out_height": "1080"},
        "source": SRC_1080,
        "prewarm_style": "upload",
    },
}

def _probe(p: pathlib.Path) -> dict:
    import subprocess, shutil
    if not shutil.which("ffprobe"):
        return {}
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,duration,r_frame_rate,codec_name",
         "-of", "json", str(p)], capture_output=True, text=True,
    )
    try:
        j = json.loads(r.stdout)
        s = j["streams"][0]
        return {"width": s["width"], "height": s["height"], "fps": s["r_frame_rate"], "duration": s.get("duration"), "codec": s.get("codec_name")}
    except Exception:
        return {}


def health(url: str, timeout: int = 5) -> dict | None:
    try:
        r = requests.get(f"{url}/health", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  health fail @ {url}: {e}")
        return None


def prewarm(cfg: dict, src: pathlib.Path) -> float:
    t0 = time.perf_counter()
    url = cfg["url"]
    if cfg["prewarm_style"] == "pod_path":
        r = requests.post(f"{url}/prewarm", json={"path": cfg["prewarm_pod_path"]}, timeout=300)
    else:
        with src.open("rb") as f:
            r = requests.post(
                f"{url}/prewarm",
                files={cfg["source_field"]: ("src.mp4", f, "video/mp4")},
                timeout=300,
            )
    r.raise_for_status()
    return time.perf_counter() - t0


def call_lipsync(cfg: dict, src: pathlib.Path, audio: pathlib.Path, params: dict, save_to: pathlib.Path | None = None) -> dict:
    t0 = time.perf_counter()
    url = cfg["url"]
    field = cfg["source_field"]
    with src.open("rb") as f_src, audio.open("rb") as f_aud:
        r = requests.post(
            f"{url}/lipsync",
            files={
                field: ("src.mp4", f_src, "video/mp4"),
                "audio": ("pitch.mp3", f_aud, "audio/mpeg"),
            },
            data=params,
            timeout=600,
        )
    r.raise_for_status()
    elapsed = time.perf_counter() - t0
    out = {"client_wall_sec": round(elapsed, 2)}
    # server-reported time (if header present)
    srv = r.headers.get("x-render-seconds")
    if srv:
        try:
            out["server_render_sec"] = float(srv)
        except ValueError:
            pass
    size_kb = r.headers.get("x-size-kb")
    if size_kb:
        out["size_kb"] = int(size_kb)
    if save_to:
        save_to.write_bytes(r.content)
        out["saved"] = str(save_to)
        out["resolution"] = _probe(save_to)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()), help="subset to run")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    print(f"sources:\n  720p = {SRC_720}\n  1080p = {SRC_1080}\n  audio = {AUDIO}")
    for p in (SRC_720, SRC_1080, AUDIO):
        if not p.exists():
            sys.exit(f"missing input: {p}")

    summary = {}
    for model in args.models:
        if model not in MODELS:
            print(f"SKIP unknown model {model}")
            continue
        cfg = MODELS[model]
        url, params, src = cfg["url"], cfg["params"], cfg["source"]
        print(f"\n==== {model} @ {url}  src={src.name} ====")

        h = health(url)
        if not h:
            print(f"  {model}: health check failed — server probably down, skipping")
            summary[model] = {"error": "server unreachable"}
            continue
        print(f"  health: {h}")

        # Prewarm (face cache / GPU warmup)
        try:
            pw = prewarm(cfg, src)
            print(f"  prewarm: {pw:.2f}s")
        except Exception as e:
            print(f"  prewarm failed ({e}) — continuing anyway")

        # Warmup render (discarded)
        try:
            w = call_lipsync(cfg, src, AUDIO, params)
            print(f"  warmup render: {w}")
        except Exception as e:
            print(f"  warmup FAILED: {e}")
            summary[model] = {"error": f"warmup failed: {e}"}
            continue

        # Timed runs
        runs = []
        for i in range(args.runs):
            save = OUT_DIR / f"{model}_run{i+1}.mp4" if i == 0 else None
            try:
                rr = call_lipsync(cfg, src, AUDIO, params, save_to=save)
                print(f"  run {i+1}: {rr}")
                runs.append(rr)
            except Exception as e:
                print(f"  run {i+1} FAILED: {e}")

        if runs:
            wall = [r["client_wall_sec"] for r in runs]
            server = [r.get("server_render_sec") for r in runs if r.get("server_render_sec")]
            summary[model] = {
                "runs": runs,
                "p50_client_sec": round(statistics.median(wall), 2),
                "p50_server_sec": round(statistics.median(server), 2) if server else None,
                "resolution": runs[0].get("resolution"),
                "source": src.name,
                "params": params,
            }
        else:
            summary[model] = {"error": "no successful runs"}

    out = OUT_DIR / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n=== SUMMARY ({out}) ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
