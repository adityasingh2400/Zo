import io
import base64
import httpx
from PIL import Image
from rembg import remove
from config import RUNPOD_POD_IP, RUNPOD_TRIPOSR_PORT


async def remove_background(frame_b64: str) -> str:
    """Remove background from product photo, return clean PNG as base64."""
    img_bytes = base64.b64decode(frame_b64)
    img = Image.open(io.BytesIO(img_bytes))
    clean = remove(img)

    buf = io.BytesIO()
    clean.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


async def generate_3d_model(frame_b64: str) -> dict | None:
    """Send product image to TripoSR on RunPod, get back GLB mesh URL."""
    if not RUNPOD_POD_IP:
        return None

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"http://{RUNPOD_POD_IP}:{RUNPOD_TRIPOSR_PORT}/generate",
                json={"image": frame_b64, "output_format": "glb"},
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        return {"error": str(e)}
