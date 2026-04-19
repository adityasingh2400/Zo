"""Phone-as-camera uploader.

Replaces drag-drop with a phone-streamed video flow:
  1. Operator opens dashboard → /api/phone/session creates a session,
     dashboard renders a QR pointing at /phone/<sid> on the Mac's LAN IP.
  2. Phone scans QR → loads the mobile recorder page over LAN.
  3. Phone hits "Record" → MediaRecorder streams chunks via WS to
     /ws/phone/upload/<sid> as binary frames.
  4. Phone hits "Stop" → server flushes the assembled file to disk and
     hands it to run_video_sell_pipeline as if it were a drag-drop.

Everything stays on LAN — no cloud relay. Latency is WiFi-bound.

Session registry lives in memory only — sessions expire after
SESSION_TTL_SECONDS so a long-running backend doesn't accumulate stale
state. Single-process deployment is fine for now (no Redis / cross-
process sharing); when we scale to multiple uvicorn workers we'd need
to push this into a shared store but the demo runs on one worker.
"""
from __future__ import annotations

import logging
import os
import secrets
import socket
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("empire.phone_uploader")

SESSION_TTL_SECONDS = 600   # 10 min — long enough to scan-and-record at a leisurely pace
_SESSIONS: dict[str, "PhoneSession"] = {}


@dataclass
class PhoneSession:
    """Lifecycle state for one phone-upload flow.

    status transitions:
        pending  → phone hasn't connected the WS yet
        connected → WS open, waiting for "start" message
        recording → phone is actively streaming chunks
        uploading → phone sent "end", we're flushing + kicking pipeline
        complete  → pipeline kicked off, session is done
        failed    → something errored; `error` carries the string
    """
    session_id: str
    created_at: float
    status: str = "pending"
    bytes_received: int = 0
    chunks_count: int = 0
    mime_type: str = ""
    file_path: str | None = None
    error: str | None = None
    # Set when run_video_sell_pipeline kicks off so the dashboard can
    # cross-reference the phone session with the pipeline's request_id.
    request_id: str | None = None


def lan_ip() -> str:
    """Best-effort Mac LAN IP for QR encoding — opens a UDP socket to a
    public address (no packets actually sent, getsockname() reads the
    routing decision) and returns the local source IP. Falls back to
    127.0.0.1 if the machine has no internet path; the QR still encodes
    a valid URL but the phone will only reach it if it's on the same
    interface (e.g., USB-tethered)."""
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(2.0)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        if s is not None:
            try:
                s.close()
            except Exception:
                pass


def create_session() -> PhoneSession:
    """Generate a fresh phone session. Reaps expired sessions on the
    way in so the registry stays small."""
    now = time.time()
    expired = [k for k, s in _SESSIONS.items() if now - s.created_at > SESSION_TTL_SECONDS]
    for k in expired:
        old = _SESSIONS.pop(k, None)
        if old and old.file_path:
            # Best-effort cleanup of any leftover temp file. Sessions
            # that completed have already been picked up by the pipeline
            # which reads the file once and is fine if it's gone after.
            try:
                Path(old.file_path).unlink(missing_ok=True)
            except Exception:
                pass

    sid = secrets.token_urlsafe(8)
    session = PhoneSession(session_id=sid, created_at=now)
    _SESSIONS[sid] = session
    logger.info("[phone] new session %s (active=%d)", sid, len(_SESSIONS))
    return session


def get_session(session_id: str) -> PhoneSession | None:
    return _SESSIONS.get(session_id)


def open_upload_file(session: PhoneSession, suffix: str) -> Path:
    """Allocate a tempfile path for the streamed upload and stash it on
    the session. Caller is responsible for writing chunks; we just pick
    the path so the tempfile module's race-free name allocation does
    its job. Suffix should be the file extension (e.g. .webm / .mp4)
    so ffmpeg downstream can probe the container correctly."""
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=f"phone_{session.session_id}_")
    os.close(fd)
    session.file_path = path
    return Path(path)


def session_summary(session: PhoneSession) -> dict:
    """Serialisable shape used in WS broadcasts to the dashboard."""
    return {
        "session_id": session.session_id,
        "status": session.status,
        "bytes_received": session.bytes_received,
        "chunks_count": session.chunks_count,
        "mime_type": session.mime_type,
        "request_id": session.request_id,
        "error": session.error,
    }
