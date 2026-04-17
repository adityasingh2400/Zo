"""Avatar Director — orchestrates the dashboard's two-tier video stack.

Architecture (see /Users/aditya/.cursor/plans/seamless_avatar_continuity_*.plan.md):
  • Tier 0 (idle): always playing, looping, muted; the safety net.
  • Tier 1 (reactive): pings between two crossfading <video> elements for
    bridges, responses, pitches; fades back out to reveal Tier 0 on ended.

The Director is a pure dispatcher. It emits `play_clip` events on the
dashboard WebSocket. The dashboard owns the actual playback machinery.

The intent of this module is to be the *only* place in the backend that
decides what should be on screen at any moment. Endpoints call into it,
they never talk to the dashboard directly.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Awaitable, Callable, Literal

from agents.bridge_clips import pick_bridge_clip

logger = logging.getLogger("empire.director")

# ── Tunables ────────────────────────────────────────────────────────────────
# Mirrored on the dashboard side (LiveStage.jsx). Keep these in sync.
TIER1_CROSSFADE_MS_DEFAULT = 350
TIER0_CROSSFADE_MS_DEFAULT = 600
TIER1_FADEOUT_MS_DEFAULT = 500
READING_CHAT_HOLD_MS = 800   # how long the attentive idle plays before bridge replaces it
IDLE_ROTATE_MIN_MS = 12_000
IDLE_ROTATE_MAX_MS = 30_000

# Tier 0 candidates and weights. Order is best-effort; missing assets are
# auto-skipped so the rotation degrades gracefully if Veo hasn't shipped a
# clip yet.
TIER0_LIBRARY: list[tuple[str, str, float]] = [
    # (intent, url, weight)
    ("idle_calm",          "/clips/idle/idle_calm.mp4",          0.70),
    ("idle_attentive",     "/clips/idle/idle_attentive.mp4",     0.10),
    ("idle_thinking",      "/clips/idle/idle_thinking.mp4",      0.05),
    ("misc_glance_aside",  "/clips/idle/misc_glance_aside.mp4",  0.05),
    ("misc_hair_touch",    "/clips/idle/misc_hair_touch.mp4",    0.04),
    ("misc_sip_drink",     "/clips/idle/misc_sip_drink.mp4",     0.04),
    ("misc_walk_off_return","/clips/idle/misc_walk_off_return.mp4",0.02),
]

# When Veo M2 hasn't shipped yet, fall back to the existing 8s state video.
# This keeps M1 demoable on day one.
TIER0_FALLBACK_URL = "/states/state_idle_pose_silent_1080p.mp4"
READING_CHAT_FALLBACK_URL = TIER0_FALLBACK_URL  # swap to /clips/idle/idle_attentive.mp4 once Veo lands


class Director:
    """One-per-process avatar choreographer."""

    def __init__(self, broadcast: Callable[[dict], Awaitable[None]]):
        self._broadcast = broadcast
        self._ready = asyncio.Event()
        self._idle_task: asyncio.Task | None = None
        self._last_intent: dict[str, str] = {}     # layer -> intent (for state_sync replay)
        self._last_url: dict[str, str] = {}        # layer -> url (for state_sync replay)

    # ── Lifecycle ──────────────────────────────────────────────────────────
    def mark_ready(self) -> None:
        """Dashboard told us Tier 0 is painting frames; safe to send tier 1."""
        if not self._ready.is_set():
            logger.info("[director] stage_ready received — Tier 1 emission unlocked")
            self._ready.set()

    async def wait_ready(self, timeout: float = 5.0) -> bool:
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    # ── Core emit ──────────────────────────────────────────────────────────
    async def emit(
        self,
        layer: Literal["tier0", "tier1"],
        intent: str,
        url: str,
        *,
        loop: bool = False,
        mode: Literal["crossfade", "queue"] = "crossfade",
        fade_ms: int | None = None,
        ttl_ms: int | None = None,
        emitted_by: str = "director",
    ) -> None:
        """Send a single play_clip event. Updates last_intent for replay."""
        if fade_ms is None:
            fade_ms = TIER1_CROSSFADE_MS_DEFAULT if layer == "tier1" else TIER0_CROSSFADE_MS_DEFAULT
        msg = {
            "type": "play_clip",
            "layer": layer,
            "intent": intent,
            "url": url,
            "loop": loop,
            "mode": mode,
            "fade_ms": fade_ms,
            "ttl_ms": ttl_ms,
            "ts": time.time_ns(),
            "emitted_by": emitted_by,
        }
        self._last_intent[layer] = intent
        self._last_url[layer] = url
        logger.info("[director] emit %s/%s -> %s (mode=%s fade=%dms)",
                    layer, intent, url, mode, fade_ms)
        try:
            await self._broadcast(msg)
        except Exception:
            logger.exception("[director] broadcast failed")

    # ── Convenience ───────────────────────────────────────────────────────
    async def reading_chat(self) -> None:
        """Show the avatar reading the incoming comment. Held briefly before
        the bridge crossfades over it. Falls back to the fallback URL
        until the polished idle_attentive clip ships."""
        await self.emit(
            "tier1",
            "reading_chat",
            READING_CHAT_FALLBACK_URL,
            loop=True,
            mode="crossfade",
            ttl_ms=2000,
            emitted_by="reading_chat",
        )
        # Hold the reading_chat moment briefly so the viewer registers it
        # before the bridge takes over. Caller can race ahead with rendering.
        await asyncio.sleep(READING_CHAT_HOLD_MS / 1000)

    async def play_bridge(self, label: str) -> dict[str, Any] | None:
        """Pick a bridge from the runtime LatentSync manifest and emit it.
        Returns the chosen entry or None if no bridge available."""
        clip = pick_bridge_clip(label)
        if not clip:
            logger.warning("[director] no bridge available for label=%s — skipping", label)
            return None
        await self.emit(
            "tier1",
            f"bridge_{label}",
            clip["url"],
            loop=False,
            mode="crossfade",
            ttl_ms=4000,
            emitted_by="play_bridge",
        )
        return clip

    async def play_response(self, url: str) -> None:
        await self.emit(
            "tier1",
            "response",
            url,
            loop=False,
            mode="crossfade",
            emitted_by="play_response",
        )

    async def play_pitch(self, url: str) -> None:
        await self.emit(
            "tier1",
            "pitch",
            url,
            loop=True,
            mode="crossfade",
            emitted_by="play_pitch",
        )

    async def fade_to_idle(self) -> None:
        """Tell the dashboard to fade Tier 1 out so Tier 0 takes over."""
        await self.emit(
            "tier1",
            "idle_release",
            "",
            loop=False,
            mode="crossfade",
            fade_ms=TIER1_FADEOUT_MS_DEFAULT,
            emitted_by="fade_to_idle",
        )

    # ── Tier 0 idle rotation ──────────────────────────────────────────────
    def start_idle_rotation(self) -> None:
        if self._idle_task and not self._idle_task.done():
            return
        self._idle_task = asyncio.create_task(self._idle_loop())

    def stop_idle_rotation(self) -> None:
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()

    async def _idle_loop(self) -> None:
        # Send the calm baseline immediately so Tier 0 never starts blank.
        await self.emit(
            "tier0",
            "idle_calm",
            TIER0_FALLBACK_URL,  # will pick from library once those exist
            loop=True,
            mode="crossfade",
            emitted_by="idle_init",
        )
        while True:
            wait_ms = random.uniform(IDLE_ROTATE_MIN_MS, IDLE_ROTATE_MAX_MS)
            await asyncio.sleep(wait_ms / 1000)
            try:
                pick = self._weighted_pick()
                if pick is None:
                    continue
                intent, url, _w = pick
                await self.emit(
                    "tier0",
                    intent,
                    url,
                    loop=True,
                    mode="crossfade",
                    emitted_by="idle_rotate",
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[director] idle rotation tick failed")

    def _weighted_pick(self) -> tuple[str, str, float] | None:
        candidates = TIER0_LIBRARY
        if not candidates:
            return None
        total = sum(w for _, _, w in candidates)
        r = random.uniform(0, total)
        acc = 0.0
        for c in candidates:
            acc += c[2]
            if r <= acc:
                return c
        return candidates[-1]

    # ── Replay state for newly connected dashboards ───────────────────────
    def replay_state(self) -> dict[str, dict[str, str]]:
        """Snapshot of the latest intent on each layer; included in state_sync
        so a freshly opened dashboard can rehydrate the stage."""
        return {
            layer: {"intent": self._last_intent.get(layer, ""),
                    "url": self._last_url.get(layer, "")}
            for layer in ("tier0", "tier1")
        }
