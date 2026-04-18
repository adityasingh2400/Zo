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
IDLE_ROTATE_MIN_MS = 8_000   # tighter cadence so demo viewer sees variety quickly
IDLE_ROTATE_MAX_MS = 18_000

# Tier 0 = looping idle clips. Each one is symmetric (boomerangable) and
# meant to play indefinitely under the reactive layer. Director rotates
# between them every 12-30s.
#
# Asymmetric clips like sip_drink and walk_off_return CANNOT loop without
# looking unnatural (reversed sip = vomit, reversed walk = moonwalk), so
# they live in TIER1_INTERJECTIONS instead and play as one-shot crossfades
# with the looping idle still painting underneath.
# (intent, dashboard URL, weight, pod path for Wav2Lip substrate)
# The pod_path is the file the Wav2Lip server reads when this idle is the
# currently-active Tier 0; matching the response substrate to the playing
# idle keeps body language continuous through the crossfade.
TIER0_LIBRARY: list[tuple[str, str, float, str]] = [
    ("idle_calm",            "/states/idle/idle_calm.mp4",            0.70, "/workspace/idle/idle_calm.mp4"),
    ("idle_reading_comments","/states/idle/idle_reading_comments.mp4",0.15, "/workspace/idle/idle_reading_comments.mp4"),
    ("idle_thinking",        "/states/idle/idle_thinking.mp4",        0.05, "/workspace/idle/idle_thinking.mp4"),
    ("misc_glance_aside",    "/states/idle/misc_glance_aside.mp4",    0.05, "/workspace/idle/misc_glance_aside.mp4"),
    ("misc_hair_touch",      "/states/idle/misc_hair_touch.mp4",      0.05, "/workspace/idle/misc_hair_touch.mp4"),
]

# Tier 1 one-shot interjections. Director picks one occasionally and plays
# it as a crossfaded one-shot over the always-on Tier 0 idle. They start
# and end close to the anchor pose so the crossfade hides any mismatch.
TIER1_INTERJECTIONS: list[tuple[str, str, float, str]] = [
    ("misc_sip_drink",      "/states/idle/misc_sip_drink.mp4",      0.6, ""),
    ("misc_walk_off_return","/states/idle/misc_walk_off_return.mp4",0.4, ""),
]

# Probability per idle-rotation tick that we play a Tier 1 interjection
# instead of swapping the Tier 0 clip. At 8-18s rotation cadence with p=0.35,
# expect one sip / walk-off roughly every 30-50s — frequent enough to feel
# alive on stage without becoming twitchy.
INTERJECTION_PROBABILITY = 0.35

# When Veo idle library hasn't shipped yet, fall back to the existing 8s
# silent state video so the dashboard never starts blank.
TIER0_FALLBACK_URL = "/states/state_idle_pose_silent_1080p.mp4"
READING_CHAT_FALLBACK_URL = "/states/idle/idle_reading_comments.mp4"


class Director:
    """One-per-process avatar choreographer."""

    # Default Wav2Lip substrate (matches POD_SPEAKING_1080P in config). Used
    # if no Tier 0 idle is currently active (early startup) — but in practice
    # the idle rotation kicks immediately so this is rarely hit.
    DEFAULT_SUBSTRATE_POD_PATH = "/workspace/state_pitching_pose_speaking_1080p.mp4"

    def __init__(self, broadcast: Callable[[dict], Awaitable[None]]):
        self._broadcast = broadcast
        self._ready = asyncio.Event()
        self._idle_task: asyncio.Task | None = None
        self._last_intent: dict[str, str] = {}     # layer -> intent (for state_sync replay)
        self._last_url: dict[str, str] = {}        # layer -> url (for state_sync replay)
        # Pod-side path of the substrate that matches whatever Tier 0 is
        # currently painting. Wav2Lip pulls this so the response inherits
        # the body language of the visible idle clip.
        self._current_substrate_pod_path: str = self.DEFAULT_SUBSTRATE_POD_PATH

    def current_substrate_pod_path(self) -> str:
        """The Wav2Lip server-side path that matches the visible Tier 0 clip."""
        return self._current_substrate_pod_path

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
        # Seed the substrate tracker before the first emit so any comment that
        # races in at boot uses the right pod-side substrate.
        self._current_substrate_pod_path = TIER0_LIBRARY[0][3]
        await self.emit(
            "tier0",
            TIER0_LIBRARY[0][0],
            TIER0_LIBRARY[0][1],
            loop=True,
            mode="crossfade",
            emitted_by="idle_init",
        )
        while True:
            wait_ms = random.uniform(IDLE_ROTATE_MIN_MS, IDLE_ROTATE_MAX_MS)
            await asyncio.sleep(wait_ms / 1000)
            try:
                # Tier 1 interjections (sip, walk-off) don't change the
                # underlying Tier 0 substrate; the Director keeps the active
                # idle pose paired with the next response.
                if random.random() < INTERJECTION_PROBABILITY and TIER1_INTERJECTIONS:
                    pick = self._weighted_pick(TIER1_INTERJECTIONS)
                    if pick:
                        intent, url, _w, _pod = pick
                        await self.emit(
                            "tier1",
                            intent,
                            url,
                            loop=False,
                            mode="crossfade",
                            emitted_by="idle_interjection",
                        )
                        continue

                # Rotate the looping idle on Tier 0 + remember its pod-side
                # substrate so the next Wav2Lip render uses the matching pose.
                pick = self._weighted_pick(TIER0_LIBRARY)
                if pick is None:
                    continue
                intent, url, _w, pod_path = pick
                if pod_path:
                    self._current_substrate_pod_path = pod_path
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

    def _weighted_pick(self, candidates):
        if not candidates:
            return None
        total = sum(c[2] for c in candidates)
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
