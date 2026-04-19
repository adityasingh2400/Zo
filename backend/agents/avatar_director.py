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
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

from agents.bridge_clips import pick_bridge_clip

logger = logging.getLogger("empire.director")

# Listening-attentive substrate for the backchannel. Reuses the existing
# `idle_reading_comments` pose (eyes scanning chat panel, head turned slightly
# toward camera-left, micro-nods) — closer to "I'm listening to you" than
# any other rendered idle. We loop it as a Tier 1 emit on mic press so it
# stays on stage until the actual response (or pitch) crossfades over.
_LISTENING_ATTENTIVE_URL = "/states/idle/idle_reading_comments.mp4"

# Default pitch video — looped 8s "pitching pose, speaking motion" Veo
# clip already on disk. Used by dispatch_audio_first_pitch (the only
# remaining pitch entry point — driven by the video-upload pipeline,
# not by chat). Karaoke captions divert eye gaze from the 8s loop so
# the 30s audio overlay reads as continuous live speech.
_DEFAULT_PITCH_VIDEO_URL = "/states/state_pitching_pose_speaking_1080p.mp4"

# ── Tunables ────────────────────────────────────────────────────────────────
# Mirrored on the dashboard side (LiveStage.jsx). Keep these in sync.
# Tier 1 transitions (bridge → response, idle → reading_chat, etc.) shortened
# from 350ms → 120ms because Tier 1 clips have very different facial poses
# (looking down, looking aside, speaking) and the long crossfade exposed both
# faces simultaneously — two mouths in different positions overlapping at
# 50% opacity each, which read as crystalline/diamond mouth artifacts.
# 120ms is fast enough that the overlap window is barely perceptible while
# still feeling like a smooth transition rather than a hard cut. Tier 0 idle
# rotation stays at 600ms (idle poses have similar face positions, so the
# longer fade is graceful, not ghosting). Fade-out stays at 500ms — the
# avatar releasing back to idle reads better as a deliberate fade.
TIER1_CROSSFADE_MS_DEFAULT = 120
TIER0_CROSSFADE_MS_DEFAULT = 600
TIER1_FADEOUT_MS_DEFAULT = 500
READING_CHAT_HOLD_MS = 3500  # how long the avatar visibly "reads the comment"
                             # before responding. 3.5s sells the human beat
                             # ("she's actually reading what I typed") and
                             # absorbs LLM + TTS latency so the audio drop
                             # feels like she just finished thinking, not
                             # like she pre-cached the response. Bridge
                             # clips were removed from the comment pipeline
                             # — this hold replaces them as the latency mask.
IDLE_ROTATE_MIN_MS = 8_000   # tighter cadence so demo viewer sees variety quickly
IDLE_ROTATE_MAX_MS = 18_000

# Set of `emitted_by` values that the Director treats as autonomous Tier 1
# emits — they DO NOT claim the Tier 1 layer (no busy_until update) and are
# themselves SUPPRESSED while a deliberate Tier 1 owns the layer. Anything
# not in this set (play_*, dispatch_*, reading_chat, etc.) is considered
# deliberate and claims Tier 1 for its ttl. Single source of truth so we
# never miss-categorise an emitter at a call site.
_IDLE_TIER1_EMITTERS: frozenset[str] = frozenset({
    "idle_init",            # bootstrap (Tier 0; included for symmetry)
    "idle_rotate",          # Tier 0 idle rotation
    "idle_interjection",    # Tier 1 random interjection (sip / glance / walk)
    "motivated_idle.thinking",  # Tier 0 swap to thinking pose
    "motivated_idle.sip",   # post-response motivated sip
})

# Tier 0 = looping idle clips. Each one is symmetric (boomerangable) and
# meant to play indefinitely under the reactive layer. Director rotates
# between them every 12-30s.
#
# Asymmetric clips like sip_drink and walk_off_return CANNOT loop without
# looking unnatural (reversed sip = vomit, reversed walk = moonwalk), so
# they live in TIER1_INTERJECTIONS instead and play as one-shot crossfades
# with the looping idle still painting underneath.
# Each row: (intent, dashboard URL, rotation weight, Wav2Lip substrate pod path).
#
# The substrate is what Wav2Lip uses when a comment fires while this idle
# is the visible Tier 0. We DO NOT use the idle clip itself as the substrate —
# Wav2Lip needs an open, expressively-moving mouth to predict on. Instead
# every idle has a paired "speaking variant" with the same body language /
# framing but with active speech motion that SETTLES to a closed-mouth
# anchor pose in the final 2.5s, so the response crossfades back to silent
# idle seamlessly. The visual continuity comes from matching body, the
# lip-sync quality comes from matching mouth motion.
#
# Speaking variants live in /workspace/idle_speaking/ on the pod, uploaded
# by phase0/scripts/upload_speaking_variants.sh. If a variant doesn't
# exist yet we fall back to the default speaking-pose substrate.
#
# Consolidation: idle_reading_comments and misc_hair_touch both reuse the
# idle_calm_speaking substrate. When a comment lands during reading_comments
# (eyes-down) or hair_touch, the response renders with calm-speaking body
# language — she "looks up to answer," which reads more natural than holding
# the eyes-down or hand-on-hair pose mid-response. Fewer assets to render +
# upload, identical product behaviour.
TIER0_LIBRARY: list[tuple[str, str, float, str]] = [
    # idle_reading_comments deliberately REMOVED from rotation. It's the
    # exclusive reading_chat clip on Tier 1 — having it also rotate on Tier 0
    # made the avatar look like she was reading another comment 5-15s after
    # a real response, even when no comment had arrived. Keep the
    # eyes-down-scanning pose semantically reserved for "I am reading the
    # incoming chat right now" and nothing else.
    ("idle_calm",       "/states/idle/idle_calm.mp4",       0.75, "/workspace/idle_speaking/idle_calm_speaking.mp4"),
    ("idle_thinking",   "/states/idle/idle_thinking.mp4",   0.10, "/workspace/idle_speaking/idle_thinking_speaking.mp4"),
    ("misc_hair_touch", "/states/idle/misc_hair_touch.mp4", 0.15, "/workspace/idle_speaking/idle_calm_speaking.mp4"),
]

# Tier 1 one-shot interjections. Director picks one occasionally and plays
# it as a crossfaded one-shot over the always-on Tier 0 idle. They start
# and end close to the anchor pose so the crossfade hides any mismatch.
#
# misc_glance_aside lives here (not in TIER0_LIBRARY) because looping a
# glance every 12s reads as "she keeps getting distracted by the same
# thing." One-shot per rotation is the right cadence.
TIER1_INTERJECTIONS: list[tuple[str, str, float, str]] = [
    ("misc_sip_drink",       "/states/idle/misc_sip_drink.mp4",          0.45, ""),
    ("misc_walk_off_return", "/states/idle/misc_walk_off_return.mp4",    0.25, ""),
    # Use the EXPLICIT _silent.mp4 variant. The previous URL pointed at
    # the *_speaking.mp4 file (rendered for Wav2Lip overlay) which has
    # visible mouth movement; played muted as an idle interjection it
    # reads as the avatar silently mouthing words — uncanny. The silent
    # render at veo_silent_idle_renders.py was made specifically for
    # this idle-rotation context; mouth stays closed.
    ("misc_glance_aside",    "/states/idle/misc_glance_aside_silent.mp4",0.30, ""),
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
    # whenever the configured speaking variant for the active idle isn't
    # available on the pod (e.g. variant not yet rendered/uploaded).
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
        # Per-substrate availability cache. Populated lazily by probing
        # the Wav2Lip server's /prewarm endpoint (which 400s if the pod
        # path doesn't exist). Kept here so we don't hit the network on
        # every comment.
        self._substrate_available: dict[str, bool] = {
            self.DEFAULT_SUBSTRATE_POD_PATH: True,
        }
        # Motivated idle rotation state (REVISIONS §12). Event-driven
        # triggers fed by observe(); the random idle_loop is the fallback.
        self._voice_state: str | None = None
        self._thinking_task: asyncio.Task | None = None
        self._last_sip_at: float = 0.0
        # Processing-chain bookkeeping for play_processing's two-clip
        # narrative (walk_off_return → processing.mp4). Each call increments
        # the id; the queued processing.mp4 emit checks the id at fire time
        # to detect supersession (back-to-back upload, pitch arriving early,
        # etc.) and skip cleanly. dispatch_audio_first_pitch bumps this so
        # an early pitch cancels the queued bridge tail.
        self._processing_chain_id: int = 0
        # Tier 1 busy horizon (monotonic seconds). Set automatically by
        # emit() whenever a deliberate Tier 1 fires (bridge / pitch /
        # response / reading_chat / listening_attentive / processing /
        # fetching), using ttl_ms (or a 60s default for looped clips).
        # _idle_loop's interjection branch checks this every tick and
        # skips the random misc_* emit while a deliberate clip owns the
        # layer — kills the "she glances aside silently DURING the
        # processing.mp4 readback" overlap class. Cleared by fade_to_idle.
        self._tier1_busy_until: float = 0.0
        # PITCH LOCK — set True for the duration of an opening pitch
        # (audio-first or legacy single-render). Stronger than busy_until:
        # blocks ALL autonomous Tier 1 emits unconditionally, immune to
        # timer drift if the pitch audio runs longer than the renderer's
        # estimate. Cleared explicitly by unlock_tier1_with_settle()
        # which is called by:
        #   1. observe(pitch_audio_end) — dashboard signals audio finished
        #   2. caller's fallback timer based on probed audio duration
        # Once cleared, _post_pitch_settle_until enforces a brief idle
        # hold (default 2.5 s of pure Tier 0) before autonomous Tier 1
        # interjections can fire again — gives the avatar a "settled"
        # beat after the pitch instead of jumping straight to a sip.
        self._tier1_locked: bool = False
        self._post_pitch_settle_until: float = 0.0

    def current_substrate_pod_path(self) -> str:
        """The Wav2Lip server-side path that matches the visible Tier 0 clip,
        falling back to the default speaking-pose substrate if the configured
        speaking variant isn't on the pod yet."""
        path = self._current_substrate_pod_path
        if self._substrate_available.get(path) is False:
            return self.DEFAULT_SUBSTRATE_POD_PATH
        return path

    def mark_substrate_status(self, pod_path: str, available: bool) -> None:
        """Record whether a substrate is on the pod (lazy cache populated
        by external probe code or by Wav2Lip 400 errors at render time)."""
        self._substrate_available[pod_path] = available
        if not available:
            logger.warning("[director] substrate %s unavailable; will fall back to default", pod_path)

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
        except TimeoutError:
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
        muted: bool = False,
        expected_duration_ms: int | None = None,
    ) -> None:
        """Send a single play_clip event. Updates last_intent for replay.

        `muted=True` tells LiveStage to set incomingEl.muted=true and skip
        the volume ramp — used on the audio-first path where the soundtrack
        is coming from a separate <audio> element.

        `expected_duration_ms` enables the dashboard's duration handshake on
        canplaythrough (REVISIONS §4): if the video duration drifts >150ms
        from this number, the dashboard rejects the video and lets audio
        play alone.
        """
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
            "muted": muted,
            "expected_duration_ms": expected_duration_ms,
            "ts": time.time_ns(),
            "emitted_by": emitted_by,
        }
        self._last_intent[layer] = intent
        self._last_url[layer] = url
        # Tier 1 busy-tracking + PITCH LOCK gate.
        #
        # Hard gate (pitch lock): autonomous Tier 1 emits are SUPPRESSED
        # entirely while a pitch is on stage — the lock is set explicitly
        # by lock_tier1_for_pitch() and cleared by unlock_tier1_with_settle()
        # which is called by the dashboard's pitch_audio_end observer.
        # Same suppression also covers the post-pitch settle window so
        # autonomous Tier 1 doesn't fire the instant the lock releases.
        # Deliberate Tier 1 emits (response, dispatch_audio_first_pitch,
        # fade_to_idle, etc.) are NEVER suppressed — they're the caller's
        # responsibility to sequence correctly.
        #
        # Soft gate (busy_until): every deliberate Tier 1 emit extends a
        # rolling horizon based on ttl_ms (or 60 s for looped clips with
        # no natural end). _idle_loop checks this for the common case
        # where no pitch is in flight but a bridge / response is on stage.
        if layer == "tier1":
            if emitted_by in _IDLE_TIER1_EMITTERS:
                # Autonomous Tier 1 — suppress while the pitch lock owns
                # the layer or we're in the post-pitch settle window.
                if self._tier1_locked or time.monotonic() < self._post_pitch_settle_until:
                    logger.info("[director] suppressing autonomous tier1 (%s) — "
                                "locked=%s settle_until=%.2f",
                                emitted_by, self._tier1_locked,
                                self._post_pitch_settle_until)
                    return
                # Otherwise fall through to the broadcast — autonomous
                # emit doesn't claim the layer.
            elif emitted_by == "fade_to_idle":
                # The release sentinel — explicitly clear the horizon so
                # idle rotation can resume at the very next tick (subject
                # to settle if it was set by the pitch path).
                self._tier1_busy_until = 0.0
            else:
                if loop:
                    busy_for_s = 60.0  # cleared by fade_to_idle
                else:
                    busy_for_s = (ttl_ms or expected_duration_ms or 8_000) / 1000
                self._tier1_busy_until = time.monotonic() + busy_for_s
        logger.info("[director] emit %s/%s -> %s (mode=%s fade=%dms muted=%s dur=%s)",
                    layer, intent, url, mode, fade_ms, muted, expected_duration_ms)
        try:
            await self._broadcast(msg)
        except Exception:
            logger.exception("[director] broadcast failed")

    # ── Convenience ───────────────────────────────────────────────────────
    async def emit_reading_chat(self) -> None:
        """Emit the reading_chat Tier 1 clip with NO internal hold. Caller
        owns the timing — used by run_routed_comment to fire reading visuals
        within ~50ms of comment arrival, then keep them visible for the
        natural duration of classify+LLM+TTS+Wav2Lip rendering. The clip
        loops, so it stays visible until the caller fades it out via
        fade_to_idle().

        Use this when you want responsive visual feedback without a fixed
        timer. Use reading_chat() (below) when you want the legacy fixed
        hold for backwards compatibility (e.g. direct REST hits)."""
        await self.emit(
            "tier1",
            "reading_chat",
            READING_CHAT_FALLBACK_URL,
            loop=True,
            mode="crossfade",
            ttl_ms=None,
            emitted_by="emit_reading_chat",
        )

    async def reading_chat(self) -> None:
        """Show the avatar reading the incoming comment. Held briefly before
        the bridge crossfades over it. Uses the polished
        idle_reading_comments clip as the visible reading-chat moment."""
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

    async def play_processing(self) -> None:
        """Tier 1 ambient cover for the upload→pitch processing window.

        Two-clip narrative chain:
          1. walk_off_return (8.0 s) — emitted as intent="fetching" so the
             HUD reads "she went to grab the item the operator just sent."
             The audience reads the off-screen beat as "AI is fetching the
             product the operator just dropped." She walks back into frame
             at the tail.
          2. processing.mp4 (14.13 s) — chained ~7.7 s after step 1 (300 ms
             tail overlap so the dashboard crossfade hides the cut). She's
             back in frame and now picks up a printed spec sheet, reads it,
             sets it down. Maps to "AI is now reviewing what was just
             handed to it."

        Total bridge ≈ 22 s. Pipeline target is 10-15 s, so the pitch
        usually crossfades over processing.mp4 mid-readback — feels like
        "she finished reading and started speaking." If the pipeline
        outruns the bridge entirely (cold Wav2Lip + large video), the
        clip ends naturally and Tier 0 idle resumes until the pitch
        crossfades in.

        Race handling — the queued processing.mp4 emit is gated on
        `_processing_chain_id`. Bumped by:
          - back-to-back call to play_processing (second upload before
            the first finishes) — the older queued tail no-ops out.
          - dispatch_audio_first_pitch — an early pitch cancels the
            queued processing tail so the pitch isn't overlaid mid-read.

        End frame of processing.mp4 is the canonical anchor pose (hands
        at waist, soft smile, eye contact) so the pitch crossfade lands
        clean — same target pose as the welcome clip + the Wav2Lip
        substrates. No special handoff logic needed.
        """
        # No debounce needed — the route handler is the sole call site
        # post-Option-A, so this fires exactly once per upload. Back-to-
        # back uploads bump the chain_id and the queued processing.mp4
        # tail of the older chain no-ops out at the supersession check.
        self._processing_chain_id += 1
        chain_id = self._processing_chain_id

        # Step 1: walk-off-and-return. Intent label "fetching" surfaces in
        # the dashboard HUD so it's obvious which narrative beat is on
        # screen. ttl_ms = 8000 (probed) so the player knows when to
        # expect the natural end if no follow-up arrives.
        await self.emit(
            "tier1",
            "fetching",
            "/states/idle/misc_walk_off_return.mp4",
            loop=False,
            mode="crossfade",
            ttl_ms=8_000,
            emitted_by="play_processing_fetch",
        )

        # Step 2: schedule processing.mp4 as the second link. 7.7 s wait
        # = walk_off duration (8.0) - 300 ms overlap. The 300 ms tail
        # gives the Tier 1 crossfade enough room to hide the seam between
        # her stepping back into frame and her picking up the paper.
        async def _chain_processing(my_id: int) -> None:
            try:
                await asyncio.sleep(7.7)
            except asyncio.CancelledError:
                return
            if self._processing_chain_id != my_id:
                # Superseded by another play_processing call or by a pitch
                # dispatch. Don't overlay stale content.
                logger.info("[director] processing chain superseded (id %d → %d), skip",
                            my_id, self._processing_chain_id)
                return
            await self.emit(
                "tier1",
                "processing",
                "/bridges/processing/processing.mp4",
                loop=False,
                mode="crossfade",
                ttl_ms=14_130,
                emitted_by="play_processing",
            )

        asyncio.create_task(_chain_processing(chain_id))

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

    async def play_response(
        self,
        url: str,
        *,
        muted: bool = False,
        expected_duration_ms: int | None = None,
    ) -> None:
        """Tier 1 response crossfade. `muted` + `expected_duration_ms` are
        used by the audio-first path (USE_AUDIO_FIRST) where the audio is
        already playing through a standalone <audio> element on the
        dashboard and the video is mounted muted underneath."""
        await self.emit(
            "tier1",
            "response",
            url,
            loop=False,
            mode="crossfade",
            emitted_by="play_response",
            muted=muted,
            expected_duration_ms=expected_duration_ms,
        )

    # ── Pitch dispatch (audio-first, ephemeral) ────────────────────────────
    # The opening 30s of the demo. Driven exclusively by the video-upload
    # pipeline (run_sell_pipeline → _run_audio_first_pitch in main.py).
    # Caller has already TTS'd the freshly-Claude-generated pitch script,
    # saved the audio to a static URL, and produced per-word timings — we
    # just emit the muted looping speaking-pose video on Tier 1 + broadcast
    # the pitch_audio WS event the dashboard plays through its standalone
    # <audio> element with karaoke captions on top.
    #
    # Note: there's intentionally no slug→manifest lookup path. Pre-rendered
    # cached pitches existed for the chat-trigger flow ("sell this for $X");
    # that flow was removed because production never gets a typed pitch
    # command. Every pitch is dynamically generated from the recorded
    # video transcript.
    async def dispatch_audio_first_pitch(
        self,
        *,
        audio_url: str,
        word_timings: list[dict],
        audio_ms: int,
        script: str = "",
        video_url: str | None = None,
        slug: str | None = None,
    ) -> None:
        """Pitch dispatch driven by the video-upload pipeline. Caller has
        already saved the audio bytes to a static-served URL and produced
        word timings.

        `video_url` defaults to the universal speaking-pose clip; pass a
        product-specific Veo render here if one exists.
        `slug` is informational (logged + included in the pitch_audio
        event) — useful for telemetry / dashboard chip variants.
        """
        url = video_url or _DEFAULT_PITCH_VIDEO_URL

        # Cancel any pending play_processing tail. Bumping the chain id
        # makes the queued processing.mp4 emit (still asyncio.sleep'ing in
        # _chain_processing) no-op when it wakes — prevents the bridge
        # from overlaying the pitch mid-speech if the pipeline finishes
        # before the walk_off → processing handoff has happened.
        self._processing_chain_id += 1
        # Also lock Tier 1 for the duration of the pitch — same hard gate
        # used by the legacy pitch path. unlock_tier1_with_settle() fires
        # from observe(pitch_audio_end) when the dashboard's <audio>
        # signals the audio finished.
        self.lock_tier1_for_pitch()

        # Tier 1 muted looping pose. Dashboard mutes the video element
        # (audio is owned by the standalone <audio>) and skips the
        # duration handshake because loop=True (no natural end).
        await self.emit(
            "tier1",
            "pitch_veo",
            url,
            loop=True,
            mode="crossfade",
            emitted_by="dispatch_audio_first_pitch",
            muted=True,
        )

        # pitch_audio WS event — the dashboard's <audio> element plays this
        # immediately and KaraokeCaptions tracks word-by-word. Same shape
        # as the cached path so the dashboard handler is one code path.
        pitch_audio_msg = {
            "type": "pitch_audio",
            "slug": slug or "ephemeral",
            "url": audio_url,
            "word_timings": word_timings or [],
            "expected_duration_ms": audio_ms or None,
            "script": script,
            "ts": time.time_ns(),
        }
        try:
            await self._broadcast(pitch_audio_msg)
        except Exception:
            logger.exception("[director] dispatch_audio_first_pitch broadcast failed")

        # Schedule the fade-to-idle when audio ends (+500ms tail). We use
        # the audio duration as the canonical timing source — the looping
        # video has no inherent end.
        if audio_ms and audio_ms > 0:
            release_ms = audio_ms + 500

            async def _release():
                await asyncio.sleep(release_ms / 1000)
                try:
                    await self._broadcast({
                        "type": "pitch_audio_end",
                        "slug": slug or "ephemeral",
                        "ts": time.time_ns(),
                    })
                except Exception:
                    pass
                await self.fade_to_idle()
                await self.set_voice_state(None)

            asyncio.create_task(_release())

        logger.info("[director] dispatch_audio_first_pitch slug=%s "
                    "video=%s audio=%s words=%d dur=%dms",
                    slug or "ephemeral", Path(url).name,
                    Path(audio_url).name, len(word_timings or []), audio_ms)

    # ── Listening backchannel (USE_BACKCHANNEL) ─────────────────────────────
    # Mic press → instantly swap Tier 1 to a "listening attentive" loop.
    # Reads as the avatar visibly registering the input ~50ms after the
    # operator starts speaking. Visual only per REVISIONS §8 — no "mhm"
    # audio (overlap risk during user speech isn't worth the win).
    async def play_listening_attentive(self) -> None:
        """Snap to a listening-attentive idle on Tier 1. Looped so it
        stays on stage until the actual response or pitch crossfades over."""
        await self.emit(
            "tier1",
            "listening_attentive",
            _LISTENING_ATTENTIVE_URL,
            loop=True,
            mode="crossfade",
            ttl_ms=10_000,
            emitted_by="play_listening_attentive",
            muted=True,
        )
        await self.set_voice_state("transcribing")

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

    # ── Pitch lock ────────────────────────────────────────────────────────
    # The opening pitch is sacred — it's the first thing the audience hears,
    # it sets the entire demo's tone, and it MUST NOT be interrupted by an
    # autonomous idle gesture firing mid-sentence. The busy_until timer
    # (set by emit() based on ttl_ms) is the soft layer; this lock is the
    # hard layer and exists because the legacy pitch path passes
    # expected_duration_ms=None which silently defaults busy_until to 8 s
    # while the pitch is actually 20+ s long. With the lock, no autonomous
    # Tier 1 emit fires regardless of timer state. Cleared explicitly by
    # the dashboard's pitch_audio_end event (truth source for "audio
    # actually stopped") with a fallback timer in case the event drops.

    def lock_tier1_for_pitch(self) -> None:
        """Mark Tier 1 as pitch-locked. While locked, _idle_loop and
        emit() refuse to fire autonomous Tier 1 (idle interjection /
        motivated sip). Deliberate emits (response, fade_to_idle,
        dispatch_audio_first_pitch's own follow-ups) still go through.
        Idempotent — safe to call multiple times."""
        if not self._tier1_locked:
            logger.info("[director] tier1 PITCH-LOCKED — autonomous Tier 1 suppressed")
        self._tier1_locked = True

    def unlock_tier1_with_settle(self, settle_seconds: float = 2.5) -> None:
        """Release the pitch lock and start a brief settle window during
        which autonomous Tier 1 still won't fire — gives the avatar a
        deliberate "phew, that was the pitch" beat in pure Tier 0 idle
        before sips / glances / walk-offs resume.

        Caller is responsible for issuing fade_to_idle separately when
        appropriate (this method only manages lock state, not the visual
        Tier 1 release)."""
        if self._tier1_locked:
            logger.info("[director] tier1 unlocked, settle for %.1f s before autonomous resume",
                        settle_seconds)
        self._tier1_locked = False
        self._post_pitch_settle_until = time.monotonic() + max(0.0, settle_seconds)
        # Also clear the soft busy horizon — it may still be in the future
        # from the pitch's own emit (60 s for looped) and would otherwise
        # extend the suppression window past the settle period.
        self._tier1_busy_until = 0.0

    # ── Voice flow integration ────────────────────────────────────────────
    # The dashboard's <Spin3D> reacts to a `state` prop ('idle' | 'listening'
    # | 'thinking' | 'responding') with rim-light gain, rotation speed, and
    # accent flashes. The avatar's <LiveStage> shows a pill for the same
    # state. Both are driven by `voice_state` events broadcast here so the
    # whole stage moves in lockstep with the voice pipeline.
    #
    # Cody's POST /api/voice_comment can call these at the right moments:
    #   set_voice_state("transcribing")   when audio upload begins
    #   set_voice_state("thinking")       when transcript lands, router about to fire
    #   set_voice_state("responding")     when render kicks off
    #   set_voice_state(None)             when response_video is on stage (LIVE pill takes over)
    #
    # Calling these is OPTIONAL — the dashboard already infers the state
    # from voice_transcript / routing_decision / comment_response_video
    # events. These exist for tighter UI sync when the backend wants to
    # step ahead of the next pipeline stage.
    async def set_voice_state(self, state: str | None) -> None:
        """Broadcast a voice_state event. `state` is 'transcribing' |
        'thinking' | 'responding' | None (clear)."""
        msg = {
            "type": "voice_state",
            "state": state,
            "ts": time.time_ns(),
            "emitted_by": "director.set_voice_state",
        }
        try:
            await self._broadcast(msg)
        except Exception:
            logger.exception("[director] voice_state broadcast failed")

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
                #
                # SUPPRESS during any of:
                #   1. _tier1_locked (pitch in progress) — hard gate
                #   2. _post_pitch_settle_until (idle hold after pitch) — hard gate
                #   3. _tier1_busy_until in the future (deliberate Tier 1
                #      currently on stage, e.g. bridge / response /
                #      reading_chat / fetching / processing) — soft gate
                # When all clear, fall through to weighted-pick + emit.
                now = time.monotonic()
                if (random.random() < INTERJECTION_PROBABILITY
                        and TIER1_INTERJECTIONS
                        and not self._tier1_locked
                        and now >= self._post_pitch_settle_until
                        and now >= self._tier1_busy_until):
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

    # ── Motivated idle observer (REVISIONS §12) ────────────────────────────
    # The doc downscoped to two triggers:
    #   1. comment_complete (>3s audio) → schedule a sip_drink interjection
    #      AFTER the response finishes (debounced; once per 30s max).
    #   2. voice_state="thinking" >2s → swap Tier 0 to idle_thinking pose.
    # Event-listener pattern: broadcast_to_dashboards() calls observe(msg)
    # so we don't sprinkle notify() calls through main.py.
    _SIP_INTERJECTION_URL = "/states/idle/misc_sip_drink.mp4"
    _IDLE_THINKING_URL = "/states/idle/idle_thinking.mp4"
    _IDLE_THINKING_SUBSTRATE = "/workspace/idle_speaking/idle_thinking_speaking.mp4"
    _SIP_DEBOUNCE_SEC = 30.0
    _SIP_AUDIO_FLOOR_MS = 3000   # only fire on responses ≥3s
    _THINKING_DELAY_SEC = 2.0    # wait this long before flipping to idle_thinking

    async def observe(self, msg: dict) -> None:
        """Inspect outgoing broadcasts and drive event-triggered idle
        behaviour. Hooked into broadcast_to_dashboards() so every dashboard
        message also informs the Director's choreography state machine."""
        try:
            mtype = msg.get("type")
            if mtype == "voice_state":
                await self._handle_voice_state(msg.get("state"))
            elif mtype == "comment_response_audio":
                # Schedule a sip-drink interjection after the response audio
                # finishes IF it's long enough to "earn" the gesture (a 1s
                # one-liner doesn't motivate sipping; a 4-5s answer does).
                dur_ms = int(msg.get("expected_duration_ms") or 0)
                if dur_ms >= self._SIP_AUDIO_FLOOR_MS:
                    self._schedule_sip_after(dur_ms + 600)
            elif mtype == "comment_response_video" and not msg.get("audio_already_playing"):
                # Legacy serial path — schedule sip after video duration since
                # there's no audio_already_playing flag carrying duration info.
                # We use a coarse estimate from total_ms (TTS+render+lipsync).
                # Conservative floor since total_ms includes render time.
                resp_text = msg.get("response", "")
                est_audio_ms = int(max(2500, len(resp_text.split()) * 350))
                if est_audio_ms >= self._SIP_AUDIO_FLOOR_MS:
                    self._schedule_sip_after(est_audio_ms + 600)
            elif mtype == "pitch_audio_end":
                # Pitch finished. Truth source from the dashboard's <audio>
                # ended event — release the pitch lock and start the
                # post-pitch settle window so autonomous Tier 1 stays
                # suppressed for an additional 2.5 s of pure idle. After
                # the settle, _idle_loop resumes random interjections at
                # the natural cadence. The motivated sip schedule below
                # picks up after the settle window ends (the sip itself
                # checks _post_pitch_settle_until before firing).
                self.unlock_tier1_with_settle(settle_seconds=2.5)
                self._schedule_sip_after(2500 + 1500)  # settle + 1.5 s
        except Exception:
            logger.exception("[director] observe failed (non-fatal)")

    async def _handle_voice_state(self, state: str | None) -> None:
        """Cancel any pending thinking timer; start a fresh one if entering
        thinking. If still thinking after _THINKING_DELAY_SEC, swap Tier 0."""
        prev = self._voice_state
        self._voice_state = state
        if state == "thinking" and prev != "thinking":
            if self._thinking_task and not self._thinking_task.done():
                self._thinking_task.cancel()
            self._thinking_task = asyncio.create_task(self._fire_thinking_after_delay())
        elif state != "thinking" and self._thinking_task:
            # Left the thinking state before the timer expired — cancel it
            # so we don't end up swapping idle clips unnecessarily.
            self._thinking_task.cancel()
            self._thinking_task = None

    async def _fire_thinking_after_delay(self) -> None:
        try:
            await asyncio.sleep(self._THINKING_DELAY_SEC)
        except asyncio.CancelledError:
            return
        if self._voice_state != "thinking":
            return
        # Only swap if Tier 0 isn't already on the thinking pose; we don't
        # want to retrigger the crossfade for no visual change.
        if self._last_intent.get("tier0") == "idle_thinking":
            return
        logger.info("[director] motivated idle: voice_state=thinking >2s → idle_thinking")
        self._current_substrate_pod_path = self._IDLE_THINKING_SUBSTRATE
        try:
            await self.emit(
                "tier0",
                "idle_thinking",
                self._IDLE_THINKING_URL,
                loop=True,
                mode="crossfade",
                emitted_by="motivated_idle.thinking",
            )
        except Exception:
            logger.exception("[director] motivated thinking emit failed")

    def _schedule_sip_after(self, delay_ms: int) -> None:
        """Schedule a sip-drink Tier 1 interjection delay_ms from now,
        debounced so we don't queue multiple sips on rapid-fire responses."""
        now = time.time()
        if now - self._last_sip_at < self._SIP_DEBOUNCE_SEC:
            return
        self._last_sip_at = now  # claim the slot pre-fire to prevent double-schedule

        async def _fire():
            try:
                await asyncio.sleep(delay_ms / 1000)
                # Three suppression gates (same set as _idle_loop's
                # interjection branch — single mechanism):
                #   1. _tier1_locked — pitch in progress (hard gate)
                #   2. _post_pitch_settle_until — post-pitch idle hold (hard gate)
                #   3. _tier1_busy_until — deliberate Tier 1 on stage (soft gate)
                # Skipping is silent + non-rescheduling: the next eligible
                # comment_response_audio event will queue a fresh sip.
                now_m = time.monotonic()
                if (self._tier1_locked
                        or now_m < self._post_pitch_settle_until
                        or now_m < self._tier1_busy_until):
                    logger.debug("[director] motivated sip skipped — locked=%s settle=%.1f busy=%.1f",
                                 self._tier1_locked,
                                 max(0, self._post_pitch_settle_until - now_m),
                                 max(0, self._tier1_busy_until - now_m))
                    return
                logger.info("[director] motivated idle: sip_drink after %dms", delay_ms)
                await self.emit(
                    "tier1",
                    "misc_sip_drink",
                    self._SIP_INTERJECTION_URL,
                    loop=False,
                    mode="crossfade",
                    emitted_by="motivated_idle.sip",
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[director] motivated sip fire failed")

        asyncio.create_task(_fire())

    # ── Replay state for newly connected dashboards ───────────────────────
    def replay_state(self) -> dict[str, dict[str, str]]:
        """Snapshot of the latest intent on each layer; included in state_sync
        so a freshly opened dashboard can rehydrate the stage."""
        return {
            layer: {"intent": self._last_intent.get(layer, ""),
                    "url": self._last_url.get(layer, "")}
            for layer in ("tier0", "tier1")
        }
