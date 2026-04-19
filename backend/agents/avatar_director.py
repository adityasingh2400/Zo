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
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

from agents.bridge_clips import pick_bridge_clip

logger = logging.getLogger("empire.director")

# Pitch asset manifest written by scripts/render_pitch_assets.py. Loaded
# lazily on first play_pitch_veo() call so backend boot doesn't depend on
# pitch_assets/ existing yet (CI / first run won't have it).
_PITCH_MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent / "pitch_assets" / "manifest.json"
)

# Listening-attentive substrate for the backchannel. Reuses the existing
# `idle_reading_comments` pose (eyes scanning chat panel, head turned slightly
# toward camera-left, micro-nods) — closer to "I'm listening to you" than
# any other rendered idle. We loop it as a Tier 1 emit on mic press so it
# stays on stage until the actual response (or pitch) crossfades over.
_LISTENING_ATTENTIVE_URL = "/states/idle/idle_reading_comments.mp4"

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
    ("idle_calm",            "/states/idle/idle_calm.mp4",            0.70, "/workspace/idle_speaking/idle_calm_speaking.mp4"),
    ("idle_reading_comments","/states/idle/idle_reading_comments.mp4",0.15, "/workspace/idle_speaking/idle_calm_speaking.mp4"),
    ("idle_thinking",        "/states/idle/idle_thinking.mp4",        0.05, "/workspace/idle_speaking/idle_thinking_speaking.mp4"),
    ("misc_hair_touch",      "/states/idle/misc_hair_touch.mp4",      0.10, "/workspace/idle_speaking/idle_calm_speaking.mp4"),
]

# Tier 1 one-shot interjections. Director picks one occasionally and plays
# it as a crossfaded one-shot over the always-on Tier 0 idle. They start
# and end close to the anchor pose so the crossfade hides any mismatch.
#
# misc_glance_aside lives here (not in TIER0_LIBRARY) because looping a
# glance every 12s reads as "she keeps getting distracted by the same
# thing." One-shot per rotation is the right cadence.
TIER1_INTERJECTIONS: list[tuple[str, str, float, str]] = [
    ("misc_sip_drink",       "/states/idle/misc_sip_drink.mp4",            0.45, ""),
    ("misc_walk_off_return", "/states/idle/misc_walk_off_return.mp4",      0.25, ""),
    ("misc_glance_aside",    "/states/idle/misc_glance_aside_speaking.mp4",0.30, ""),
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
        logger.info("[director] emit %s/%s -> %s (mode=%s fade=%dms muted=%s dur=%s)",
                    layer, intent, url, mode, fade_ms, muted, expected_duration_ms)
        try:
            await self._broadcast(msg)
        except Exception:
            logger.exception("[director] broadcast failed")

    # ── Convenience ───────────────────────────────────────────────────────
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

    async def play_pitch(self, url: str) -> None:
        await self.emit(
            "tier1",
            "pitch",
            url,
            loop=True,
            mode="crossfade",
            emitted_by="play_pitch",
        )

    # ── Pitch-Veo path (USE_PITCH_VEO) ─────────────────────────────────────
    # The opening 30s of the demo. Caller supplies a product slug; we look
    # the slug up in the pitch manifest (rendered offline by
    # scripts/render_pitch_assets.py) and emit BOTH the muted looping video
    # AND the standalone audio + word_timings via a `pitch_audio` event.
    # The dashboard's <audio> element plays the audio while the looping
    # 8s pitching-pose video runs underneath. Karaoke captions divert eye
    # gaze from the mouth so the video loop is invisible.
    @classmethod
    def _load_pitch_manifest(cls) -> dict:
        """Reload the pitch manifest from disk on each lookup so we pick up
        new renders without restarting the backend."""
        if not _PITCH_MANIFEST_PATH.exists():
            return {}
        try:
            return json.loads(_PITCH_MANIFEST_PATH.read_text())
        except Exception as e:
            logger.warning("[director] pitch manifest read failed: %s", e)
            return {}

    async def play_pitch_veo(self, slug: str) -> dict | None:
        """Audio-first pitch playback for `slug`. Emits the muted looping
        video on Tier 1 AND broadcasts a `pitch_audio` event with the
        cached audio_url + word_timings + expected_duration_ms.

        Returns the manifest entry, or None if the slug isn't in the
        manifest (caller falls back to legacy pitch path)."""
        manifest = self._load_pitch_manifest()
        entry = manifest.get(slug)
        if not entry:
            logger.warning("[director] play_pitch_veo: no manifest entry for slug=%s", slug)
            return None

        video_url = entry.get("video_url") or ""
        audio_ms = int(entry.get("audio_ms") or 0)

        # Fire the muted looping video first (no duration handshake — the
        # loop never ends naturally; the audio's `ended` event drives the
        # release timeline).
        await self.emit(
            "tier1",
            "pitch_veo",
            video_url,
            loop=True,
            mode="crossfade",
            emitted_by="play_pitch_veo",
            muted=True,
        )

        # Audio + karaoke chip. Audience hears this immediately and
        # KaraokeCaptions starts populating word-by-word.
        pitch_audio_msg = {
            "type": "pitch_audio",
            "slug": slug,
            "url": entry.get("audio_url"),
            "word_timings": entry.get("word_timings") or [],
            "expected_duration_ms": audio_ms or None,
            "script": entry.get("script", ""),
            "ts": time.time_ns(),
        }
        try:
            await self._broadcast(pitch_audio_msg)
        except Exception:
            logger.exception("[director] pitch_audio broadcast failed")

        # Schedule fade_to_idle when the audio is done (+500ms tail so the
        # last word has a beat before the looping video crossfades out).
        if audio_ms > 0:
            release_ms = audio_ms + 500

            async def _release():
                await asyncio.sleep(release_ms / 1000)
                try:
                    await self._broadcast({
                        "type": "pitch_audio_end",
                        "slug": slug,
                        "ts": time.time_ns(),
                    })
                except Exception:
                    pass
                await self.fade_to_idle()

            asyncio.create_task(_release())

        logger.info("[director] play_pitch_veo slug=%s video=%s audio=%s words=%d dur=%dms",
                    slug, Path(video_url).name, Path(entry.get("audio_url", "")).name,
                    len(entry.get("word_timings") or []), audio_ms)
        return entry

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

    # ── Judge-object opener ───────────────────────────────────────────────
    # Demo script beat 0:30-1:15: judge holds up an object, user says "sell
    # this for $X targeting Y." We want the avatar to react INSTANTLY (before
    # the LLM/TTS/render finishes) so the audience feels the responsiveness.
    #
    # This pre-warms the visual pipeline: snap to the most engaged idle
    # (idle_reading_comments_speaking substrate), play a "I'm thinking"
    # bridge clip, and broadcast the voice_state so the carousel rim light
    # flares. Total cost: one bridge clip render time (already pre-rendered)
    # plus the WS roundtrip. Subjectively feels like 0ms.
    async def play_judge_object_opener(self, label: str | None = None) -> None:
        """Snap to a thinking-attentive idle, play a generic acknowledgement
        bridge clip, and signal voice 'thinking' state. Use this at the
        moment the demo mic is pressed."""
        # 1. Force the substrate to the engaged-attentive variant so the next
        #    Wav2Lip render inherits the right body language.
        engaged_substrate = "/workspace/idle_speaking/idle_reading_comments_speaking.mp4"
        self._current_substrate_pod_path = engaged_substrate

        # 2. Show the "reading" idle on tier 1 so the audience sees the
        #    avatar visibly engage with the held-up object.
        await self.emit(
            "tier1",
            "judge_object_engage",
            READING_CHAT_FALLBACK_URL,
            loop=True,
            mode="crossfade",
            ttl_ms=2200,
            emitted_by="judge_object_opener",
        )

        # 3. Light the carousel rim — the spin reacts in 350ms.
        await self.set_voice_state("thinking")

        # 4. Optional bridge clip — "let me check that out" — fills audio
        #    while the real pitch renders. Picks from the "neutral" or
        #    "question" pool depending on what's available.
        try:
            await self.play_bridge("neutral")
        except Exception:
            logger.debug("[director] judge_object_opener: no bridge available, skipping")

        if label:
            logger.info("[director] judge_object_opener fired for label=%s", label)

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
                # Pitch ended — schedule sip 1.5s after fade-to-idle settles
                # so the avatar reads as "phew, that was a lot, taking a sip".
                self._schedule_sip_after(2000)
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
                # Only sip if Tier 1 is currently idle (Tier 0 painting).
                # If a new response is in flight we don't want to interrupt.
                if self._last_intent.get("tier1") not in (
                    "", "idle_release", "idle_init", "reading_chat",
                    "judge_object_engage", "listening_attentive", None,
                ):
                    # Keep the timestamp but skip the sip; another response
                    # is on stage.
                    logger.debug("[director] motivated sip skipped — tier1 active")
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
