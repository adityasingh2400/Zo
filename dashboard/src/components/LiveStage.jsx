import React, { useEffect, useRef, useState } from 'react';
import { useAvatarStream, TIER1_FADEOUT_MS } from '../hooks/useAvatarStream';
import { dlog } from '../lib/dlog';
import { KaraokeCaptions } from './KaraokeCaptions';
import { TranslationChip } from './TranslationChip';

// Blur pulse — symmetric 0 → peak → 0 of CSS filter:blur() on the
// .stage-videos wrapper during every crossfade. Destroys high-frequency
// facial edges (mouth, eyes, jawline) so two faces overlapping at
// intermediate opacity can't be picked apart by the eye. Lets us keep
// crossfades visually smooth without re-introducing the pose-mismatch
// ghosting that the 350→120ms shrink fixed by brute force. Mirror of
// the same trick in /dev/transitions — tune visually there with the
// peak/max sliders before bumping these. Set BLUR_PEAK_PX = 0 to
// disable and A/B against pure-opacity behaviour.
const BLUR_PEAK_PX = 8;
const BLUR_MAX_MS = 500;

const API_BASE = `http://${window.location.hostname}:8000`;

/**
 * Cinema stage with seamless avatar continuity.
 *
 * Three stacked <video> elements:
 *   - tier0      (z=0)  always-on idle layer, muted, looping
 *   - tier1A/B   (z=1)  reactive layer, two elements ping-pong with opacity crossfade
 *
 * Source-of-truth for what should be on each tier comes from useAvatarStream,
 * which subscribes to the backend Director's `play_clip` events.
 *
 * The legacy props (responseVideo, pitchVideoUrl) are still accepted so the
 * existing parent App can render the floating comment overlay + latency badge.
 * Source selection itself is no longer driven by them — that's the Director's
 * job now.
 */
const API_BASE_FOR_AUDIO = `http://${typeof window !== 'undefined' ? window.location.hostname : 'localhost'}:8000`;

export function LiveStage({
  productData,
  pitchVideoUrl,           // kept for placeholder context only
  responseVideo,           // kept to drive the floating overlay UI
  pendingComments = [],
  liveStage,
  wsRef,
  audioResponse,           // audio-first dispatch: {url, word_timings, expected_duration_ms, ...}
  pitchAudio,              // 30s Veo pitch audio: same shape, separate slot
  onAudioEnded,            // (kind: 'response' | 'pitch') -> void; parent clears state
  inOverlay = false,       // /stage: TikTokShopOverlay wraps us — suppress chrome that
                            // duplicates the overlay's chat rail / LIVE badge / BUY card.
                            // Default off keeps the operator dashboard at / pixel-identical.
  connected,               // re-trigger for useVoiceStage's WS listener attach. Optional
                            // (defaults undefined) so callers that don't pass it still
                            // mount, just without the voice pill / routing badge — same
                            // as before this prop existed. Pass it from useEmpireSocket
                            // to enable the in-stage voice/routing overlays.
  minimalChrome = false,   // when true (driven by MINIMAL_STAGE in StageView), also
                            // hide voice-state pill + routing badge + TranslationChip
                            // so the stage is a clean canvas for transition + chat work.
                            // Karaoke captions stay because they ARE the agent's reaction.
}) {
  // Voice/routing state listened off the shared WS directly so LiveStage stays
  // self-contained — no prop changes needed in App.jsx (which Cody is also
  // editing to add VoiceMic). When Cody's server starts broadcasting these
  // events, the pills + badge come alive automatically.
  const { voiceState, routingDecision } = useVoiceStage({ wsRef, connected });
  // Tier 0 = ping-pong of two looping idle videos with opacity crossfade,
  // matching the Tier 1 design. Eliminates the visible flash that used to
  // happen when a single <video> swapped src between rotating idle clips.
  const tier0ARef = useRef(null);
  const tier0BRef = useRef(null);
  const tier1ARef = useRef(null);
  const tier1BRef = useRef(null);

  // Tracks the in-flight Tier 1 volume-fade rAF id so a rapidly-arriving new
  // Tier 1 clip can cancel the previous fade before starting its own. Without
  // this, two rAF loops were running simultaneously over the same A/B
  // elements — fighting over each other's volume targets — which produced
  // dual-audio bleed (bridge audio + response audio playing together) plus
  // the IndexSizeError volume crashes from out-of-range t values.
  const tier1FadeRafRef = useRef(null);

  // Wrapper around just the four <video> elements (carrier for the
  // blur pulse — see blurStage). Overlays (LIVE pill, voice/routing
  // pills, comment card, karaoke captions, translation chip) live as
  // siblings of this wrapper inside styles.stage so the blur applies
  // ONLY to the videos and the chrome stays sharp.
  const stageVideosRef = useRef(null);
  // rAF id for the in-flight blur pulse. Cancelled at the start of
  // each new transition so rapid back-to-back crossfades (reading_chat
  // → bridge → response in the cloud-escalate path, all within ~200ms)
  // don't stack two blur ramps fighting over the same filter property.
  const blurRafRef = useRef(null);

  // Speculative preload (Pass 2). When the router fires a routing_decision
  // we often know the URL of the upcoming Tier 1 clip 100-300ms BEFORE the
  // play_clip event lands. Stashing it here makes the Tier 1 driver use the
  // already-decoded incoming element directly, skipping the canplay round
  // trip on the actual play. Cleared after the play_clip consumes it.
  const tier1PreloadRef = useRef({ url: null, slot: null });

  // Hidden <audio> element owned by LiveStage. Drives both audio-first
  // comment responses AND 30s pitch playback. KaraokeCaptions (Phase 4)
  // reads currentTime off this element via the ref. Single instance reused
  // across plays — setting src + calling load() is cheaper than churning
  // through new Audio() objects.
  const audioRef = useRef(null);
  const [audioPlaying, setAudioPlaying] = useState(null); // {kind, url, word_timings, ts}

  // Which element of each tier is currently visible (true = A, false = B).
  const [tier0ActiveIsA, setTier0ActiveIsA] = useState(true);
  const [tier0Opacity, setTier0Opacity] = useState(1);   // tier0 starts visible
  const [tier1ActiveIsA, setTier1ActiveIsA] = useState(true);
  const [tier1Opacity, setTier1Opacity] = useState(0);
  const [overlayVisible, setOverlayVisible] = useState(false);

  const stream = useAvatarStream({ wsRef, connected });

  // ── Blur pulse ──────────────────────────────────────────────────────────
  // Triggered at the start of every crossfade. Duration scales with the
  // crossfade itself: Tier 1 in (120ms) → 120ms pulse, Tier 0 rotation
  // (600ms) → clipped to BLUR_MAX_MS so the blur ends well before the
  // opacity ramp does. 60ms floor keeps the 120ms case from flickering
  // on/off in two frames. sin(t·π) gives a smooth bell with peak at
  // midpoint — no perceptible "stays blurred" tail, no hard turn-off.
  // Identical algorithm to the dev simulator at /dev/transitions; keep
  // them in sync when tuning.
  function blurStage(crossfadeMs) {
    if (BLUR_PEAK_PX <= 0) return;
    const stage = stageVideosRef.current;
    if (!stage) return;
    if (blurRafRef.current) {
      cancelAnimationFrame(blurRafRef.current);
      blurRafRef.current = null;
    }
    const dur = Math.max(60, Math.min(crossfadeMs, BLUR_MAX_MS));
    const start = performance.now();
    function tick(now) {
      const t = Math.max(0, Math.min(1, (now - start) / dur));
      const px = BLUR_PEAK_PX * Math.sin(t * Math.PI);
      try {
        stage.style.filter = px > 0.05 ? `blur(${px.toFixed(2)}px)` : 'none';
      } catch {
        blurRafRef.current = null;
        return;
      }
      if (t < 1) {
        blurRafRef.current = requestAnimationFrame(tick);
      } else {
        stage.style.filter = 'none';
        blurRafRef.current = null;
      }
    }
    blurRafRef.current = requestAnimationFrame(tick);
  }

  // ── Tier 0 driver ────────────────────────────────────────────────────────
  // Always-on idle layer. Two stacked elements ping-pong with a 600ms opacity
  // crossfade so swapping idle clips never shows a visible flash.
  const lastTier0Url = useRef(null);
  useEffect(() => {
    const url = stream.tier0?.url
      || `/states/state_idle_pose_silent_1080p.mp4`;
    if (url === lastTier0Url.current) return;

    const incomingEl = tier0ActiveIsA ? tier0BRef.current : tier0ARef.current;
    const outgoingEl = tier0ActiveIsA ? tier0ARef.current : tier0BRef.current;
    if (!incomingEl) return;

    const fadeMs = stream.tier0?.fadeMs ?? 600;
    incomingEl.muted = true;
    incomingEl.loop = stream.tier0?.loop ?? true;
    incomingEl.volume = 0;

    // First-frame match (REVISIONS §17). Loads the new src, seeks to t=0,
    // waits for the seeked event so the decoder has the EXACT first frame
    // ready before we start the opacity ramp. Without this, the incoming
    // element was sometimes painting frame 2-5 (whatever decoded first
    // after canplay) when the fade kicked in — a small but visible "her
    // head jumped" pop on slo-mo replay.
    prepareFirstFrame(incomingEl, `${API_BASE}${url}`)
      .then(() => incomingEl.play())
      .then(() => {
        blurStage(fadeMs);
        setTier0ActiveIsA(prev => !prev);
        setTier0Opacity(1);
        stream.sendStageReady();
        setTimeout(() => {
          try { outgoingEl?.pause(); } catch {}
        }, fadeMs + 50);
      })
      .catch((err) => {
        console.warn('[LiveStage] tier0 prep failed', err);
      });

    lastTier0Url.current = url;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stream.tier0?.url, stream.tier0?.loop]);

  // ── Tier 1 driver ────────────────────────────────────────────────────────
  // On every new tier1 instruction, load it into the inactive element, wait
  // for canplay, then crossfade. Special case: empty url = fade out (idle release).
  useEffect(() => {
    if (!stream.tier1) return;
    const url = stream.tier1.url;
    const fadeMs = stream.tier1.fadeMs;

    // Cancel any in-flight volume-fade rAF from a previous Tier 1 transition.
    // Without this, rapid back-to-back Tier 1 clips (reading_chat → bridge →
    // response, all within ~200ms in the cloud-escalate path) leave multiple
    // rAF loops running concurrently, each setting volumes on the same A/B
    // pair. The result was bridge audio bleeding into response audio (the
    // "two audios at once" bug) and occasional IndexSizeError crashes when
    // the loops disagreed about which element should be at volume 0.
    if (tier1FadeRafRef.current) {
      cancelAnimationFrame(tier1FadeRafRef.current);
      tier1FadeRafRef.current = null;
      dlog('tier1', 'prev_raf_cancelled', { intent: stream.tier1?.intent });
    }
    dlog('tier1', 'event_received', {
      intent: stream.tier1?.intent || '(none)',
      url: stream.tier1?.url || '(empty=fade-out)',
      fadeMs: stream.tier1?.fadeMs,
      muted: !!stream.tier1?.muted,
    });

    // Idle release — fade Tier 1 out, reveal Tier 0 underneath.
    if (!url) {
      const activeEl = tier1ActiveIsA ? tier1ARef.current : tier1BRef.current;

      // Audio ramp alongside the CSS opacity fade. Previously we hard-muted
      // the element immediately on idle release — audio dropped to silence
      // while the video kept visually fading, which read as "the audio cut
      // out right before her face vanished". Now we ramp volume to 0 over
      // the same fadeMs window so audio + visual fade in lockstep, no
      // perceptual lag at the seam. Skip the ramp on muted elements
      // (audio-first paths) since their soundtrack is owned by the
      // standalone <audio> element and is fading via its own onEnded.
      const startVol = (() => {
        try { return activeEl ? activeEl.volume : 0; } catch { return 0; }
      })();
      if (activeEl && !activeEl.muted && startVol > 0) {
        const start = performance.now();
        function fadeOutTick(now) {
          const t = Math.max(0, Math.min(1, (now - start) / fadeMs));
          try { activeEl.volume = startVol * (1 - t); } catch {
            tier1FadeRafRef.current = null;
            return;
          }
          if (t < 1) {
            tier1FadeRafRef.current = requestAnimationFrame(fadeOutTick);
          } else {
            tier1FadeRafRef.current = null;
          }
        }
        tier1FadeRafRef.current = requestAnimationFrame(fadeOutTick);
      } else {
        // Already silent or muted — just clamp to 0 to be safe.
        try { if (activeEl) activeEl.volume = 0; } catch {}
      }

      blurStage(fadeMs);
      setTier1Opacity(0);
      setOverlayVisible(false);
      // After the fade settles, pause the active element so the decoder is free.
      setTimeout(() => {
        try { activeEl?.pause(); } catch {}
        stream.sendAck(stream.tier1.intent, '', 'ended');
      }, fadeMs + 50);
      return;
    }

    // Crossfade to a new clip. Load into the inactive element first.
    const activeIsA = tier1ActiveIsA;
    const incomingEl = activeIsA ? tier1BRef.current : tier1ARef.current;
    const outgoingEl = activeIsA ? tier1ARef.current : tier1BRef.current;
    if (!incomingEl) return;

    // Force-mute the outgoing element NOW (before the new clip even loads)
    // so the previous bridge/response can't keep blasting audio for the
    // canplay → play() round trip (~150-300ms cold) while the new clip is
    // preparing. The video continues playing visually under the new opacity
    // crossfade — only the audio is silenced. This is the second half of the
    // dual-audio fix: even if the previous fade's rAF was mid-flight when
    // the new event arrived, this clamp guarantees no audio bleed.
    try { if (outgoingEl) outgoingEl.volume = 0; } catch {}

    // Audio-first path: Director sets muted=true on the play_clip event. The
    // soundtrack is coming from a standalone <audio> element (audioResponse
    // / pitchAudio), so this video element must stay silent or we double
    // the audio. We also skip the volume ramp below.
    const audioFirstMuted = !!stream.tier1.muted;
    incomingEl.muted = audioFirstMuted;
    incomingEl.loop = !!stream.tier1.loop;
    // Start muted in volume terms then ramp up alongside opacity to avoid a
    // hard audio cut. On audio-first we leave volume at 0 forever (muted).
    incomingEl.volume = 0;

    const expectedDurationMs = stream.tier1.expectedDurationMs;

    // Speculative preload reuse (Pass 2). If a routing_decision recently
    // primed the inactive element with this exact URL, we can skip the
    // full prepareFirstFrame round-trip entirely — the element already
    // has the first frame decoded + seeked. Slot match is required so we
    // don't accidentally promote the wrong element on a tier1 ping-pong.
    const incomingSlot = tier1ActiveIsA ? 'B' : 'A';
    const fullSrc = `${API_BASE}${url}`;
    const preloaded =
      tier1PreloadRef.current.url === url &&
      tier1PreloadRef.current.slot === incomingSlot &&
      incomingEl.src === fullSrc &&
      incomingEl.readyState >= 2 &&     // HAVE_CURRENT_DATA — first frame ready
      incomingEl.currentTime <= 0.01;   // not already playing from a previous run
    if (preloaded) {
      tier1PreloadRef.current = { url: null, slot: null };
      dlog('tier1', 'preload_hit', { intent: stream.tier1.intent, url, slot: incomingSlot });
    }

    const ready = preloaded
      ? Promise.resolve()
      : prepareFirstFrame(incomingEl, fullSrc);

    ready
      .then(() => {
        // Duration handshake (REVISIONS §4 — relaxed to 250ms after
        // commit 2c98beb's pipeline restructure). Both audio and video
        // now derive from the SAME audio_bytes (Wav2Lip ingests + muxes
        // the exact mp3 the standalone <audio> is playing), so genuine
        // drift is typically 30-80ms. Sources of remaining drift:
        //   • mp4 container re-encoding rounding (~10ms)
        //   • 33ms frame-quantization at 30fps
        //   • silence-trimming variance in the mux step
        //   • Wav2Lip occasionally padding audio to match exact frame count
        // 250ms gives ~3x headroom over expected drift while still
        // catching genuine bugs (e.g. accidentally feeding different audio
        // bytes to the two paths shows up as 1000ms+ drift). Removing
        // the handshake entirely is a regression — we silently desync
        // any time a future code path slips in different audio.
        if (audioFirstMuted && expectedDurationMs && incomingEl.duration) {
          const videoMs = incomingEl.duration * 1000;
          const driftMs = Math.abs(videoMs - expectedDurationMs);
          if (driftMs > 250) {
            console.warn('[LiveStage] audio-first duration drift', {
              videoMs: Math.round(videoMs), expectedDurationMs, driftMs: Math.round(driftMs),
            });
            stream.sendAck(stream.tier1.intent, url, 'skipped');
            throw new Error('duration_drift');
          }
        }
        return incomingEl.play();
      })
      .then(() => {
        // First frame is decoded AND positioned at t=0; opacity ramp
        // happens against a known-good frame, no head-jump pop. Blur
        // pulse fires HERE (not before play) so it's synchronised with
        // the same render frame the opacity ramp begins on — otherwise
        // the blur arc could lead the visual swap by 30-100ms (canplay
        // → play() round trip) and the seam moment wouldn't line up.
        blurStage(fadeMs);
        setTier1ActiveIsA(prev => !prev);
        setTier1Opacity(1);
        setOverlayVisible(stream.tier1.intent === 'response');
        stream.sendAck(stream.tier1.intent, url, 'started');
        dlog('tier1', 'play_started', {
          intent: stream.tier1.intent, url, fadeMs,
          muted: audioFirstMuted,
          slot: tier1ActiveIsA ? 'B' : 'A',
          preloaded,
        });

        if (audioFirstMuted) {
          // No audio ramp — the standalone <audio> already owns the
          // soundtrack. Pause the outgoing video after the opacity fade
          // so we don't keep two decoders running.
          setTimeout(() => {
            try { outgoingEl?.pause(); } catch {}
          }, fadeMs + 50);
          return;
        }

        // Default path: rAF audio fade alongside the CSS opacity transition.
        // t is clamped to [0, 1] on BOTH ends — without the lower clamp,
        // performance.now() jitter could produce a slightly negative t,
        // throwing IndexSizeError. Storing the rAF id in a ref lets the
        // NEXT Tier 1 transition cancel this loop before starting its own.
        const start = performance.now();
        function tick(now) {
          const t = Math.max(0, Math.min(1, (now - start) / fadeMs));
          try {
            incomingEl.volume = t;
            if (outgoingEl) outgoingEl.volume = 1 - t;
          } catch {
            tier1FadeRafRef.current = null;
            return;
          }
          if (t < 1) {
            tier1FadeRafRef.current = requestAnimationFrame(tick);
          } else {
            tier1FadeRafRef.current = null;
            try { outgoingEl?.pause(); } catch {}
          }
        }
        tier1FadeRafRef.current = requestAnimationFrame(tick);
      })
      .catch(err => {
        if (err?.message === 'duration_drift') return;
        console.warn('[LiveStage] tier1 play() rejected', err);
        stream.sendAck(stream.tier1.intent, url, 'stalled');
      });

    const onError = () => {
      stream.sendAck(stream.tier1.intent, url, 'stalled');
    };
    incomingEl.addEventListener('error', onError, { once: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stream.tier1?.url, stream.tier1?.ts]);

  // When the active Tier 1 ends naturally (non-looping clip), fade it out so
  // Tier 0 takes over.
  useEffect(() => {
    function onEnded(e) {
      // Only act if THIS element is the active one.
      const activeEl = tier1ActiveIsA ? tier1ARef.current : tier1BRef.current;
      if (e.target !== activeEl) return;
      // Blur the Tier 0 floor reveal — this seam is also a pose change
      // (avatar dropping back to idle) and benefits from the same edge-
      // softening trick as the in-bound crossfades.
      blurStage(stream.tier1?.fadeMs ?? TIER1_FADEOUT_MS);
      setTier1Opacity(0);
      setOverlayVisible(false);
      stream.sendAck(stream.tier1?.intent || '', stream.tier1?.url || '', 'ended');
    }
    const a = tier1ARef.current, b = tier1BRef.current;
    a?.addEventListener('ended', onEnded);
    b?.addEventListener('ended', onEnded);
    return () => {
      a?.removeEventListener('ended', onEnded);
      b?.removeEventListener('ended', onEnded);
    };
  }, [tier1ActiveIsA, stream.tier1?.intent, stream.tier1?.url, stream.sendAck]);

  // ── Audio-first playback ────────────────────────────────────────────────
  // Single hidden <audio> element, mounted once, reused across plays. Driven
  // by audioResponse (live comment responses) and pitchAudio (30s Veo pitch).
  // Pitch wins ties — if both fire in the same render, we play the pitch
  // (rare; pitch_audio also clears any in-flight responseAudio because the
  // backend won't fire a comment response while a pitch is broadcasting).
  useEffect(() => {
    const a = audioRef.current;
    if (!a) return;
    // Pick whichever dispatch is newer by `seq`. Avoids the race where a
    // pitch starts and a stale comment response from the same tick stomps it.
    const respSeq = audioResponse?.seq || 0;
    const pitchSeq = pitchAudio?.seq || 0;
    const next =
      pitchSeq > respSeq && pitchAudio
        ? { kind: 'pitch', payload: pitchAudio }
        : audioResponse
          ? { kind: 'response', payload: audioResponse }
          : null;

    if (!next) {
      // Nothing to play — pause + clear so a stale src doesn't auto-resume.
      try { a.pause(); a.currentTime = 0; } catch {}
      setAudioPlaying(null);
      return;
    }

    // Identical seq → no-op (avoid restarting playback on every parent re-render).
    if (audioPlaying && audioPlaying.payload.seq === next.payload.seq) return;

    a.src = `${API_BASE_FOR_AUDIO}${next.payload.url}`;
    a.preload = 'auto';
    a.volume = 1;
    setAudioPlaying(next);

    // Defense-in-depth: when the standalone <audio> takes over the
    // soundtrack, force-mute both Tier 1 video elements. This prevents the
    // dual-audio bug where a still-playing bridge clip (or a response video
    // that arrived with embedded audio while the standalone audio was also
    // playing) keeps blasting its own TTS audio over the standalone audio.
    // Their VIDEO continues — only the audio is silenced. The next regular
    // Tier 1 emit will reset muted=audioFirstMuted as part of its setup.
    try { if (tier1ARef.current) tier1ARef.current.muted = true; } catch {}
    try { if (tier1BRef.current) tier1BRef.current.muted = true; } catch {}
    dlog('audio', 'play_started', {
      kind: next.kind, seq: next.payload.seq,
      url: next.payload.url,
      tier1_force_muted: true,
    });

    const playPromise = a.play();
    if (playPromise && typeof playPromise.catch === 'function') {
      playPromise.catch((err) => {
        // Most common cause: StartDemoOverlay wasn't completed before this
        // fired (operator skipped the button or the persisted unlock flag
        // is stale on a different machine). Log and continue — the user
        // hears nothing this round but the rest of the demo still works.
        console.warn('[LiveStage] audio-first play() rejected', err);
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [audioResponse?.seq, pitchAudio?.seq]);

  // Audio ended → tell parent so it can clear the matching state slot.
  useEffect(() => {
    const a = audioRef.current;
    if (!a) return;
    function onEnded() {
      const playing = audioPlaying;
      if (!playing) return;
      console.log('[LiveStage] audio-first ended', playing.kind, playing.payload.seq);
      dlog('audio', 'ended', { kind: playing.kind, seq: playing.payload.seq });
      onAudioEnded?.(playing.kind);
      setAudioPlaying(null);
    }
    a.addEventListener('ended', onEnded);
    return () => a.removeEventListener('ended', onEnded);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [audioPlaying?.payload?.seq]);

  // Unmute on first user gesture (mobile autoplay restriction). Tier 0 stays
  // muted always; Tier 1 gets the unmute.
  useEffect(() => {
    function unmute() {
      const a = tier1ARef.current, b = tier1BRef.current;
      [a, b].forEach(el => { if (el) el.muted = false; });
      window.removeEventListener('click', unmute);
      window.removeEventListener('keydown', unmute);
    }
    window.addEventListener('click', unmute);
    window.addEventListener('keydown', unmute);
    return () => {
      window.removeEventListener('click', unmute);
      window.removeEventListener('keydown', unmute);
    };
  }, []);

  const oldestPending = pendingComments[0];
  const tier1FadeStyle = {
    transition: `opacity ${stream.tier1?.fadeMs ?? TIER1_FADEOUT_MS}ms ease`,
  };

  return (
    <div style={styles.container}>
      <div style={styles.stage}>
        {/* Wrapper around just the four <video> elements so the blur
            pulse (filter:blur on this div, ramped via blurStage) only
            applies to the avatar imagery and leaves the LIVE pill /
            voice pill / routing badge / comment card / karaoke captions
            sharp. Empty `style` lets blurStage own the inline filter. */}
        <div ref={stageVideosRef} style={styles.stageVideos}>
          {/* Tier 0A: always-on idle layer (ping-pong A) */}
          <video
            ref={tier0ARef}
            playsInline
            autoPlay
            loop
            muted
            controls={false}
            style={{
              ...styles.video,
              ...styles.tier0,
              opacity: tier0ActiveIsA ? tier0Opacity : 0,
              transition: `opacity ${stream.tier0?.fadeMs ?? 600}ms ease`,
            }}
          />

          {/* Tier 0B: always-on idle layer (ping-pong B) */}
          <video
            ref={tier0BRef}
            playsInline
            loop
            muted
            controls={false}
            style={{
              ...styles.video,
              ...styles.tier0,
              opacity: !tier0ActiveIsA ? tier0Opacity : 0,
              transition: `opacity ${stream.tier0?.fadeMs ?? 600}ms ease`,
            }}
          />

          {/* Tier 1A */}
          <video
            ref={tier1ARef}
            playsInline
            controls={false}
            style={{
              ...styles.video,
              ...styles.tier1,
              opacity: tier1ActiveIsA ? tier1Opacity : 0,
              ...tier1FadeStyle,
            }}
          />

          {/* Tier 1B */}
          <video
            ref={tier1BRef}
            playsInline
            controls={false}
            style={{
              ...styles.video,
              ...styles.tier1,
              opacity: !tier1ActiveIsA ? tier1Opacity : 0,
              ...tier1FadeStyle,
            }}
          />
        </div>

        {/* LIVE pill — visible whenever Tier 1 is active (i.e. avatar is reactive).
            Hidden in overlay mode because TikTokShopOverlay paints its own
            (more prominent) LIVE badge in the top-right of the 9:16 frame. */}
        {tier1Opacity > 0 && !inOverlay && (
          <div style={styles.livePill}>
            <span style={styles.liveDot} />
            LIVE
          </div>
        )}

        {/* Voice-flow state — drives the audience's mental model of where we are
            in the request. Cody's VoiceMic + router will broadcast the events
            that flip voiceState; this component just renders. Hidden in
            minimalChrome — the avatar's actual reaction (clip swap + karaoke)
            IS the state in that mode. */}
        {voiceState && voiceState !== 'idle' && !minimalChrome && (
          <VoiceStatePill state={voiceState} />
        )}

        {/* Routing decision badge — pops on every comment routed locally vs cloud.
            Auto-fades after 3.2s so the stage stays clean. Hidden in minimalChrome
            (it's a dev/operator signal, not part of the audience-facing reaction). */}
        {routingDecision && !minimalChrome && (
          <RoutingBadge decision={routingDecision} />
        )}

        {/* Pending comment chip — operator dashboard only. In /stage the
            TikTokShopOverlay chat rail already shows incoming comments
            with @username:text, and this chip would land on top of the
            LIVE badge / viewers pill cluster (top-right). */}
        {oldestPending && !overlayVisible && !inOverlay && (
          <PendingChip text={oldestPending.text} startedAt={oldestPending.t0} />
        )}

        {/* Floating comment card — operator dashboard only. In /stage the
            chat rail surfaces the comment text and the avatar speaks the
            answer aloud; the floating card would just bury behind the
            buyDock + chatRail and add visual noise. */}
        {overlayVisible && responseVideo && !inOverlay && (
          <div style={styles.commentOverlay}>
            <div style={styles.commentCard}>
              <div style={styles.commentHeader}>
                <span style={styles.commentAvatar}>👤</span>
                <div style={{ flex: 1 }}>
                  <div style={styles.commentUser}>viewer</div>
                  <div style={styles.commentText}>{responseVideo.comment}</div>
                </div>
                {responseVideo.total_ms && (
                  <span style={styles.latencyBadge}>
                    ⚡ {(responseVideo.total_ms / 1000).toFixed(1)}s
                  </span>
                )}
              </div>
              {responseVideo.response && (
                <div style={styles.responseLine}>
                  → "{responseVideo.response}"
                </div>
              )}
            </div>
          </div>
        )}

        {/* Hidden audio element — drives the audio-first playback path AND
            the 30s Veo pitch path. Kept hidden because we don't want
            controls visible on stage; KaraokeCaptions reads currentTime
            off audioRef.current to highlight words. */}
        <audio
          ref={audioRef}
          playsInline
          style={{ display: 'none' }}
        />

        {/* Karaoke captions + translation chip — both driven by audioPlaying.
            Captions sync to currentTime via rAF; chip primes the audience
            for "live, captioned" so any audio/visual micro-asynchrony reads
            as caption-layer behaviour rather than bad lip-sync. */}
        <KaraokeCaptions
          audioRef={audioRef}
          wordTimings={audioPlaying?.payload?.word_timings || null}
          windowSize={audioPlaying?.kind === 'pitch' ? 9 : 7}
          visible={!!audioPlaying}
        />
        {/* TranslationChip is itself a pill ("🔴 LIVE · auto-captioned"),
            so suppress in minimalChrome — captions speak for themselves. */}
        <TranslationChip
          visible={!!audioPlaying && !minimalChrome}
          variant={audioPlaying?.kind === 'pitch' ? 'pitch' : 'live'}
        />

        {/* Empty placeholder only shown if Tier 0 has nothing to play */}
        {!stream.tier0 && !lastTier0Url.current && (
          <div style={styles.placeholder}>
            <span style={{ fontSize: 96 }}>🎥</span>
            <p style={{ color: '#71717a', marginTop: 12, fontSize: 16 }}>
              {productData ? 'Warming up the avatar…' : 'Drop a product video to start'}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

// ── useVoiceStage — listen for Cody's voice + router events ──────────────────
// Pure observer of the shared WS. No prop drilling, no double subscription
// (we attach via addEventListener so we don't clobber the existing handler).
//
// State machine:
//   voice_transcript   → 'thinking'   (transcript landed, router about to fire)
//   routing_decision   → 'responding' (tool picked, render kicking off)
//   comment_response_video → null     (response is on stage, hand off to LIVE)
//
// Safety auto-clear after 12s so a dropped follow-up never sticks the pill.
function useVoiceStage({ wsRef, connected }) {
  const [voiceState, setVoiceState] = useState(null);
  const [routingDecision, setRoutingDecision] = useState(null);
  const clearTimerRef = useRef(null);

  // Dep includes `connected` so the listener re-attaches when the WS opens.
  // Without it, this effect runs on mount with wsRef.current === null
  // (useEmpireSocket's connect() runs in a parent useEffect, AFTER child
  // effects), bails, and never re-triggers — voice pill + routing badge
  // would silently never appear.
  useEffect(() => {
    const ws = wsRef?.current;
    if (!ws) return;
    function setStateSafe(s) {
      setVoiceState(s);
      if (clearTimerRef.current) clearTimeout(clearTimerRef.current);
      if (s) clearTimerRef.current = setTimeout(() => setVoiceState(null), 12_000);
    }
    function onMessage(e) {
      let msg;
      try { msg = JSON.parse(e.data); } catch { return; }
      switch (msg.type) {
        case 'voice_state':
          // Director-driven explicit state. Wins over inferred state.
          setStateSafe(msg.state || null);
          break;
        case 'voice_transcript':
          setStateSafe('thinking');
          break;
        case 'routing_decision':
          setRoutingDecision(msg);
          setStateSafe('responding');
          break;
        case 'comment_response_video':
          setStateSafe(null);
          break;
      }
    }
    ws.addEventListener('message', onMessage);
    return () => {
      ws.removeEventListener('message', onMessage);
      if (clearTimerRef.current) clearTimeout(clearTimerRef.current);
    };
  }, [wsRef, connected]);

  return { voiceState, routingDecision };
}

function PendingChip({ text, startedAt }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setElapsed(((Date.now() - startedAt) / 1000)), 100);
    return () => clearInterval(id);
  }, [startedAt]);
  return (
    <div style={styles.pendingChip}>
      <span style={styles.pendingDot} />
      <span style={{ fontSize: 14, color: '#fde68a', fontWeight: 600 }}>
        Reading "{text.slice(0, 40)}{text.length > 40 ? '…' : ''}"
      </span>
      <span style={styles.pendingTimer}>{elapsed.toFixed(1)}s</span>
    </div>
  );
}

// ── Voice flow state pill ────────────────────────────────────────────────────
// Lives at the top-center of the stage when the voice path is mid-flight.
// Each state has its own color + message so the audience knows exactly
// where we are in the pipeline. 10ft readable.
const VOICE_STATES = {
  transcribing: {
    label: 'TRANSCRIBING',
    sub: 'Cactus + Gemma 4 · on-device',
    color: '#22d3ee',
    bg: 'rgba(8, 51, 68, 0.92)',
    border: '#0e7490',
  },
  thinking: {
    label: 'ROUTING',
    sub: 'FunctionGemma · picking tool',
    color: '#a855f7',
    bg: 'rgba(45, 16, 78, 0.92)',
    border: '#7c3aed',
  },
  responding: {
    label: 'RESPONDING',
    sub: 'Wav2Lip + GFPGAN · render',
    color: '#22c55e',
    bg: 'rgba(6, 51, 32, 0.92)',
    border: '#16a34a',
  },
};

function VoiceStatePill({ state }) {
  const cfg = VOICE_STATES[state] || VOICE_STATES.transcribing;
  return (
    <div
      style={{
        ...styles.voicePill,
        background: cfg.bg,
        borderColor: cfg.border,
        color: cfg.color,
      }}
    >
      <span style={{ ...styles.voicePillDot, background: cfg.color }} />
      <div style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.1 }}>
        <span style={styles.voicePillLabel}>{cfg.label}</span>
        <span style={styles.voicePillSub}>{cfg.sub}</span>
      </div>
    </div>
  );
}

// ── Routing decision badge ───────────────────────────────────────────────────
// Shows up briefly each time the FunctionGemma router fires a tool. Local
// routes are framed as savings, cloud routes as deliberate escalation.
function RoutingBadge({ decision }) {
  const [visible, setVisible] = useState(true);
  useEffect(() => {
    setVisible(true);
    const id = setTimeout(() => setVisible(false), 3200);
    return () => clearTimeout(id);
  }, [decision]);
  if (!visible) return null;

  const local = !!decision.was_local;
  const tone = local
    ? { color: '#4ade80', bg: 'rgba(6, 51, 32, 0.92)', border: '#16a34a', label: 'LOCAL' }
    : { color: '#fbbf24', bg: 'rgba(60, 38, 4, 0.92)', border: '#d97706', label: 'CLOUD' };
  const toolLabel = String(decision.tool || 'unknown').replace(/_/g, ' ');
  const ms = Number.isFinite(decision.ms) ? `${decision.ms}ms` : null;
  const saved = Number.isFinite(decision.cost_saved_usd) && decision.cost_saved_usd > 0
    ? `· saved $${decision.cost_saved_usd.toFixed(4)}`
    : null;

  return (
    <div
      style={{
        ...styles.routingBadge,
        background: tone.bg,
        borderColor: tone.border,
        color: tone.color,
      }}
    >
      <span style={{ ...styles.routingBadgeTag, background: tone.color, color: '#0a0a0a' }}>
        {tone.label}
      </span>
      <span style={styles.routingBadgeBody}>
        <strong>{toolLabel}</strong>
        {ms && <span style={styles.routingBadgeMeta}> · {ms}</span>}
        {saved && <span style={styles.routingBadgeMeta}>{` ${saved}`}</span>}
      </span>
    </div>
  );
}

const styles = {
  container: {
    background: '#09090b', borderRadius: 16, padding: 0, height: '100%',
    display: 'flex', flexDirection: 'column', overflow: 'hidden',
    border: '1px solid #27272a',
  },
  stage: {
    position: 'relative', width: '100%', flex: 1, minHeight: 0,
    background: '#000',
  },
  stageVideos: {
    position: 'absolute', inset: 0,
    // Hint the GPU compositor to keep this layer ready for filter
    // changes — without it the FIRST blur tick after a fresh mount
    // can drop a frame while the browser promotes the layer.
    willChange: 'filter',
  },
  video: {
    position: 'absolute', inset: 0,
    width: '100%', height: '100%', objectFit: 'contain', background: '#000',
    imageRendering: 'auto',
    WebkitTransform: 'translateZ(0)',
  },
  tier0: { zIndex: 0 },
  tier1: { zIndex: 1, opacity: 0 },
  placeholder: {
    position: 'absolute', inset: 0,
    display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
    textAlign: 'center', zIndex: 2,
  },
  livePill: {
    position: 'absolute', top: 14, left: 14, zIndex: 5,
    background: 'rgba(239,68,68,0.95)', color: '#fff',
    padding: '6px 14px', borderRadius: 8, fontSize: 13, fontWeight: 900, letterSpacing: 1.5,
    display: 'flex', alignItems: 'center', gap: 8,
    boxShadow: '0 2px 12px rgba(239,68,68,0.4)',
  },
  liveDot: {
    width: 9, height: 9, borderRadius: 5, background: '#fff',
    animation: 'pulse 1.4s ease-in-out infinite',
  },
  pendingChip: {
    position: 'absolute', top: 14, right: 14, zIndex: 5,
    background: 'rgba(120,53,15,0.94)', border: '1px solid #f59e0b',
    padding: '8px 14px', borderRadius: 999,
    display: 'flex', alignItems: 'center', gap: 10, maxWidth: '60%',
    backdropFilter: 'blur(6px)',
    boxShadow: '0 2px 12px rgba(245,158,11,0.25)',
  },
  pendingDot: {
    width: 10, height: 10, borderRadius: 5, background: '#f59e0b',
    animation: 'pulse 1s ease-in-out infinite',
  },
  pendingTimer: { fontSize: 13, color: '#fcd34d', fontVariantNumeric: 'tabular-nums', fontWeight: 800 },
  voicePill: {
    position: 'absolute', top: 14, left: '50%', transform: 'translateX(-50%)', zIndex: 6,
    border: '1px solid', borderRadius: 12,
    padding: '8px 14px',
    display: 'flex', alignItems: 'center', gap: 10,
    backdropFilter: 'blur(8px)',
    boxShadow: '0 4px 16px rgba(0,0,0,0.35)',
    animation: 'voicePillIn 220ms cubic-bezier(0.4, 0.0, 0.2, 1)',
  },
  voicePillDot: {
    width: 10, height: 10, borderRadius: 5,
    animation: 'pulse 0.9s ease-in-out infinite',
    boxShadow: '0 0 12px currentColor',
  },
  voicePillLabel: {
    fontSize: 13, fontWeight: 900, letterSpacing: 1.6,
  },
  voicePillSub: {
    fontSize: 10, fontWeight: 600, letterSpacing: 0.6, opacity: 0.85,
    fontFamily: 'ui-monospace, SFMono-Regular, monospace',
  },
  routingBadge: {
    position: 'absolute', bottom: 18, left: '50%', transform: 'translateX(-50%)', zIndex: 6,
    border: '1px solid', borderRadius: 999,
    padding: '6px 6px 6px 6px',
    display: 'flex', alignItems: 'center', gap: 10,
    backdropFilter: 'blur(8px)',
    boxShadow: '0 4px 16px rgba(0,0,0,0.45)',
    animation: 'routingBadgeIn 280ms cubic-bezier(0.4, 0.0, 0.2, 1)',
    maxWidth: '70%',
  },
  routingBadgeTag: {
    fontSize: 10, fontWeight: 900, letterSpacing: 1.4,
    padding: '3px 8px', borderRadius: 999,
  },
  routingBadgeBody: {
    fontSize: 13, fontWeight: 700, paddingRight: 12,
    textTransform: 'uppercase', letterSpacing: 0.5,
  },
  routingBadgeMeta: {
    opacity: 0.8, fontWeight: 500,
    fontFamily: 'ui-monospace, SFMono-Regular, monospace',
    textTransform: 'none',
    fontSize: 12,
  },
  commentOverlay: {
    position: 'absolute', left: 0, right: 0, bottom: 0, zIndex: 5,
    padding: 16,
    background: 'linear-gradient(to top, rgba(0,0,0,0.85), rgba(0,0,0,0))',
    animation: 'slideUp 0.32s ease-out',
  },
  commentCard: {
    background: 'rgba(24,24,27,0.92)', border: '1px solid #3f3f46',
    borderRadius: 12, padding: 12,
    backdropFilter: 'blur(8px)',
  },
  commentHeader: { display: 'flex', alignItems: 'center', gap: 10 },
  commentAvatar: {
    fontSize: 24, width: 36, height: 36, borderRadius: 18,
    background: '#27272a', display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  commentUser: { color: '#a1a1aa', fontSize: 12, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1 },
  commentText: { color: '#fafafa', fontSize: 16, fontWeight: 500 },
  latencyBadge: {
    background: '#16a34a', color: '#fff', padding: '6px 12px', borderRadius: 999,
    fontSize: 13, fontWeight: 900, letterSpacing: 0.5,
    fontVariantNumeric: 'tabular-nums',
    boxShadow: '0 2px 8px rgba(34,197,94,0.35)',
  },
  responseLine: {
    color: '#22c55e', fontSize: 14, fontStyle: 'italic', marginTop: 8, paddingLeft: 46,
  },
};

// ── Frame-accurate clip preparation (REVISIONS §17 fix) ─────────────────────
// Loads a video src into an element, seeks to t=0, and resolves only after
// the seeked event fires + the element has decoded the first frame
// (readyState >= HAVE_CURRENT_DATA). Used by both Tier 0 and Tier 1
// drivers so the opacity ramp always starts against a known-good first
// frame — eliminates the "her head jumped" pop visible on slo-mo replay
// when canplay returned a frame from an unspecified position.
//
// Caller must set incomingEl.muted/loop/volume BEFORE calling. Promise
// rejects on element error or 4-second hard timeout (network stall) so
// the caller can surface a stalled ack instead of hanging forever.
//
// Idempotent on a hot-cache hit: if the element is already at this exact
// src + t=0 and decoded, resolves immediately on the next microtask
// without re-loading. This is what makes the speculative preload path
// (Tier 1 routing_decision pre-warm) zero-cost on the actual play.
function prepareFirstFrame(el, src, { timeoutMs = 4000 } = {}) {
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      el.removeEventListener('loadedmetadata', onLoadedMeta);
      el.removeEventListener('seeked', onSeeked);
      el.removeEventListener('error', onError);
      clearTimeout(timer);
    };

    function fail(err) {
      cleanup();
      reject(err instanceof Error ? err : new Error(String(err)));
    }

    function onError() {
      fail(el.error || 'video_error');
    }

    function trySeekToZero() {
      // currentTime = 0 is a no-op if we're already at 0 AND the element
      // hasn't seeked since load — so we force a seek for new srcs only.
      if (el.currentTime > 0.001) {
        el.currentTime = 0;
      } else {
        // Already at 0; emit a synthetic seeked tick so the resolve path
        // is uniform.
        Promise.resolve().then(onSeeked);
      }
    }

    function onLoadedMeta() {
      el.removeEventListener('loadedmetadata', onLoadedMeta);
      trySeekToZero();
    }

    function onSeeked() {
      // readyState >= 2 (HAVE_CURRENT_DATA) means the first frame is
      // decoded and available for compositing. Browsers usually have
      // this by the time seeked fires after a load, but we double-check.
      if (el.readyState >= 2) {
        cleanup();
        resolve();
      } else {
        // Rare: seeked fired but the frame isn't decoded yet. Wait one
        // canplay then resolve.
        const onCanPlay = () => {
          el.removeEventListener('canplay', onCanPlay);
          cleanup();
          resolve();
        };
        el.addEventListener('canplay', onCanPlay, { once: true });
      }
    }

    el.addEventListener('error', onError, { once: true });
    el.addEventListener('seeked', onSeeked);
    const timer = setTimeout(() => fail(new Error('prepare_timeout')), timeoutMs);

    // Hot path: same src + already at t=0 + already decoded → skip the
    // load entirely, just resolve. This is what makes speculative
    // preload zero-cost on play.
    if (el.src === src && el.readyState >= 2 && el.currentTime <= 0.01) {
      cleanup();
      resolve();
      return;
    }

    if (el.src !== src) {
      el.src = src;
      el.addEventListener('loadedmetadata', onLoadedMeta, { once: true });
      // Some browsers (Safari) need an explicit load() call after src
      // change for autoplay-disabled elements to start buffering.
      try { el.load(); } catch {}
    } else {
      // Same src, possibly mid-playback or post-end. Seek directly.
      trySeekToZero();
    }
  });
}

if (typeof document !== 'undefined' && !document.getElementById('livestage-keyframes')) {
  const s = document.createElement('style');
  s.id = 'livestage-keyframes';
  s.innerHTML = `
    @keyframes pulse { 0%,100% { opacity: 1 } 50% { opacity: 0.45 } }
    @keyframes slideUp { from { transform: translateY(100%); opacity: 0 } to { transform: translateY(0); opacity: 1 } }
    @keyframes voicePillIn {
      from { transform: translate(-50%, -8px); opacity: 0 }
      to   { transform: translate(-50%,  0);   opacity: 1 }
    }
    @keyframes routingBadgeIn {
      from { transform: translate(-50%, 8px); opacity: 0 }
      to   { transform: translate(-50%, 0);   opacity: 1 }
    }
  `;
  document.head.appendChild(s);
}
