import React, { useEffect, useRef, useState } from 'react';
import { useAvatarStream, TIER1_FADEOUT_MS } from '../hooks/useAvatarStream';

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
export function LiveStage({
  productData,
  pitchVideoUrl,           // kept for placeholder context only
  responseVideo,           // kept to drive the floating overlay UI
  pendingComments = [],
  liveStage,
  wsRef,
}) {
  const tier0Ref = useRef(null);
  const tier1ARef = useRef(null);
  const tier1BRef = useRef(null);

  // Which Tier 1 element is currently visible (true = A, false = B). The other
  // one is hidden, ready to receive the next clip.
  const [tier1ActiveIsA, setTier1ActiveIsA] = useState(true);
  // Visibility of the active Tier 1 element (0..1). Tier 0 sits underneath
  // and is always painting, so visibility 0 means "show idle".
  const [tier1Opacity, setTier1Opacity] = useState(0);
  const [overlayVisible, setOverlayVisible] = useState(false);

  const stream = useAvatarStream({ wsRef });

  // ── Tier 0 driver ────────────────────────────────────────────────────────
  // Drives the always-on idle layer. Loop, muted, crossfade-in on src change.
  const lastTier0Url = useRef(null);
  useEffect(() => {
    const v = tier0Ref.current;
    if (!v) return;

    const url = stream.tier0?.url;
    if (!url) {
      // No instruction from Director yet — make sure something is playing
      // so we never see a black stage. Falls back to the static state video.
      const fallback = `${API_BASE}/states/state_idle_pose_silent_1080p.mp4`;
      if (v.src !== fallback) {
        v.src = fallback;
        v.muted = true;
        v.loop = true;
        v.play().catch(() => {});
      }
      return;
    }

    // Director told us to play this Tier 0 clip; if it changed, swap src.
    // (Tier 0 single-element swap; the seamless crossfade between *two* Tier 0
    // elements is M3 polish — for M1 we just swap source which has a tiny
    // flash but is hidden by Tier 1 most of the time.)
    if (url !== lastTier0Url.current) {
      v.src = `${API_BASE}${url}`;
      v.muted = true;
      v.loop = stream.tier0.loop ?? true;
      v.play().then(() => stream.sendStageReady()).catch(() => {});
      lastTier0Url.current = url;
    }
  }, [stream.tier0?.url, stream.tier0?.loop]);

  // Send stage_ready once Tier 0 actually starts painting frames, so the
  // Director knows it's safe to send Tier 1 events.
  useEffect(() => {
    const v = tier0Ref.current;
    if (!v) return;
    const onPlaying = () => stream.sendStageReady();
    v.addEventListener('playing', onPlaying);
    return () => v.removeEventListener('playing', onPlaying);
    // stream.sendStageReady is stable enough; we want this once per mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Tier 1 driver ────────────────────────────────────────────────────────
  // On every new tier1 instruction, load it into the inactive element, wait
  // for canplay, then crossfade. Special case: empty url = fade out (idle release).
  useEffect(() => {
    if (!stream.tier1) return;
    const url = stream.tier1.url;
    const fadeMs = stream.tier1.fadeMs;

    // Idle release — fade Tier 1 out, reveal Tier 0 underneath.
    if (!url) {
      const activeEl = tier1ActiveIsA ? tier1ARef.current : tier1BRef.current;
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

    incomingEl.src = `${API_BASE}${url}`;
    incomingEl.muted = false;
    incomingEl.loop = !!stream.tier1.loop;
    // Start muted in volume terms then ramp up alongside opacity to avoid a
    // hard audio cut.
    incomingEl.volume = 0;

    const onCanPlay = () => {
      incomingEl.removeEventListener('canplay', onCanPlay);
      incomingEl.play().then(() => {
        // Swap which element is active and run the opacity transition.
        setTier1ActiveIsA(prev => !prev);
        setTier1Opacity(1);
        setOverlayVisible(stream.tier1.intent === 'response');
        stream.sendAck(stream.tier1.intent, url, 'started');

        // rAF audio fade alongside the CSS opacity transition.
        const start = performance.now();
        function tick(now) {
          const t = Math.min(1, (now - start) / fadeMs);
          incomingEl.volume = t;
          if (outgoingEl) outgoingEl.volume = 1 - t;
          if (t < 1) requestAnimationFrame(tick);
          else {
            // Pause outgoing after the fade so the decoder is free for the
            // next preload.
            try { outgoingEl?.pause(); } catch {}
          }
        }
        requestAnimationFrame(tick);
      }).catch(err => {
        console.warn('[LiveStage] tier1 play() rejected', err);
        stream.sendAck(stream.tier1.intent, url, 'stalled');
      });
    };
    incomingEl.addEventListener('canplay', onCanPlay, { once: true });

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
        {/* Tier 0: always-on idle layer */}
        <video
          ref={tier0Ref}
          playsInline
          autoPlay
          loop
          muted
          controls={false}
          style={{ ...styles.video, ...styles.tier0 }}
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

        {/* LIVE pill — visible whenever Tier 1 is active (i.e. avatar is reactive) */}
        {tier1Opacity > 0 && (
          <div style={styles.livePill}>
            <span style={styles.liveDot} />
            LIVE
          </div>
        )}

        {/* Pending comment chip */}
        {oldestPending && !overlayVisible && (
          <PendingChip text={oldestPending.text} startedAt={oldestPending.t0} />
        )}

        {/* Floating comment card */}
        {overlayVisible && responseVideo && (
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

function PendingChip({ text, startedAt }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setElapsed(((Date.now() - startedAt) / 1000)), 100);
    return () => clearInterval(id);
  }, [startedAt]);
  return (
    <div style={styles.pendingChip}>
      <span style={styles.pendingDot} />
      <span style={{ fontSize: 12, color: '#fde68a' }}>
        Reading "{text.slice(0, 40)}{text.length > 40 ? '…' : ''}"
      </span>
      <span style={styles.pendingTimer}>{elapsed.toFixed(1)}s</span>
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
    padding: '4px 10px', borderRadius: 6, fontSize: 11, fontWeight: 800, letterSpacing: 1,
    display: 'flex', alignItems: 'center', gap: 6,
  },
  liveDot: {
    width: 7, height: 7, borderRadius: 4, background: '#fff',
    animation: 'pulse 1.4s ease-in-out infinite',
  },
  pendingChip: {
    position: 'absolute', top: 14, right: 14, zIndex: 5,
    background: 'rgba(120,53,15,0.92)', border: '1px solid #f59e0b',
    padding: '6px 10px', borderRadius: 999,
    display: 'flex', alignItems: 'center', gap: 8, maxWidth: '60%',
    backdropFilter: 'blur(6px)',
  },
  pendingDot: {
    width: 8, height: 8, borderRadius: 4, background: '#f59e0b',
    animation: 'pulse 1s ease-in-out infinite',
  },
  pendingTimer: { fontSize: 11, color: '#fcd34d', fontVariantNumeric: 'tabular-nums', fontWeight: 700 },
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
  commentUser: { color: '#a1a1aa', fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1 },
  commentText: { color: '#fafafa', fontSize: 14, fontWeight: 500 },
  latencyBadge: {
    background: '#16a34a', color: '#fff', padding: '4px 10px', borderRadius: 999,
    fontSize: 11, fontWeight: 800, letterSpacing: 0.5,
  },
  responseLine: {
    color: '#22c55e', fontSize: 12, fontStyle: 'italic', marginTop: 6, paddingLeft: 46,
  },
};

if (typeof document !== 'undefined' && !document.getElementById('livestage-keyframes')) {
  const s = document.createElement('style');
  s.id = 'livestage-keyframes';
  s.innerHTML = `
    @keyframes pulse { 0%,100% { opacity: 1 } 50% { opacity: 0.45 } }
    @keyframes slideUp { from { transform: translateY(100%); opacity: 0 } to { transform: translateY(0); opacity: 1 } }
  `;
  document.head.appendChild(s);
}
