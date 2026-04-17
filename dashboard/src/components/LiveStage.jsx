import React, { useEffect, useMemo, useRef, useState } from 'react';

const API_BASE = `http://${window.location.hostname}:8000`;

/**
 * Cinema stage. Shows the live AI seller as a 16:9 video.
 * Priority: latest comment-response video (live), else pitch video (loop), else placeholder.
 * A floating "comment card" slides up over the video while a response is playing
 * so viewers can see the avatar reading and reacting.
 */
export function LiveStage({
  productData,
  pitchVideoUrl,
  responseVideo,        // { url, comment, response, total_ms, ... }
  pendingComments = [], // [{id, text, t0}]
  liveStage,
}) {
  const videoRef = useRef(null);
  const [overlayVisible, setOverlayVisible] = useState(false);
  const [activeKind, setActiveKind] = useState('idle'); // 'response' | 'pitch' | 'idle'

  // Choose source: latest response wins, then pitch, else nothing.
  const source = useMemo(() => {
    if (responseVideo?.url) return { url: responseVideo.url, kind: 'response' };
    if (pitchVideoUrl) return { url: pitchVideoUrl, kind: 'pitch' };
    return null;
  }, [responseVideo, pitchVideoUrl]);

  // Play the chosen source whenever it changes.
  useEffect(() => {
    if (!source || !videoRef.current) return;
    const v = videoRef.current;
    v.src = `${API_BASE}${source.url}`;
    v.muted = false;
    v.loop = source.kind === 'pitch';
    v.play().catch(() => {});
    setActiveKind(source.kind);
    setOverlayVisible(source.kind === 'response');
  }, [source?.url, source?.kind]);

  // Hide the overlay when the response video ends.
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const onEnded = () => {
      if (activeKind === 'response') {
        setOverlayVisible(false);
      }
    };
    v.addEventListener('ended', onEnded);
    return () => v.removeEventListener('ended', onEnded);
  }, [activeKind]);

  const oldestPending = pendingComments[0];

  return (
    <div style={styles.container}>
      <div style={styles.stage}>
        {source ? (
          <video
            ref={videoRef}
            playsInline
            autoPlay
            controls={false}
            style={styles.video}
          />
        ) : (
          <div style={styles.placeholder}>
            <span style={{ fontSize: 96 }}>🎥</span>
            <p style={{ color: '#71717a', marginTop: 12, fontSize: 16 }}>
              {liveStage === 'INTRO'
                ? productData ? 'Warming up the avatar…' : 'Drop a product video to start'
                : 'Waiting for next response…'}
            </p>
          </div>
        )}

        {/* LIVE pill */}
        {activeKind !== 'idle' && (
          <div style={styles.livePill}>
            <span style={styles.liveDot} />
            LIVE
          </div>
        )}

        {/* Pending comment chip (top-right) — shows only while waiting for a response */}
        {oldestPending && !overlayVisible && (
          <PendingChip text={oldestPending.text} startedAt={oldestPending.t0} />
        )}

        {/* Floating comment card while avatar is reading + reacting */}
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
    background: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  video: {
    width: '100%', height: '100%', objectFit: 'contain', background: '#000',
    // Crisp downscale on Retina displays; the rendered MP4 is already 1080p so
    // any browser-side resampling should preserve detail rather than blur it.
    imageRendering: 'auto',
    WebkitTransform: 'translateZ(0)',
  },
  placeholder: {
    textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center',
  },
  livePill: {
    position: 'absolute', top: 14, left: 14,
    background: 'rgba(239,68,68,0.95)', color: '#fff',
    padding: '4px 10px', borderRadius: 6, fontSize: 11, fontWeight: 800, letterSpacing: 1,
    display: 'flex', alignItems: 'center', gap: 6,
  },
  liveDot: {
    width: 7, height: 7, borderRadius: 4, background: '#fff',
    animation: 'pulse 1.4s ease-in-out infinite',
  },
  pendingChip: {
    position: 'absolute', top: 14, right: 14,
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
    position: 'absolute', left: 0, right: 0, bottom: 0,
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

// Inject CSS keyframes once (fine to do at module level — idempotent in StrictMode).
if (typeof document !== 'undefined' && !document.getElementById('livestage-keyframes')) {
  const s = document.createElement('style');
  s.id = 'livestage-keyframes';
  s.innerHTML = `
    @keyframes pulse { 0%,100% { opacity: 1 } 50% { opacity: 0.45 } }
    @keyframes slideUp { from { transform: translateY(100%); opacity: 0 } to { transform: translateY(0); opacity: 1 } }
  `;
  document.head.appendChild(s);
}
