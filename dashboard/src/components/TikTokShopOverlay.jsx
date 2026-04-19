import React, { useEffect, useRef, useState } from 'react';
import { LiveStage } from './LiveStage';

/**
 * TikTokShopOverlay — the demo's framing layer.
 *
 * Renders the existing LiveStage avatar inside a vertical 9:16 phone-screen
 * silhouette centered on a 16:9 black stage. Adds the visual conventions of
 * a live commerce stream so the audience's brain instantly maps Zo to
 * the $68B TikTok Shop / Whatnot / Instagram Live category, without
 * actually needing real platform integration.
 *
 * Per the design doc (DESIGN_PRINCIPLES.md / design plan), we deliberately
 * avoid TikTok's literal logo or trade dress — the visual conventions
 * (vertical 9:16, hearts, BUY button, scrolling chat) carry all the
 * recognition we need. Wordmark stays generic ("LIVE COMMERCE").
 *
 * Props pass straight through to LiveStage (productData, pitchVideoUrl,
 * responseVideo, pendingComments, liveStage, wsRef) so the demo can keep
 * using the existing pipeline state with zero rewiring.
 *
 * Audience comments come in via the new `audience_comment` WS event
 * (server side wired in backend/main.py /api/audience_comment). Each one
 * slides up from the bottom of the chat rail, lives 8s, then fades out.
 *
 * The Cost Ticker and compact RoutingPanel are NOT children of this
 * component — they live on the bezel as siblings in StageView. That keeps
 * this file focused on "what the audience sees inside the phone screen."
 */

const HEART_LIFETIME_MS = 4200;
const COMMENT_LIFETIME_MS = 9000;

export function TikTokShopOverlay({
  productData,
  pitchVideoUrl,
  responseVideo,
  pendingComments,
  liveStage,
  wsRef,
  audioResponse,
  pitchAudio,
  onAudioEnded,
  productHandle = '@zo_demo',
}) {
  // Viewer count — pure visual, ticks up by 1-3 every 4-7s so the room
  // reads it as a real stream gathering traction. Seeded from a stable-feeling
  // value so it never starts at 0 during a stage demo.
  const [viewers, setViewers] = useState(1247);
  useEffect(() => {
    let alive = true;
    function bump() {
      if (!alive) return;
      setViewers(v => v + 1 + Math.floor(Math.random() * 3));  // +1..+3
      setTimeout(bump, 4000 + Math.floor(Math.random() * 3000));
    }
    const t = setTimeout(bump, 3500);
    return () => { alive = false; clearTimeout(t); };
  }, []);

  // Floating hearts — fired periodically AND on every audience comment so
  // the room sees the "engagement" reacting to their participation. Each
  // heart is an absolute-positioned <span> with a unique key + a CSS
  // animation that sweeps it from the BUY button up off-screen.
  const [hearts, setHearts] = useState([]);
  const heartIdRef = useRef(0);
  function spawnHeart() {
    const id = ++heartIdRef.current;
    const offset = Math.random() * 24 - 12;  // jitter so they don't stack
    const sway = Math.random() * 60 - 30;    // horizontal drift end
    setHearts(prev => [...prev.slice(-12), { id, offset, sway, born: Date.now() }]);
    setTimeout(() => {
      setHearts(prev => prev.filter(h => h.id !== id));
    }, HEART_LIFETIME_MS);
  }
  useEffect(() => {
    // Steady stream of background hearts so the rail feels alive even
    // before the QR audience starts engaging.
    const id = setInterval(spawnHeart, 1100);
    return () => clearInterval(id);
  }, []);

  // Chat rail comments. SINGLE source of truth: useEmpireSocket appends
  // every inbound comment (audience_comment from /api/audience_comment,
  // simulate_comment from ChatPanel, voice_transcript from VoiceMic) to
  // pendingComments — the same list ChatPanel uses on the operator
  // dashboard. We mirror new entries into our local rail with an 8s
  // lifetime + guard against double-pushing the same id.
  //
  // Earlier revisions of this file ALSO subscribed to `audience_comment`
  // directly here AND mirrored pendingComments. That double-counted every
  // QR-submitted comment (one bubble from the WS listener, one from the
  // pendingComments effect). Now there's exactly one path: pendingComments.
  const [comments, setComments] = useState([]);

  function pushComment(c) {
    const id = c.id || `${c.ts}-${Math.random().toString(36).slice(2, 7)}`;
    setComments(prev => [...prev.slice(-30), { ...c, id }]);
    setTimeout(() => {
      setComments(prev => prev.filter(x => x.id !== id));
    }, COMMENT_LIFETIME_MS);
  }

  // Hearts react to a new audience_comment specifically — the QR audience
  // is the demo's headline interaction, so each submission gets a 1-2 heart
  // burst on top of the steady idle stream. We listen on the WS directly
  // (not pendingComments) so the heart fires the instant the broadcast
  // lands rather than after the React render cycle for pendingComments.
  // Bubble rendering itself is owned by the pendingComments effect below
  // so we don't double-add the comment to the chat rail.
  useEffect(() => {
    const ws = wsRef?.current;
    if (!ws) return;
    function onMessage(e) {
      let msg;
      try { msg = JSON.parse(e.data); } catch { return; }
      if (msg.type !== 'audience_comment') return;
      spawnHeart();
      if (Math.random() > 0.5) setTimeout(spawnHeart, 180);
    }
    ws.addEventListener('message', onMessage);
    return () => ws.removeEventListener('message', onMessage);
  }, [wsRef]);

  // Mirror new pendingComments into the rail. Three flavors land here:
  //   - source: 'audience'  → @<username from QR form>, pink accent
  //   - source: 'voice'     → @voice_user (operator's mic), purple accent
  //   - (no source)         → @preview     (operator typed via ChatPanel)
  // seenPendingRef is mount-scoped so a comment that already animated in
  // doesn't re-push when the same id reappears (e.g. another effect re-fires).
  const seenPendingRef = useRef(new Set());
  useEffect(() => {
    if (!pendingComments) return;
    for (const p of pendingComments) {
      if (seenPendingRef.current.has(p.id)) continue;
      seenPendingRef.current.add(p.id);
      const username = p.source === 'audience'
        ? (p.username || 'guest')
        : p.source === 'voice'
          ? 'voice_user'
          : 'preview';
      pushComment({
        id: `pc_${p.id}`,
        username,
        text: p.text || '',
        ts: p.t0 || Date.now(),
        source: p.source === 'audience' ? 'audience' : 'preview',
      });
    }
  }, [pendingComments]);

  const productName = productData?.name || 'New Drop';
  const productPrice = productData?.price || '—';

  return (
    <div style={styles.outer}>
      {/* Phone-screen safe area: vertical 9:16 inside a centered card on
          the 16:9 stage. Aspect ratio is enforced via aspect-ratio CSS
          (Vite + modern Chrome handle this fine), with a max-height fall-
          back so the frame never overflows the viewport on weird displays. */}
      <div style={styles.phoneFrame}>
        {/* Live avatar layer — the existing LiveStage handles all the
            video machinery (Tier 0/1, voice pills, routing badge, pending
            chips, response overlay). We just give it a vertical container. */}
        <div style={styles.stageFill}>
          <LiveStage
            productData={productData}
            pitchVideoUrl={pitchVideoUrl}
            responseVideo={responseVideo}
            pendingComments={pendingComments}
            liveStage={liveStage}
            wsRef={wsRef}
            audioResponse={audioResponse}
            pitchAudio={pitchAudio}
            onAudioEnded={onAudioEnded}
            inOverlay
          />
        </div>

        {/* Top bar inside the 9:16: host handle + LIVE pill + viewers + Follow.
            Sits on a subtle gradient so text stays legible over the avatar. */}
        <div style={styles.topBar}>
          <div style={styles.hostStrip}>
            <div style={styles.hostAvatar}>Z</div>
            <div style={styles.hostMeta}>
              <span style={styles.hostHandle}>{productHandle}</span>
              <span style={styles.hostSub}>Zo Live · selling now</span>
            </div>
            <button type="button" style={styles.followBtn}>+ Follow</button>
          </div>
          <div style={styles.topRightCluster}>
            <div style={styles.liveBadge}>
              <span style={styles.liveBadgeDot} />
              LIVE
            </div>
            <div style={styles.viewerPill}>
              <span style={styles.viewerEye}>👁</span>
              <span style={styles.viewerCount}>{formatViewers(viewers)}</span>
            </div>
          </div>
        </div>

        {/* Floating hearts rail. Lives just to the left of the BUY button so
            the eye reads "hearts coming up from the buy button" — the same
            interaction grammar TikTok / Instagram trained the audience on. */}
        <div style={styles.heartsRail}>
          {hearts.map(h => (
            <span
              key={h.id}
              style={{
                ...styles.heart,
                left: `calc(50% + ${h.offset}px)`,
                ['--sway']: `${h.sway}px`,
              }}
            >
              ❤
            </span>
          ))}
        </div>

        {/* Right action rail — share/comment/cart icons. Visual only; the
            BUY button is the action that matters. */}
        <div style={styles.rightRail}>
          <RailIcon icon="↗" label="Share" />
          <RailIcon icon="💬" label={comments.length > 0 ? String(comments.length) : 'Chat'} />
          <RailIcon icon="🛍" label="Cart" />
        </div>

        {/* BUY button — sourced from productData so it reads the actual
            item the avatar is selling. When no product loaded, shows a
            generic "Shop now" so the demo never reads empty. */}
        <div style={styles.buyDock}>
          <div style={styles.buyCard}>
            <div style={styles.buyMeta}>
              <span style={styles.buyName}>{productName}</span>
              <span style={styles.buyPrice}>{productPrice}</span>
            </div>
            <button type="button" style={styles.buyBtn}>
              <span style={styles.buyBtnLabel}>BUY NOW</span>
              <span style={styles.buyBtnArrow}>→</span>
            </button>
          </div>
        </div>

        {/* Comment scroll — left side, comments slide up from the bottom
            and fade as they near the buy dock. Each entry is @username:text
            in the recognizable platform format. */}
        <div style={styles.chatRail}>
          {comments.map(c => (
            <CommentLine key={c.id} comment={c} />
          ))}
        </div>
      </div>
    </div>
  );
}

function CommentLine({ comment }) {
  const isAudience = comment.source === 'audience';
  return (
    <div style={{
      ...styles.commentLine,
      borderLeftColor: isAudience ? '#ec4899' : '#7c3aed',
    }}>
      <span style={{ ...styles.commentUser, color: isAudience ? '#fbcfe8' : '#c4b5fd' }}>
        @{comment.username}
      </span>
      <span style={styles.commentText}>{comment.text}</span>
    </div>
  );
}

function RailIcon({ icon, label }) {
  return (
    <div style={styles.railIcon}>
      <span style={styles.railIconGlyph}>{icon}</span>
      <span style={styles.railIconLabel}>{label}</span>
    </div>
  );
}

function formatViewers(n) {
  if (n < 1000) return String(n);
  if (n < 10000) return `${(n / 1000).toFixed(1)}K`;
  return `${Math.round(n / 1000)}K`;
}

const styles = {
  outer: {
    width: '100%', height: '100%',
    background: '#000',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    overflow: 'hidden',
    position: 'relative',
  },
  phoneFrame: {
    position: 'relative',
    height: '100%',
    aspectRatio: '9 / 16',
    maxWidth: '100%',
    background: '#0a0a0a',
    overflow: 'hidden',
    borderRadius: 18,
    boxShadow: '0 0 80px rgba(124,58,237,0.18), 0 0 0 1px rgba(63,63,70,0.6)',
  },
  stageFill: {
    position: 'absolute', inset: 0, zIndex: 0,
  },
  topBar: {
    position: 'absolute', top: 0, left: 0, right: 0, zIndex: 5,
    padding: '14px 14px 24px',
    display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between',
    gap: 10,
    background: 'linear-gradient(to bottom, rgba(0,0,0,0.55), rgba(0,0,0,0))',
    pointerEvents: 'none',
  },
  hostStrip: {
    display: 'flex', alignItems: 'center', gap: 10,
    background: 'rgba(0,0,0,0.45)',
    border: '1px solid rgba(255,255,255,0.08)',
    backdropFilter: 'blur(10px)',
    padding: '5px 6px 5px 5px',
    borderRadius: 999,
    pointerEvents: 'auto',
  },
  hostAvatar: {
    width: 32, height: 32, borderRadius: 16,
    background: 'linear-gradient(135deg, #ec4899, #7c3aed)',
    color: '#fff', fontSize: 14, fontWeight: 900,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    border: '2px solid rgba(255,255,255,0.35)',
  },
  hostMeta: {
    display: 'flex', flexDirection: 'column', lineHeight: 1.1,
  },
  hostHandle: {
    color: '#fff', fontSize: 13, fontWeight: 800, letterSpacing: 0.3,
  },
  hostSub: {
    color: '#d4d4d8', fontSize: 10, fontWeight: 600,
  },
  followBtn: {
    background: '#ef4444', color: '#fff', fontSize: 11, fontWeight: 800,
    border: 'none', borderRadius: 999, padding: '5px 12px',
    cursor: 'pointer', letterSpacing: 0.4,
  },
  topRightCluster: {
    display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 6,
    pointerEvents: 'auto',
  },
  liveBadge: {
    display: 'flex', alignItems: 'center', gap: 6,
    background: '#ef4444', color: '#fff',
    padding: '4px 10px', borderRadius: 6,
    fontSize: 11, fontWeight: 900, letterSpacing: 1.4,
    boxShadow: '0 2px 8px rgba(239,68,68,0.4)',
  },
  liveBadgeDot: {
    width: 7, height: 7, borderRadius: 4, background: '#fff',
    animation: 'pulse 1.4s ease-in-out infinite',
  },
  viewerPill: {
    display: 'flex', alignItems: 'center', gap: 6,
    background: 'rgba(0,0,0,0.55)',
    border: '1px solid rgba(255,255,255,0.08)',
    backdropFilter: 'blur(10px)',
    padding: '3px 9px', borderRadius: 999,
    fontSize: 11, fontWeight: 700, color: '#fff',
  },
  viewerEye: { fontSize: 12 },
  viewerCount: { fontVariantNumeric: 'tabular-nums', letterSpacing: 0.2 },
  heartsRail: {
    position: 'absolute', right: 14, bottom: 96, top: '38%',
    width: 80, zIndex: 4, pointerEvents: 'none',
    overflow: 'visible',
  },
  heart: {
    position: 'absolute', bottom: 0,
    color: '#ec4899', fontSize: 24,
    textShadow: '0 0 12px rgba(236,72,153,0.65)',
    animation: `heartFloat ${HEART_LIFETIME_MS}ms ease-out forwards`,
    pointerEvents: 'none',
  },
  rightRail: {
    position: 'absolute', right: 12, bottom: 110, zIndex: 5,
    display: 'flex', flexDirection: 'column', gap: 16,
    alignItems: 'center',
  },
  railIcon: {
    display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2,
  },
  railIconGlyph: {
    width: 42, height: 42, borderRadius: 21,
    background: 'rgba(0,0,0,0.5)',
    border: '1px solid rgba(255,255,255,0.1)',
    backdropFilter: 'blur(6px)',
    color: '#fff', fontSize: 18,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  railIconLabel: {
    color: '#fff', fontSize: 10, fontWeight: 700,
    textShadow: '0 1px 4px rgba(0,0,0,0.85)',
    fontVariantNumeric: 'tabular-nums',
  },
  buyDock: {
    position: 'absolute', left: 12, right: 12, bottom: 14, zIndex: 6,
    display: 'flex', justifyContent: 'flex-end',
    pointerEvents: 'none',
  },
  buyCard: {
    background: 'rgba(0,0,0,0.65)',
    border: '1px solid rgba(255,255,255,0.12)',
    backdropFilter: 'blur(10px)',
    borderRadius: 14, padding: '8px 8px 8px 14px',
    display: 'flex', alignItems: 'center', gap: 12,
    pointerEvents: 'auto',
    maxWidth: '78%',
    boxShadow: '0 8px 24px rgba(0,0,0,0.55)',
  },
  buyMeta: {
    display: 'flex', flexDirection: 'column', lineHeight: 1.15,
    overflow: 'hidden',
  },
  buyName: {
    color: '#fff', fontSize: 12, fontWeight: 700,
    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
    maxWidth: 140,
  },
  buyPrice: {
    color: '#fbbf24', fontSize: 14, fontWeight: 900,
    fontVariantNumeric: 'tabular-nums', letterSpacing: -0.2,
  },
  buyBtn: {
    background: 'linear-gradient(135deg, #ec4899, #f43f5e)',
    color: '#fff', border: 'none', borderRadius: 10,
    padding: '8px 14px', cursor: 'pointer',
    display: 'flex', alignItems: 'center', gap: 8,
    boxShadow: '0 4px 12px rgba(244,63,94,0.45)',
  },
  buyBtnLabel: {
    fontSize: 12, fontWeight: 900, letterSpacing: 1.5,
  },
  buyBtnArrow: { fontSize: 14, fontWeight: 900 },
  chatRail: {
    position: 'absolute', left: 12, bottom: 88, zIndex: 5,
    width: '60%',
    display: 'flex', flexDirection: 'column', gap: 6,
    pointerEvents: 'none',
    maxHeight: '38%',
    overflow: 'hidden',
    justifyContent: 'flex-end',
  },
  commentLine: {
    background: 'rgba(0,0,0,0.45)',
    backdropFilter: 'blur(6px)',
    border: '1px solid rgba(255,255,255,0.06)',
    borderLeft: '3px solid #7c3aed',
    borderRadius: 10, padding: '5px 9px',
    display: 'flex', alignItems: 'baseline', gap: 6,
    fontSize: 12, lineHeight: 1.3,
    animation: 'commentRise 360ms cubic-bezier(0.2, 0.8, 0.2, 1.0)',
  },
  commentUser: {
    fontWeight: 800, fontSize: 11, flexShrink: 0,
  },
  commentText: {
    color: '#fafafa', fontWeight: 500, wordBreak: 'break-word',
  },
};

// Inject the keyframes once. The `pulse` keyframe already exists in
// LiveStage.jsx — we add `heartFloat` and `commentRise` here so the
// overlay is self-contained when LiveStage isn't yet mounted.
if (typeof document !== 'undefined' && !document.getElementById('tt-overlay-keyframes')) {
  const s = document.createElement('style');
  s.id = 'tt-overlay-keyframes';
  s.innerHTML = `
    @keyframes heartFloat {
      0%   { transform: translate(0, 0)     scale(0.7); opacity: 0; }
      12%  { transform: translate(0, -10px) scale(1.0); opacity: 1; }
      85%  { opacity: 1; }
      100% { transform: translate(var(--sway, 0), -380px) scale(1.15); opacity: 0; }
    }
    @keyframes commentRise {
      from { transform: translateY(14px); opacity: 0; }
      to   { transform: translateY(0);    opacity: 1; }
    }
    @keyframes pulse { 0%,100% { opacity: 1 } 50% { opacity: 0.45 } }
  `;
  document.head.appendChild(s);
}
