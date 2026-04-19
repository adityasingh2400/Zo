import React, { useEffect, useState } from 'react';
import { useEmpireSocket } from './hooks/useEmpireSocket';
import { TikTokShopOverlay } from './components/TikTokShopOverlay';
import { CostTicker } from './components/CostTicker';
import { RoutingPanel } from './components/RoutingPanel';
import { StartDemoOverlay } from './components/StartDemoOverlay';

/**
 * StageView — the demo's full-screen surface.
 *
 * Loaded at /stage. Renders ONLY:
 *   - TikTokShopOverlay (centered 9:16, fills the vertical viewport)
 *   - CostTicker (top-right of the dark bezel)
 *   - RoutingPanel compact={true} (bottom-right of the dark bezel)
 *
 * No nav, no header, no settings — anything that exists outside the phone-
 * screen frame must serve "the unit-economics story" or it doesn't belong.
 *
 * Hotkeys (per the design doc):
 *   G — Go Live: POST /api/go_live so the backend Director plays a generic
 *       intro clip ("hey everyone, welcome back to the stream") via the
 *       same Tier 1 crossfade machinery the response pipeline uses. Zero
 *       latency, pre-rendered.
 *   R — reset CostTicker (handled inside CostTicker itself).
 *   F — toggle browser fullscreen (saves the operator from hunting for F11).
 *
 * Stage operator setup:
 *   1. Open this view at http://<demo-mac>:5173/stage on the demo MacBook.
 *   2. Press F to enter fullscreen (or F11 — both work).
 *   3. Run scripts/demo_prewarm.sh until it prints PASS.
 *   4. Walk on stage. Press G when you start talking.
 */

const API_BASE = `http://${window.location.hostname}:8000`;

// Below this viewport width the side-bezel layout becomes too cramped
// (cost ticker + routing panel start to overlap the 9:16 phone frame).
// We flip into "stacked" mode where the chrome sits ABOVE the frame instead.
// Stage usage at 1920×1080 always uses side-bezel mode; this is just so
// the dev experience in a narrow IDE pane stays usable.
const WIDE_BEZEL_MIN_PX = 1280;

export default function StageView() {
  const {
    productData, pitchVideoUrl, responseVideo, pendingComments,
    liveStage, routingDecisions, routingStats, wsRef, connected,
    audioResponse, setAudioResponse, pitchAudio, setPitchAudio,
  } = useEmpireSocket();

  // Same audio-end handler as the operator dashboard — clear the slot so
  // the same payload won't auto-replay.
  const handleAudioEnded = (kind) => {
    if (kind === 'pitch') setPitchAudio(null);
    else setAudioResponse(null);
  };

  const [hintVisible, setHintVisible] = useState(true);
  const [goLiveAt, setGoLiveAt] = useState(null);
  const [fullscreen, setFullscreen] = useState(false);
  const [wideBezel, setWideBezel] = useState(() =>
    typeof window !== 'undefined' ? window.innerWidth >= WIDE_BEZEL_MIN_PX : true
  );

  // Track real fullscreen state so the F-key toggle is symmetric (same key
  // exits and enters), and so the hint can hide whenever we're in
  // presentation mode regardless of which keyboard shortcut got us there.
  useEffect(() => {
    function onFs() {
      setFullscreen(!!document.fullscreenElement);
    }
    document.addEventListener('fullscreenchange', onFs);
    return () => document.removeEventListener('fullscreenchange', onFs);
  }, []);

  // Respond to window resize so a developer widening the browser window
  // mid-session sees the layout flip into the projector-style bezel mode
  // without a hard reload.
  useEffect(() => {
    function onResize() {
      setWideBezel(window.innerWidth >= WIDE_BEZEL_MIN_PX);
    }
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  async function fireGoLive() {
    setGoLiveAt(Date.now());
    setHintVisible(false);
    try {
      const r = await fetch(`${API_BASE}/api/go_live`, { method: 'POST' });
      if (!r.ok) {
        console.warn('[stage] /api/go_live failed', r.status);
      }
    } catch (e) {
      // Don't block the stage flow on a network hiccup. The error surface
      // we care about is "no clip played" — the audience reads silence.
      console.warn('[stage] /api/go_live error', e);
    }
  }

  async function toggleFullscreen() {
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
      } else {
        await document.documentElement.requestFullscreen();
      }
    } catch (e) {
      console.warn('[stage] fullscreen toggle failed', e);
    }
  }

  // Hotkey wiring. Filter out events while typing into a text field so
  // operator can still use any debug input without firing the live intro.
  // R is owned by CostTicker — we deliberately don't intercept it here.
  useEffect(() => {
    function onKey(e) {
      const tag = (e.target?.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'textarea') return;
      if (e.key === 'g' || e.key === 'G') {
        e.preventDefault();
        fireGoLive();
      } else if (e.key === 'f' || e.key === 'F') {
        // Don't intercept Cmd+F / Ctrl+F (browser find).
        if (e.metaKey || e.ctrlKey) return;
        e.preventDefault();
        toggleFullscreen();
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  // Auto-hide the hint after 12s so the stage stays clean if the operator
  // forgets to dismiss it before walking on.
  useEffect(() => {
    if (!hintVisible) return;
    const id = setTimeout(() => setHintVisible(false), 12_000);
    return () => clearTimeout(id);
  }, [hintVisible]);

  return (
    <div style={styles.root}>
      {/* Autoplay-unlock — one-tap ceremony BEFORE any audio-first comment
          fires (REVISIONS §3). Banks browser permission for <audio> elements
          created later at WS-message time. */}
      <StartDemoOverlay />

      {/* The phone screen — TikTokShopOverlay handles all the chrome inside
          its own 9:16 area. We just give it the full viewport to center in. */}
      <TikTokShopOverlay
        productData={productData}
        pitchVideoUrl={pitchVideoUrl}
        responseVideo={responseVideo}
        pendingComments={pendingComments}
        liveStage={liveStage}
        wsRef={wsRef}
        audioResponse={audioResponse}
        pitchAudio={pitchAudio}
        onAudioEnded={handleAudioEnded}
      />

      {/* Bezel chrome — lives in the black bars on either side of 9:16
          at projector widths (≥1280px). At narrower viewports it stacks
          along the top edge so it never overlaps the phone frame. */}
      <div style={wideBezel ? styles.bezelTopRight : styles.stackedTopRight}>
        <CostTicker wsRef={wsRef} />
      </div>

      <div style={wideBezel ? styles.bezelBottomRight : styles.stackedRoutingNarrow}>
        <RoutingPanel
          routingDecisions={routingDecisions}
          routingStats={routingStats}
          compact
        />
      </div>

      {/* Stage operator hint — visible until the first Go Live press, or
          12s, whichever comes first. Hidden in stacked mode (no room
          without overlapping the phone frame). */}
      {hintVisible && wideBezel && (
        <div style={styles.bezelBottomLeft}>
          <div style={styles.hint}>
            <div style={styles.hintHeader}>
              <span style={{
                ...styles.hintDot,
                background: connected ? '#22c55e' : '#ef4444',
                boxShadow: connected ? '0 0 8px #22c55e' : '0 0 8px #ef4444',
              }} />
              <span style={styles.hintLabel}>STAGE READY</span>
            </div>
            <div style={styles.hintRow}><kbd style={styles.kbd}>G</kbd> Go Live</div>
            <div style={styles.hintRow}><kbd style={styles.kbd}>R</kbd> Reset cost</div>
            <div style={styles.hintRow}>
              <kbd style={styles.kbd}>F</kbd> Fullscreen {fullscreen ? '(on)' : '(off)'}
            </div>
            <div style={styles.hintFoot}>
              dismisses on first Go Live · auto-hide 12s
            </div>
          </div>
        </div>
      )}

      {/* Brief Go Live ping so the operator gets visual confirmation the
          intro clip request hit the backend. Disappears after 1.4s. */}
      {goLiveAt && Date.now() - goLiveAt < 1400 && (
        <div style={styles.goLivePing}>
          <span style={styles.goLivePingDot} />
          INTRO FIRED
        </div>
      )}
    </div>
  );
}

const styles = {
  root: {
    position: 'fixed', inset: 0,
    background: '#000',
    overflow: 'hidden',
    color: '#fafafa',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  },
  bezelTopRight: {
    position: 'absolute', top: 24, right: 24, zIndex: 50,
    pointerEvents: 'auto',
  },
  bezelBottomRight: {
    position: 'absolute', bottom: 24, right: 24, zIndex: 50,
    pointerEvents: 'auto',
  },
  bezelBottomLeft: {
    position: 'absolute', bottom: 24, left: 24, zIndex: 50,
    pointerEvents: 'auto',
  },
  // Narrow-viewport fallbacks. Both chrome elements stack vertically on
  // the top-right so they never overlap the centered 9:16 phone frame
  // OR each other. Cost ticker on top (most important), routing strip
  // immediately below.
  stackedTopRight: {
    position: 'absolute', top: 12, right: 12, zIndex: 50,
    pointerEvents: 'auto',
  },
  stackedRoutingNarrow: {
    position: 'absolute', top: 116, right: 12, zIndex: 50,
    pointerEvents: 'auto',
  },
  hint: {
    background: 'rgba(9,9,11,0.85)',
    border: '1px solid #27272a',
    borderRadius: 12, padding: '10px 14px',
    backdropFilter: 'blur(8px)',
    boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
    fontSize: 12, color: '#a1a1aa',
    minWidth: 180,
  },
  hintHeader: {
    display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8,
  },
  hintDot: {
    width: 8, height: 8, borderRadius: 4,
  },
  hintLabel: {
    fontSize: 10, fontWeight: 800, letterSpacing: 1.5,
    color: '#fafafa', textTransform: 'uppercase',
  },
  hintRow: {
    display: 'flex', alignItems: 'center', gap: 8,
    paddingTop: 2, paddingBottom: 2,
    color: '#d4d4d8',
  },
  kbd: {
    display: 'inline-block', minWidth: 22, padding: '2px 6px',
    background: '#27272a', border: '1px solid #3f3f46',
    borderRadius: 4, fontSize: 11, fontWeight: 700,
    fontFamily: 'inherit', textAlign: 'center', color: '#fafafa',
  },
  hintFoot: {
    fontSize: 10, color: '#52525b', marginTop: 8,
    fontStyle: 'italic',
  },
  goLivePing: {
    position: 'absolute', top: '50%', left: '50%',
    transform: 'translate(-50%, -50%)', zIndex: 60,
    background: 'rgba(22, 163, 74, 0.92)',
    color: '#fff', padding: '14px 24px',
    borderRadius: 14, fontSize: 16, fontWeight: 900, letterSpacing: 2,
    display: 'flex', alignItems: 'center', gap: 12,
    boxShadow: '0 0 60px rgba(34,197,94,0.6)',
    pointerEvents: 'none',
  },
  goLivePingDot: {
    width: 12, height: 12, borderRadius: 6, background: '#fff',
    animation: 'pulse 1.2s ease-in-out infinite',
  },
};
