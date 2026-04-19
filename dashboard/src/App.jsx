import React, { useState } from 'react';
import { useEmpireSocket } from './hooks/useEmpireSocket';
import { TikTokShopOverlay } from './components/TikTokShopOverlay';
import { StartDemoOverlay } from './components/StartDemoOverlay';
import { LanguagePicker } from './components/LanguagePicker';

/**
 * App — operator stage at /
 *
 * Single-purpose UI: drop a product video anywhere on the page, the backend
 * end-to-end pipeline takes over (Gemma analyze + script → ElevenLabs TTS →
 * Wav2Lip render → pitch live → comment routing). Zero buttons, no operator
 * controls, no telemetry chrome.
 *
 * Layout is identical to /stage's TikTokShopOverlay with minimalChrome=true:
 *   - Avatar fills the centered 9:16 frame
 *   - Top-right floating glass-bubble showcase: Spin3D (live rotation) on
 *     top, HeroSlideshow (auto-cycling hero shots) underneath
 *   - Bottom-left audience comment rail (live WS-fed bubbles)
 *   - Bottom-right BUY card (only when productData is loaded)
 *
 * Why merge / and /stage layouts: the operator wanted ONE clean stage to
 * test the full pipeline against, not two. /stage stays as-is for the live
 * demo (extra hotkeys + StageView's bezel chrome); / is the dev surface
 * with the same audience-facing visual but drag-drop as the entry point.
 *
 * Anything previously here (ProductSelector, sell input, Hold-to-speak,
 * Upload Video / Photo buttons, Telemetry overlay, Control Room toggle,
 * Cinema grid, ProductPanel sidebar, AgentLog, RoutingPanel, BrainPanel,
 * etc.) was operator chrome added during merges. All those components
 * still exist on disk — to bring any back, import + render where needed.
 */
export default function App() {
  const {
    productData, pitchVideoUrl, responseVideo, pendingComments,
    liveStage, wsRef, connected,
    audioResponse, setAudioResponse, pitchAudio, setPitchAudio,
    view3d,
    activeClips,
    activeLanguage, setActiveLanguage,
  } = useEmpireSocket();

  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  // Session-level "has the operator dropped anything in THIS browser tab".
  // Backend state can carry leftovers from earlier sessions (a previous
  // upload's product_data + view3d will still be in pipeline_state until
  // a fresh pipeline run overwrites them). Gating the entire avatar UI on
  // this session-local flag means a fresh page load always shows a clean
  // empty stage even if the backend has stale carousel data sitting around.
  // Flips true the moment the operator drops a file; never resets within
  // a tab session (refreshing the tab is the only way back to empty).
  const [hasUploaded, setHasUploaded] = useState(false);

  const handleAudioEnded = (kind) => {
    if (kind === 'pitch') setPitchAudio(null);
    else setAudioResponse(null);
  };

  // Drop-anywhere upload. voice_text defaults to "sell this" — Gemma
  // doesn't actually use the literal string for anything dispositive,
  // it's just a hint that this is a sell-style request. The interesting
  // signal comes from the video itself (frame analysis + audio
  // transcription via the intake phase). For a richer hint, the
  // operator can speak into the video clip itself.
  async function uploadFile(file) {
    setUploading(true);
    // Flip the session flag synchronously so the avatar mounts the moment
    // the operator commits to dropping — no waiting for the POST to land.
    // The backend pipeline auto-broadcasts state as it progresses, so the
    // newly-mounted LiveStage starts in idle and crossfades to the paper
    // clip the instant Director.play_processing fires (~50-300ms after
    // the POST is received).
    setHasUploaded(true);
    try {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('voice_text', 'sell this');
      const endpoint = file.type.startsWith('video/')
        ? `http://${window.location.hostname}:8000/api/sell-video`
        : `http://${window.location.hostname}:8000/api/sell`;
      await fetch(endpoint, { method: 'POST', body: fd });
    } catch (e) {
      console.warn('[app] upload failed', e);
    } finally {
      // Clear the dropping state after a beat so the visual ack lingers
      // briefly even if the POST returns instantly.
      setTimeout(() => setUploading(false), 700);
    }
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
  }

  return (
    <div
      style={styles.root}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
      {/* Autoplay-unlock — one-tap ceremony BEFORE any audio-first comment
          fires. Banks browser permission for <audio> elements created
          later at WS-message time. Same component /stage uses. */}
      <StartDemoOverlay />

      {/* Pre-upload state: clean black canvas with a single drop hint at
          the center. The avatar, Spin3D bubble, slideshow, chat rail —
          NOTHING renders until the operator commits to a drop. This is
          deliberate: backend pipeline_state can carry stale view3d /
          product_data from earlier sessions, and showing any of that
          before a real upload reads as "the demo is on" when it isn't.
          Hint disappears the moment hasUploaded flips. */}
      {!hasUploaded && (
        <div style={styles.emptyHint}>
          {/* Step 1: pick language. Big tappable tiles so the operator
              knows ahead of the drop which language the avatar will
              speak. Backend reads pipeline_state["active_language"] at
              pitch-time, so picking AFTER drop is fine too — but we
              surface it first so the demo flow is "pick → drop → speak". */}
          <LanguagePicker
            activeLanguage={activeLanguage}
            onChange={setActiveLanguage}
          />
          <span style={styles.emptyHintIcon}>📦</span>
          <p style={styles.emptyHintLabel}>Drop a product video to start</p>
          <p style={styles.emptyHintSub}>
            Gemma analyzes · ElevenLabs voices · Wav2Lip lip-syncs · live
          </p>
        </div>
      )}

      {/* The whole stage. minimalChrome=true strips the LIVE pill, host
          info, follow button, viewer count, hearts column, etc. — what's
          left is the avatar + audience chat rail + product showcase + BUY
          card. Same layout as /stage so a/b'ing the two surfaces is just
          a hotkey switch. Only mounts AFTER the operator's first drop
          so a fresh tab never inherits stale backend state visually. */}
      {hasUploaded && (
        <TikTokShopOverlay
          productData={productData}
          pitchVideoUrl={pitchVideoUrl}
          responseVideo={responseVideo}
          pendingComments={pendingComments}
          liveStage={liveStage}
          wsRef={wsRef}
          connected={connected}
          audioResponse={audioResponse}
          pitchAudio={pitchAudio}
          onAudioEnded={handleAudioEnded}
          view3d={view3d}
          minimalChrome={true}
        />
      )}

      {/* Drag-over visual feedback. Covers the whole viewport so the
          operator gets unmissable "yes I see you dropping" affordance.
          Matches the green/purple of the existing demo palette. */}
      {dragging && (
        <div style={styles.dropOverlay}>
          <span style={{ fontSize: 72 }}>🎬</span>
          <p style={styles.dropOverlayLabel}>Drop video to go live</p>
          <p style={styles.dropOverlaySub}>
            full pipeline runs end-to-end · no clicks needed
          </p>
        </div>
      )}

      {/* Brief upload-ack pulse — stays on screen ~700ms after a successful
          POST so the operator knows the file landed before the visible
          processing.mp4 paper-clip starts (~50ms emit + intake). */}
      {uploading && !dragging && (
        <div style={styles.uploadPing}>
          <span style={styles.uploadPingDot} />
          UPLOADED · ANALYZING
        </div>
      )}

      {/* Post-upload language chip — stays reachable so the operator can
          flip languages mid-stream (e.g. switch from English pitch to a
          Spanish Q&A response if a viewer comments in Spanish). Mounted
          outside the 9:16 phone silhouette so it doesn't clutter the
          audience-facing surface. Hidden pre-upload because the full
          picker is already centered in the empty-state hint. */}
      {hasUploaded && (
        <div style={styles.langChipSlot}>
          <LanguagePicker
            activeLanguage={activeLanguage}
            onChange={setActiveLanguage}
            compact
          />
        </div>
      )}

      {/* Debug HUD — top-left fixed pill showing the active Director clip
          per layer. Always mounted (even pre-upload) so we can identify
          idle-rotation clips in real time and catch any bad pool entries
          (e.g., the misc_glance_aside_speaking.mp4 silent-mouthing bug
          we just fixed). Tier 0 = always-on idle background, Tier 1 =
          one-shot interjections / pitch / processing bridge. The intent
          name lines up with the avatar_director.py library entries. */}
      <div style={styles.clipHud}>
        <ClipHudRow label="T0" clip={activeClips?.tier0} />
        <ClipHudRow label="T1" clip={activeClips?.tier1} />
      </div>

      {/* Tiny connection indicator — bottom-right corner. Only loud when
          DISCONNECTED so the operator knows when to refresh. CONNECTED
          state stays whisper-quiet (no green spam during a stable run). */}
      <div
        style={{
          ...styles.connectionPill,
          ...(connected ? styles.connectionPillIdle : styles.connectionPillAlert),
        }}
      >
        <span
          style={{
            ...styles.connectionDot,
            background: connected ? '#22c55e' : '#ef4444',
            boxShadow: connected ? 'none' : '0 0 8px #ef4444',
          }}
        />
        {connected ? 'live' : 'DISCONNECTED'}
      </div>
    </div>
  );
}

// Single row of the debug clip HUD. Renders the layer label (T0/T1),
// the intent name, and the mp4 basename so we can immediately identify
// which idle/interjection/bridge clip the Director just emitted. Empty
// (dimmed) when no clip has played on that layer yet this session.
function ClipHudRow({ label, clip }) {
  const isActive = !!clip;
  const filename = clip?.url ? clip.url.split('/').pop() : null;
  // Color-code the row when the muted flag and intent semantically
  // disagree — speaking-named clips ("_speaking") that are emitted
  // muted are the exact bug class we're hunting. Loud red when caught.
  const isSilentSpeak = isActive && filename?.includes('_speaking') && clip.muted;
  return (
    <div style={{
      ...styles.clipHudRow,
      ...(isSilentSpeak ? styles.clipHudRowAlert : null),
      opacity: isActive ? 1 : 0.35,
    }}>
      <span style={styles.clipHudTier}>{label}</span>
      <span style={styles.clipHudIntent}>{clip?.intent || '—'}</span>
      <span style={styles.clipHudFile}>{filename || 'idle'}</span>
      {clip?.muted && <span style={styles.clipHudMuted}>MUTED</span>}
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
  // Debug HUD — top-left, low chrome, monospace so filenames stay
  // readable. Always visible regardless of hasUploaded.
  clipHud: {
    position: 'fixed', top: 14, left: 14, zIndex: 80,
    display: 'flex', flexDirection: 'column', gap: 4,
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    pointerEvents: 'none',
  },
  clipHudRow: {
    display: 'flex', alignItems: 'center', gap: 8,
    background: 'rgba(15,15,18,0.8)',
    backdropFilter: 'blur(10px)',
    border: '1px solid #27272a',
    borderRadius: 6, padding: '4px 8px',
    fontSize: 10, lineHeight: 1.2,
    minWidth: 280,
  },
  clipHudRowAlert: {
    border: '1px solid #ef4444',
    background: 'rgba(127,29,29,0.65)',
    boxShadow: '0 0 12px rgba(239,68,68,0.5)',
  },
  clipHudTier: {
    fontWeight: 800, color: '#a1a1aa', letterSpacing: 1,
    minWidth: 18,
  },
  clipHudIntent: {
    fontWeight: 700, color: '#fafafa',
    minWidth: 130,
  },
  clipHudFile: {
    color: '#71717a', flex: 1,
    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
  },
  clipHudMuted: {
    fontWeight: 800, color: '#fbbf24', fontSize: 9,
    background: 'rgba(146,64,14,0.4)',
    padding: '1px 4px', borderRadius: 3, letterSpacing: 1,
  },
  // Pre-upload state — centered drop affordance over a black canvas.
  // Intentionally minimal: one icon, one prompt line, one tiny tech-stack
  // sub-line. No buttons (drag-drop is the only entry point), no logo,
  // no nav. Disappears the instant the operator drops a file.
  emptyHint: {
    position: 'absolute', inset: 0, zIndex: 5,
    display: 'flex', flexDirection: 'column',
    alignItems: 'center', justifyContent: 'center',
    gap: 14, color: '#52525b',
    pointerEvents: 'none',
  },
  emptyHintIcon: {
    fontSize: 64, opacity: 0.7,
  },
  emptyHintLabel: {
    fontSize: 18, fontWeight: 700, letterSpacing: 1,
    color: '#a1a1aa', margin: 0,
    fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif',
  },
  emptyHintSub: {
    fontSize: 11, fontWeight: 600, letterSpacing: 1.2,
    color: '#3f3f46', margin: 0,
    textTransform: 'uppercase',
  },
  dropOverlay: {
    position: 'fixed', inset: 0, zIndex: 999,
    background: 'rgba(124,58,237,0.22)',
    backdropFilter: 'blur(10px)',
    display: 'flex', flexDirection: 'column',
    alignItems: 'center', justifyContent: 'center',
    gap: 12, color: '#fff',
    border: '4px dashed rgba(255,255,255,0.5)',
    pointerEvents: 'none',
  },
  dropOverlayLabel: {
    fontSize: 28, fontWeight: 900, letterSpacing: 2,
    margin: 0,
  },
  dropOverlaySub: {
    fontSize: 13, fontWeight: 600, letterSpacing: 1,
    color: 'rgba(255,255,255,0.8)',
    margin: 0,
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  },
  uploadPing: {
    position: 'fixed', top: 18, left: '50%', transform: 'translateX(-50%)',
    zIndex: 100,
    background: 'rgba(22,163,74,0.92)', color: '#fff',
    padding: '8px 16px', borderRadius: 999,
    fontSize: 12, fontWeight: 900, letterSpacing: 2,
    display: 'flex', alignItems: 'center', gap: 10,
    boxShadow: '0 4px 20px rgba(22,163,74,0.4)',
    backdropFilter: 'blur(8px)',
  },
  // Top-right corner slot for the compact LanguagePicker chip. zIndex
  // sits above the avatar's TikTok overlay (which uses zIndex up to 60
  // for the LIVE pill / chat rail) so the popout grid isn't clipped by
  // any 9:16-internal layer when the operator clicks to expand mid-stream.
  langChipSlot: {
    position: 'fixed', top: 14, right: 14, zIndex: 110,
  },
  uploadPingDot: {
    width: 8, height: 8, borderRadius: 4, background: '#fff',
    animation: 'pulse 1s ease-in-out infinite',
  },
  connectionPill: {
    position: 'fixed', bottom: 14, right: 14, zIndex: 60,
    display: 'flex', alignItems: 'center', gap: 6,
    background: 'rgba(15,15,18,0.7)',
    backdropFilter: 'blur(8px)',
    borderRadius: 999, padding: '4px 10px',
    fontSize: 10, fontWeight: 700, letterSpacing: 1,
    pointerEvents: 'none',
  },
  connectionPillIdle: {
    border: '1px solid #27272a', color: '#71717a',
    textTransform: 'lowercase',
  },
  connectionPillAlert: {
    border: '1px solid rgba(239,68,68,0.6)',
    color: '#ef4444',
    boxShadow: '0 0 14px rgba(239,68,68,0.3)',
    animation: 'pulse 1.6s ease-in-out infinite',
  },
  connectionDot: { width: 6, height: 6, borderRadius: 3 },
};
