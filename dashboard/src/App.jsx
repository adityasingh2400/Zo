import React, { useState, useEffect } from 'react';
import { useEmpireSocket } from './hooks/useEmpireSocket';
import { TikTokShopOverlay } from './components/TikTokShopOverlay';
import { StartDemoOverlay } from './components/StartDemoOverlay';
import { ChatPanel } from './components/ChatPanel';
import { PhoneQRPanel } from './components/PhoneQRPanel';
import { EventLogHUD } from './components/EventLogHUD';
import { dlog } from './lib/dlog';
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
    commentResponse, sendComment,
    routingDecision,
    phoneUpload,
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

  // dlog hasUploaded transitions so the EventLog reflects the same
  // mount/unmount story you'd see by inspecting React DevTools.
  useEffect(() => {
    dlog('app', hasUploaded ? 'live_stage_mounted' : 'live_stage_unmounted',
         { hasUploaded });
  }, [hasUploaded]);

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
          <PhoneQRPanel
            phoneUpload={phoneUpload}
            connected={connected}
            onUploadComplete={() => setHasUploaded(true)}
          />
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

      {/* Gemma decision HUD — top-left, directly under the clip HUD.
          Mirrors the dispatcher's view of an incoming comment in real
          time:
            • the moment classify_comment_gemma returns (~300 ms after
              POST), `routing_decision` lands and we render the comment
              text + intent badge + Gemma's draft response.
            • when the Wav2Lip render lands (~5-8 s later),
              `comment_response_video` lands and we overlay the
              substrate filename + class/tts/lipsync/total ms breakdown.
          Color-coded badges match the bridge bucket (green=compliment,
          orange=objection, blue=question, gray=spam, purple=neutral). */}
      <GemmaDecisionHud
        routingDecision={routingDecision}
        commentResponse={commentResponse}
      />
      

      {/* Dev comment-tester panel — bottom-left floating chat. Always
          visible so we can test the bridge+wav2lip dispatch path even
          before an upload (Maya responds against whatever active
          product_data is loaded — leather_wallet by default). Wired to
          sendComment from useEmpireSocket which posts to /api/comment,
          and reads commentResponse so the panel renders the AI reply
          with a latency chip. Same UX as the operator's existing chat
          surface in /stage; this is the test surface. */}
      <div style={styles.chatDock}>
        <ChatPanel
          onSendComment={sendComment}
          commentResponse={commentResponse}
          pendingComments={pendingComments}
        />
      </div>

      {/* Live event-log HUD — bottom-right floating panel that shows
          every dlog() call (WS connect/close, phone status changes,
          play_clip events received, listener attach/detach, tier1
          driver state). Critical when iterating on phone-upload bugs:
          a 5-second glance at this panel tells you exactly which step
          in the chain dropped instead of having to triangulate from
          the avatar's frozen frame + browser console. Toggle hidden
          via the ✕ button — subscription stays alive in the background
          so re-opening shows the live feed instantly. */}
      <EventLogHUD />

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

// Color tokens per Gemma intent bucket. Matches the bridge clip
// directories under /workspace/bridges/<intent>/ on the pod (and
// phase0/assets/bridges/<intent>/ locally). Picked for high contrast
// against the dark stage so the badge reads at-a-glance — green for
// "yes!", orange for "hmm not sure", blue for "thinking question",
// gray for blocked spam.
const INTENT_PALETTE = {
  compliment: { bg: 'rgba(34,197,94,0.92)',  fg: '#052e16', label: 'COMPLIMENT' },
  objection:  { bg: 'rgba(249,115,22,0.92)', fg: '#431407', label: 'OBJECTION' },
  question:   { bg: 'rgba(59,130,246,0.92)', fg: '#172554', label: 'QUESTION' },
  spam:       { bg: 'rgba(113,113,122,0.92)',fg: '#0a0a0b', label: 'SPAM' },
  neutral:    { bg: 'rgba(168,85,247,0.92)', fg: '#2e1065', label: 'NEUTRAL' },
  unknown:    { bg: 'rgba(82,82,91,0.92)',   fg: '#fafafa', label: '?' },
};

// Compact panel showing the current Gemma + router decision. Two-phase
// hydration so the operator never sees blank UI:
//   1. routing_decision arrives ~300 ms after the comment POST. We have
//      `comment`, `intent`, `intent_hint`, `tool`, `draft_response`,
//      `classify_ms`. Render that immediately with a "rendering…" hint.
//   2. comment_response_video arrives ~5-8 s later when Wav2Lip is done.
//      We pick up `substrate`, full `total_ms`, and per-stage timings.
// Mismatched comments (response is for a stale comment) are ignored —
// keyed on comment string equality.
function GemmaDecisionHud({ routingDecision, commentResponse }) {
  // Prefer the latest commentResponse fields when they cover the SAME
  // comment as the routing event. Otherwise show whichever is newer
  // (the routing decision for an in-flight render, or the response for
  // the previous comment if no fresh routing has fired yet).
  const sameComment =
    routingDecision && commentResponse &&
    routingDecision.comment === commentResponse.comment;
  const haveAny = !!(routingDecision || commentResponse);
  if (!haveAny) {
    return (
      <div style={styles.gemmaHud}>
        <div style={{ ...styles.gemmaRow, opacity: 0.45 }}>
          <span style={styles.gemmaTier}>GEMMA</span>
          <span style={styles.gemmaIdle}>idle — waiting for comment</span>
        </div>
      </div>
    );
  }

  // Choose the source of truth.
  const fromResp = !!commentResponse;
  const ds = commentResponse || routingDecision;
  const comment = ds.comment || '';
  // Intent priority: explicit `intent` field > router's `intent_hint` >
  // 'unknown'. Spam suppresses the hint (the dispatcher blocks before
  // picking a substrate).
  const isSpam = ds.tool === 'block_comment' || ds.intent === 'spam';
  const intentKey = isSpam
    ? 'spam'
    : (ds.intent_hint || ds.intent || 'unknown').toLowerCase();
  const palette = INTENT_PALETTE[intentKey] || INTENT_PALETTE.unknown;

  // Substrate filename — appears in TWO places:
  //   1. routingDecision.substrate — set by comment_substrate_picked
  //      WS event (fires within ~300 ms of POST, before render starts)
  //   2. commentResponse.substrate — set by comment_response_video WS
  //      event (fires 5-15 s later when render completes)
  // Prefer the freshest one; fall back to the early signal so the HUD
  // can show the picked clip during the render window instead of just
  // a "● rendering" pulse.
  const substrateRaw =
    commentResponse?.substrate ??
    routingDecision?.substrate ??
    null;
  const substrate = substrateRaw ? String(substrateRaw).split('/').pop() : null;

  // Timing chips.
  const classMs = commentResponse?.class_ms ?? routingDecision?.classify_ms;
  const ttsMs = commentResponse?.tts_ms;
  const lipMs = commentResponse?.lipsync_ms;
  const totalMs = commentResponse?.total_ms;

  // Draft response (Gemma's 1-sentence) — surfaces what the avatar will
  // SAY before TTS even runs. Useful debug for "wait, why did Maya say
  // that?" moments. Truncate to keep the HUD compact.
  const draft = routingDecision?.draft_response;

  return (
    <div style={styles.gemmaHud}>
      <div style={styles.gemmaRow}>
        <span style={styles.gemmaTier}>GEMMA</span>
        <span style={{
          ...styles.gemmaBadge,
          background: palette.bg,
          color: palette.fg,
        }}>
          {palette.label}
        </span>
        {!fromResp && (
          <span style={styles.gemmaPending}>● rendering</span>
        )}
        {fromResp && totalMs != null && (
          <span style={styles.gemmaTotal}>⚡ {(totalMs / 1000).toFixed(1)}s</span>
        )}
      </div>
      {comment && (
        <div style={styles.gemmaComment}>
          <span style={styles.gemmaQuote}>"</span>
          {comment.length > 60 ? comment.slice(0, 60) + '…' : comment}
          <span style={styles.gemmaQuote}>"</span>
        </div>
      )}
      {substrate && (
        <div style={styles.gemmaMeta}>
          <span style={styles.gemmaSubstrate}>📼 {substrate}</span>
        </div>
      )}
      {draft && (
        <div style={styles.gemmaMeta}>
          <span style={styles.gemmaDraft}>
            draft → "{draft.length > 80 ? draft.slice(0, 80) + '…' : draft}"
          </span>
        </div>
      )}
      {(classMs != null || ttsMs != null || lipMs != null) && (
        <div style={styles.gemmaTimings}>
          {classMs != null && <span>class {classMs}ms</span>}
          {ttsMs != null && <span>· tts {ttsMs}ms</span>}
          {lipMs != null && <span>· lip {lipMs}ms</span>}
        </div>
      )}
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
  // Bottom-left floating chat dock — invisible container. The
  // ChatPanel's children (message bubbles, preset chips, input pill)
  // each carry their own translucent backdrop; the dock itself has NO
  // background, border, or shadow so they read as floating directly
  // over the stage video à la TikTok Live. Sized to give the chips +
  // input enough horizontal room without crowding the centered avatar.
  chatDock: {
    position: 'fixed', bottom: 14, left: 14, zIndex: 70,
    width: 340, height: 460, maxHeight: '70vh',
    background: 'transparent',
    border: 'none',
    boxShadow: 'none',
    overflow: 'visible',
    display: 'flex',
  },
  // Gemma decision HUD — sits just under the clip HUD in the same
  // top-left stack. Slightly wider than the clip rows because it
  // surfaces the comment text + the substrate filename, both of which
  // need horizontal room. Always rendered (even when empty) so its
  // position never jumps — the empty state is just dimmed.
  gemmaHud: {
    position: 'fixed', top: 70, left: 14, zIndex: 80,
    display: 'flex', flexDirection: 'column', gap: 4,
    background: 'rgba(15,15,18,0.88)',
    backdropFilter: 'blur(10px)',
    border: '1px solid #27272a',
    borderRadius: 8, padding: '6px 8px',
    minWidth: 320, maxWidth: 380,
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    fontSize: 10, lineHeight: 1.3,
    pointerEvents: 'none',
  },
  gemmaRow: {
    display: 'flex', alignItems: 'center', gap: 8,
  },
  gemmaTier: {
    fontWeight: 800, color: '#a1a1aa', letterSpacing: 1,
    minWidth: 38,
  },
  gemmaBadge: {
    fontWeight: 900, fontSize: 10, letterSpacing: 1.2,
    padding: '2px 6px', borderRadius: 4,
  },
  gemmaPending: {
    color: '#fbbf24', fontWeight: 700, fontSize: 9,
    letterSpacing: 0.5,
    animation: 'pulse 1.4s ease-in-out infinite',
  },
  gemmaTotal: {
    color: '#22c55e', fontWeight: 700, fontSize: 10,
    marginLeft: 'auto',
  },
  gemmaIdle: {
    color: '#52525b', fontStyle: 'italic',
  },
  gemmaComment: {
    color: '#fafafa', fontSize: 11, fontStyle: 'italic',
    paddingLeft: 4,
    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
  },
  gemmaQuote: {
    color: '#52525b', fontWeight: 900,
  },
  gemmaMeta: {
    display: 'flex', gap: 6, alignItems: 'center',
    paddingLeft: 4,
    color: '#a1a1aa',
  },
  gemmaSubstrate: {
    fontWeight: 700,
    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
  },
  gemmaDraft: {
    color: '#71717a', fontSize: 9, fontStyle: 'italic',
    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
  },
  gemmaTimings: {
    display: 'flex', gap: 4,
    color: '#71717a', fontSize: 9,
    paddingLeft: 4,
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
