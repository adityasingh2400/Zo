import React, { useState } from 'react';
import { useEmpireSocket } from './hooks/useEmpireSocket';
import { LiveStage } from './components/LiveStage';
import { ProductPanel } from './components/ProductPanel';
import { AgentLog } from './components/AgentLog';
import { ChatPanel } from './components/ChatPanel';
import { VoiceMic } from './components/VoiceMic';
import { RoutingPanel } from './components/RoutingPanel';
import { StartDemoOverlay } from './components/StartDemoOverlay';

export default function App() {
  const {
    connected, productData, productPhoto,
    agentLog, transcript, sendComment,
    pitchVideoUrl, responseVideo, liveStage, pendingComments,
    view3d, transcriptExtract, voiceTranscript,
    routingDecisions, routingStats,
    audioResponse, setAudioResponse, pitchAudio, setPitchAudio,
    wsRef,
  } = useEmpireSocket();

  // When audio playback ends, parent clears the matching slot so a stale
  // payload doesn't auto-replay if the same audio element is reused.
  const handleAudioEnded = (kind) => {
    if (kind === 'pitch') setPitchAudio(null);
    else setAudioResponse(null);
  };

  const [sellInput, setSellInput] = useState('sell this for $49');
  const [dragging, setDragging] = useState(false);
  const [showTelemetry, setShowTelemetry] = useState(false);

  async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('voice_text', sellInput);
    const endpoint = file.type.startsWith('video/')
      ? `http://${window.location.hostname}:8000/api/sell-video`
      : `http://${window.location.hostname}:8000/api/sell`;
    await fetch(endpoint, { method: 'POST', body: formData });
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
  }

  return (
    <div
      style={{ ...styles.app, ...(dragging ? styles.appDragging : {}) }}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
      {/* One-time autoplay-unlock ceremony so audio-first <audio> elements
          created at WS-message time can play without browser policy blocking.
          See REVISIONS §3 in the design doc. */}
      <StartDemoOverlay />

      {dragging && (
        <div style={styles.dropOverlay}>
          <span style={{ fontSize: 64 }}>🎬</span>
          <p style={{ fontSize: 24, fontWeight: 700 }}>Drop video or photo here</p>
        </div>
      )}

      {/* Header */}
      <header style={styles.header}>
        <div style={styles.headerLeft}>
          <h1 style={styles.logo}>Zo</h1>
        </div>
      </header>

      {/* Floating top-right: telemetry button + connection status */}
      <div style={styles.floatingTopRight}>
        <button
          type="button"
          onClick={() => setShowTelemetry(true)}
          style={styles.telemetryButton}
          title="Open routing + agent activity telemetry"
        >
          ◎ Telemetry
          {routingStats?.total > 0 && (
            <span style={styles.telemetryBadge}>{routingStats.total}</span>
          )}
        </button>
        <div style={styles.connectionPill}>
          <div style={{
            ...styles.connectionDot,
            background: connected ? '#22c55e' : '#ef4444',
            boxShadow: connected ? '0 0 8px #22c55e' : '0 0 8px #ef4444',
          }} />
          <span style={{ color: connected ? '#22c55e' : '#ef4444', fontSize: 12, fontWeight: 700, letterSpacing: 1 }}>
            {connected ? 'CONNECTED' : 'DISCONNECTED'}
          </span>
        </div>
      </div>

      {/* Demo Controls */}
      <div style={styles.controls}>
        <input
          value={sellInput}
          onChange={e => setSellInput(e.target.value)}
          style={styles.sellInput}
          placeholder='e.g. "sell this for $49 targeting young professionals"'
        />
        <VoiceMic voiceTranscript={voiceTranscript} wsRef={wsRef} />
        <label style={styles.uploadLabel}>
          🎬 Upload Video
          <input
            type="file" accept="video/*" style={{ display: 'none' }}
            onChange={async (e) => { const f = e.target.files[0]; if (f) await uploadFile(f); }}
          />
        </label>
        <label style={styles.uploadLabel}>
          📷 Photo
          <input
            type="file" accept="image/*" style={{ display: 'none' }}
            onChange={async (e) => { const f = e.target.files[0]; if (f) await uploadFile(f); }}
          />
        </label>
      </div>

      {/* Cinema Layout: big stage on the left, sidebar on the right */}
      <div style={styles.cinemaGrid}>
        <div style={styles.stageCol}>
          <LiveStage
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
        </div>
        <div style={styles.sideCol}>
          <ProductPanel productData={productData} productPhoto={productPhoto} transcript={transcript} view3d={view3d} transcriptExtract={transcriptExtract} wsRef={wsRef} />
          <ChatPanel
            onSendComment={sendComment}
            commentResponse={responseVideo}
            pendingComments={pendingComments}
          />
        </div>
      </div>

      {/* Telemetry overlay — click "◎ Telemetry" to open */}
      {showTelemetry && (
        <div style={styles.telemetryOverlay} onClick={(e) => e.target === e.currentTarget && setShowTelemetry(false)}>
          <div style={styles.telemetryPanel}>
            <div style={styles.telemetryHeader}>
              <h2 style={styles.telemetryTitle}>Telemetry</h2>
              <button
                type="button"
                onClick={() => setShowTelemetry(false)}
                style={styles.telemetryClose}
                title="Close (Esc)"
              >
                ✕
              </button>
            </div>
            <div style={styles.telemetryBody}>
              <div style={styles.telemetryCol}>
                <RoutingPanel routingDecisions={routingDecisions} routingStats={routingStats} />
              </div>
              <div style={styles.telemetryCol}>
                <AgentLog log={agentLog} />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer style={styles.footer}>
        <span>Gemma 4 on Cactus (on-device) • Claude on AWS Bedrock • ElevenLabs • Wav2Lip + LatentSync on RunPod 5090</span>
      </footer>
    </div>
  );
}

const styles = {
  app: {
    minHeight: '100vh', display: 'flex', flexDirection: 'column',
    padding: 16, gap: 12, maxWidth: 1600, margin: '0 auto',
  },
  header: {
    display: 'flex', alignItems: 'center', padding: '8px 0',
  },
  headerLeft: { display: 'flex', alignItems: 'baseline', gap: 12 },
  logo: {
    fontSize: 32, fontWeight: 900, letterSpacing: 4, color: '#fafafa',
    background: 'linear-gradient(135deg, #7c3aed, #3b82f6)', WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent', margin: 0,
  },
  floatingTopRight: {
    position: 'fixed', top: 16, right: 16, zIndex: 50,
    display: 'flex', alignItems: 'center', gap: 10,
  },
  connectionPill: {
    display: 'flex', alignItems: 'center', gap: 8,
    background: 'rgba(15,15,18,0.8)', backdropFilter: 'blur(8px)',
    border: '1px solid #27272a', borderRadius: 999,
    padding: '6px 12px',
  },
  connectionDot: { width: 8, height: 8, borderRadius: 4 },
  controls: {
    display: 'flex', gap: 8, padding: '0 0 8px',
  },
  sellInput: {
    flex: 1, background: '#18181b', border: '1px solid #3f3f46', borderRadius: 8,
    padding: '10px 14px', color: '#fafafa', fontSize: 14, outline: 'none',
  },
  uploadLabel: {
    background: '#27272a', color: '#a1a1aa', border: '1px solid #3f3f46',
    borderRadius: 8, padding: '10px 16px', fontSize: 14, cursor: 'pointer',
    fontWeight: 600,
  },
  cinemaGrid: {
    flex: 1, display: 'grid',
    gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 1fr)',
    gap: 12, minHeight: 0,
  },
  stageCol: { display: 'flex', flexDirection: 'column', gap: 12, minHeight: 0, minWidth: 0 },
  sideCol: {
    // Two rows: ProductPanel, ChatPanel. Telemetry (routing + agent log)
    // lives in a separate overlay opened via the header button.
    display: 'grid', gridTemplateRows: 'minmax(0, 1fr) minmax(0, 1.2fr)',
    gap: 12, minHeight: 0, minWidth: 0,
  },
  telemetryButton: {
    display: 'inline-flex', alignItems: 'center', gap: 6,
    background: '#18181b', color: '#a1a1aa',
    border: '1px solid #3f3f46', borderRadius: 8,
    padding: '6px 12px', fontSize: 12, fontWeight: 700,
    letterSpacing: 1, cursor: 'pointer',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  },
  telemetryBadge: {
    background: '#22c55e', color: '#09090b', borderRadius: 999,
    padding: '1px 6px', fontSize: 10, fontWeight: 800,
    minWidth: 18, textAlign: 'center',
  },
  telemetryOverlay: {
    position: 'fixed', inset: 0, zIndex: 500,
    background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(8px)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    padding: 24,
  },
  telemetryPanel: {
    width: '100%', maxWidth: 1400, height: '90vh',
    background: '#09090b', border: '1px solid #27272a',
    borderRadius: 14, display: 'flex', flexDirection: 'column',
    overflow: 'hidden',
  },
  telemetryHeader: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '14px 20px', borderBottom: '1px solid #27272a',
  },
  telemetryTitle: {
    margin: 0, fontSize: 18, fontWeight: 800, letterSpacing: 2,
    textTransform: 'uppercase', color: '#fafafa',
  },
  telemetryClose: {
    background: 'transparent', color: '#a1a1aa',
    border: '1px solid #3f3f46', borderRadius: 8,
    width: 32, height: 32, fontSize: 14,
    cursor: 'pointer',
  },
  telemetryBody: {
    flex: 1, display: 'grid',
    gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)',
    gap: 14, padding: 14, minHeight: 0,
  },
  telemetryCol: { minHeight: 0, minWidth: 0, display: 'flex', flexDirection: 'column' },
  footer: {
    textAlign: 'center', padding: '12px 0', color: '#3f3f46', fontSize: 12,
  },
  appDragging: {
    outline: '3px dashed #7c3aed',
    outlineOffset: -3,
  },
  dropOverlay: {
    position: 'fixed', inset: 0, background: 'rgba(124,58,237,0.15)',
    display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
    zIndex: 999, color: '#fff', gap: 12, backdropFilter: 'blur(4px)',
  },
};
