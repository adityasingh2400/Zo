import React, { useState } from 'react';
import { useEmpireSocket } from './hooks/useEmpireSocket';
import { LiveStage } from './components/LiveStage';
import { StageBar } from './components/StageBar';
import { ProductPanel } from './components/ProductPanel';
import { AgentLog } from './components/AgentLog';
import { ChatPanel } from './components/ChatPanel';

export default function App() {
  const {
    connected, status, productData, productPhoto, salesScript,
    agentLog, transcript, sendComment,
    pitchVideoUrl, responseVideo, liveStage, pendingComments,
  } = useEmpireSocket();

  const [sellInput, setSellInput] = useState('sell this for $49');
  const [dragging, setDragging] = useState(false);

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
      {dragging && (
        <div style={styles.dropOverlay}>
          <span style={{ fontSize: 64 }}>🎬</span>
          <p style={{ fontSize: 24, fontWeight: 700 }}>Drop video or photo here</p>
        </div>
      )}

      {/* Header */}
      <header style={styles.header}>
        <div style={styles.headerLeft}>
          <h1 style={styles.logo}>EMPIRE</h1>
          <span style={styles.tagline}>AI Commerce Agent Swarm</span>
        </div>
        <div style={styles.headerCenter}>
          <StageBar stage={liveStage} />
        </div>
        <div style={styles.headerRight}>
          <div style={{
            ...styles.connectionDot,
            background: connected ? '#22c55e' : '#ef4444',
            boxShadow: connected ? '0 0 8px #22c55e' : '0 0 8px #ef4444',
          }} />
          <span style={{ color: connected ? '#22c55e' : '#ef4444', fontSize: 12, fontWeight: 700, letterSpacing: 1 }}>
            {connected ? 'CONNECTED' : 'DISCONNECTED'}
          </span>
        </div>
      </header>

      {/* Demo Controls */}
      <div style={styles.controls}>
        <input
          value={sellInput}
          onChange={e => setSellInput(e.target.value)}
          style={styles.sellInput}
          placeholder='e.g. "sell this for $49 targeting young professionals"'
        />
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
          />
          <div style={styles.stageBelow}>
            <AgentLog log={agentLog} />
          </div>
        </div>
        <div style={styles.sideCol}>
          <ProductPanel productData={productData} productPhoto={productPhoto} transcript={transcript} />
          <ChatPanel
            onSendComment={sendComment}
            commentResponse={responseVideo}
            pendingComments={pendingComments}
          />
        </div>
      </div>

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
    display: 'grid', gridTemplateColumns: 'auto 1fr auto', alignItems: 'center', gap: 16, padding: '8px 0',
  },
  headerLeft: { display: 'flex', alignItems: 'baseline', gap: 12 },
  headerCenter: { display: 'flex', justifyContent: 'center' },
  logo: {
    fontSize: 32, fontWeight: 900, letterSpacing: 4, color: '#fafafa',
    background: 'linear-gradient(135deg, #7c3aed, #3b82f6)', WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent', margin: 0,
  },
  tagline: { color: '#52525b', fontSize: 14 },
  headerRight: { display: 'flex', alignItems: 'center', gap: 8 },
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
  stageBelow: { height: 220, minHeight: 220 },
  sideCol: {
    display: 'grid', gridTemplateRows: '1fr 1fr', gap: 12, minHeight: 0, minWidth: 0,
  },
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
