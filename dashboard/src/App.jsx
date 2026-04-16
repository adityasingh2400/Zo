import React, { useState } from 'react';
import { useEmpireSocket } from './hooks/useEmpireSocket';
import { AvatarPanel } from './components/AvatarPanel';
import { ProductPanel } from './components/ProductPanel';
import { AgentLog } from './components/AgentLog';
import { ChatPanel } from './components/ChatPanel';

export default function App() {
  const {
    connected, status, productData, productPhoto, salesScript,
    agentLog, latestAudio, commentResponse, sendComment, sendSell,
  } = useEmpireSocket();

  const [sellInput, setSellInput] = useState('sell this for $49');

  return (
    <div style={styles.app}>
      {/* Header */}
      <header style={styles.header}>
        <div style={styles.headerLeft}>
          <h1 style={styles.logo}>EMPIRE</h1>
          <span style={styles.tagline}>AI Commerce Agent Swarm</span>
        </div>
        <div style={styles.headerRight}>
          <div style={{
            ...styles.connectionDot,
            background: connected ? '#22c55e' : '#ef4444',
            boxShadow: connected ? '0 0 8px #22c55e' : '0 0 8px #ef4444',
          }} />
          <span style={{ color: connected ? '#22c55e' : '#ef4444', fontSize: 13, fontWeight: 600 }}>
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
        <button onClick={() => sendSell(sellInput)} style={styles.sellBtn}>
          🚀 SELL THIS
        </button>
        <label style={styles.uploadLabel}>
          📷 Upload Photo
          <input
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={async (e) => {
              const file = e.target.files[0];
              if (!file) return;
              const formData = new FormData();
              formData.append('file', file);
              formData.append('voice_text', sellInput);
              await fetch(`http://${window.location.hostname}:8000/api/sell`, {
                method: 'POST',
                body: formData,
              });
            }}
          />
        </label>
      </div>

      {/* Main Grid */}
      <div style={styles.grid}>
        <div style={styles.gridLeft}>
          <AvatarPanel status={status} latestAudio={latestAudio} salesScript={salesScript} />
        </div>
        <div style={styles.gridRight}>
          <ProductPanel productData={productData} productPhoto={productPhoto} />
        </div>
        <div style={styles.gridBottomLeft}>
          <AgentLog log={agentLog} />
        </div>
        <div style={styles.gridBottomRight}>
          <ChatPanel onSendComment={sendComment} commentResponse={commentResponse} />
        </div>
      </div>

      {/* Footer */}
      <footer style={styles.footer}>
        <span>Gemma 4 on Cactus (on-device) • Claude on AWS Bedrock • ElevenLabs • LiveTalking on RunPod</span>
      </footer>
    </div>
  );
}

const styles = {
  app: {
    minHeight: '100vh', display: 'flex', flexDirection: 'column',
    padding: 16, gap: 12, maxWidth: 1400, margin: '0 auto',
  },
  header: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 0',
  },
  headerLeft: { display: 'flex', alignItems: 'baseline', gap: 12 },
  logo: {
    fontSize: 32, fontWeight: 900, letterSpacing: 4, color: '#fafafa',
    background: 'linear-gradient(135deg, #7c3aed, #3b82f6)', WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
  },
  tagline: { color: '#52525b', fontSize: 14 },
  headerRight: { display: 'flex', alignItems: 'center', gap: 8 },
  connectionDot: { width: 8, height: 8, borderRadius: 4 },
  controls: {
    display: 'flex', gap: 8, padding: '8px 0',
  },
  sellInput: {
    flex: 1, background: '#18181b', border: '1px solid #3f3f46', borderRadius: 8,
    padding: '10px 14px', color: '#fafafa', fontSize: 14, outline: 'none',
  },
  sellBtn: {
    background: 'linear-gradient(135deg, #7c3aed, #3b82f6)', color: '#fff',
    border: 'none', borderRadius: 8, padding: '10px 24px', fontWeight: 800,
    fontSize: 15, cursor: 'pointer', letterSpacing: 1,
  },
  uploadLabel: {
    background: '#27272a', color: '#a1a1aa', border: '1px solid #3f3f46',
    borderRadius: 8, padding: '10px 16px', fontSize: 14, cursor: 'pointer',
    fontWeight: 600,
  },
  grid: {
    flex: 1, display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gridTemplateRows: '1fr 1fr',
    gap: 12, minHeight: 0,
  },
  gridLeft: { minHeight: 300 },
  gridRight: { minHeight: 300 },
  gridBottomLeft: { minHeight: 250 },
  gridBottomRight: { minHeight: 250 },
  footer: {
    textAlign: 'center', padding: '12px 0', color: '#3f3f46', fontSize: 12,
  },
};
