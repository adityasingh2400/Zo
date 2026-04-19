import React from 'react';

/**
 * MetricsStrip — 6-cell top strip matching the empire-mac.jsx hero row.
 *
 * Real values come from WS state (routingStats + derived latency averages).
 * VIEWERS / GMV / BASKETS are mocked in v1 — they'll wire to Hands in Item 5
 * once distribution actuation is live. The moat metrics (% local, latency)
 * are real and already wired through useEmpireSocket.
 *
 * Layout: 6 equal-width cells, 1px hairline between them, 12px border radius
 * on the outer container, white background, monospace micro-labels.
 */
export function MetricsStrip({ stage = 'INTRO', routingStats = {}, agentLog = [] }) {
  // Derive Wav2Lip p50 from the most recent "Wav2Lip pitch rendered (NNNms)"
  // log entry if present. Agent log is already shape {agent, message, ...}
  // per main.log_event.
  const wav2lipMs = latestWav2LipMs(agentLog);
  const gemma4Ms  = latestGemmaMs(agentLog);

  const cells = [
    { k: 'PHASE',   v: stage,                  s: 'stage machine' },
    { k: 'VIEWERS', v: mockViewers(),          s: '+248/min' },
    { k: 'GMV',     v: mockGMV(routingStats),  s: 'this stream' },
    { k: 'BASKETS', v: mockBaskets(),          s: 'last 5min' },
    { k: 'WAV2LIP', v: wav2lipMs ? `${(wav2lipMs/1000).toFixed(1)}s` : '—',
                    s: wav2lipMs ? 'warm p50' : 'no render yet' },
    { k: 'GEMMA 4', v: gemma4Ms ? `${gemma4Ms}ms` : '—',
                    s: 'on-device' },
  ];

  return (
    <div style={styles.strip}>
      {cells.map((c, i) => (
        <div key={c.k} style={{
          ...styles.cell,
          borderLeft: i === 0 ? 'none' : '1px solid #18181b',
        }}>
          <div style={styles.label}>{c.k}</div>
          <div style={styles.value}>{c.v}</div>
          <div style={styles.sub}>{c.s}</div>
        </div>
      ))}
    </div>
  );
}

// ── Derivations ────────────────────────────────────────────────────────────

function latestWav2LipMs(log) {
  for (let i = log.length - 1; i >= 0; i--) {
    const msg = log[i]?.message || '';
    const m = msg.match(/Wav2Lip.*?\((\d+)ms\)/);
    if (m) return parseInt(m[1], 10);
  }
  return null;
}

function latestGemmaMs(log) {
  // gemma4.e4b → @... · intent=... · 284ms
  for (let i = log.length - 1; i >= 0; i--) {
    const msg = log[i]?.message || '';
    const m = msg.match(/gemma[^·]*·\s*(\d+)ms/i);
    if (m) return parseInt(m[1], 10);
  }
  return null;
}

// v1 mocks (pre-Hands). These go to real data in Item 5 when Hands broadcasts
// hands_published events with basket_impressions.
function mockViewers() { return '12,408'; }
function mockGMV(routingStats) {
  // Gesture at reality: mockViewers × average basket × routing-derived conversion.
  // Until Hands is wired the displayed value is static.
  return '$4,812';
}
function mockBaskets() { return '+47'; }

const styles = {
  strip: {
    display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)',
    gap: 0,
    background: '#fff',
    border: '1px solid #18181b',
    borderRadius: 12,
    overflow: 'hidden',
    fontFamily: '-apple-system, "SF Pro Text", "Inter", system-ui, sans-serif',
  },
  cell: {
    padding: '12px 16px',
    display: 'flex', flexDirection: 'column', gap: 1,
    background: '#fff',
  },
  label: {
    fontSize: 9, fontWeight: 700, letterSpacing: 0.8,
    color: '#86868b',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  },
  value: {
    fontSize: 22, fontWeight: 600, letterSpacing: -0.6,
    color: '#1d1d1f',
  },
  sub: {
    fontSize: 10, color: '#86868b', marginTop: 1,
  },
};
