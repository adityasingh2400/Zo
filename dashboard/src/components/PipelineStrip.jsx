import React from 'react';

/**
 * PipelineStrip — 5-cell intake pipeline visualization matching the
 * empire-mac.jsx mockup. Shows the Clip → Deepgram → Claude → Eleven →
 * Wav2Lip chain with per-stage latency pulled from the backend's agent_log
 * stream (same source that powers the iPhone's PipelineProgressView, but
 * rendered horizontally here as the mockup's "Intake pipeline" card).
 *
 * Data derivation: scans the last ~20 agent_log entries for known stage
 * taglines. Null-safe — missing stages render "—" rather than blowing up.
 */
const STAGES = [
  { key: 'clip',     label: 'Clip',     device: 'phone' },
  { key: 'deepgram', label: 'Deepgram', device: 'asr' },
  { key: 'claude',   label: 'Claude',   device: 'pitch' },
  { key: 'eleven',   label: 'Eleven',   device: 'tts' },
  { key: 'wav2lip',  label: 'Wav2Lip',  device: 'lipsync' },
];

export function PipelineStrip({ agentLog = [] }) {
  const latencies = extractLatencies(agentLog);

  return (
    <div style={styles.container}>
      <div style={styles.headerRow}>
        <span style={styles.headerTitle}>Intake pipeline</span>
        <button type="button" style={styles.reintakeButton}>RE-INTAKE</button>
      </div>
      <div style={styles.grid}>
        {STAGES.map((s, i) => {
          const sec = latencies[s.key];
          const hit = sec != null;
          const isLast = i === STAGES.length - 1;
          return (
            <div key={s.key} style={{
              ...styles.cell,
              background: isLast ? 'transparent' : (hit ? 'rgba(0,0,0,0.03)' : '#fafafa'),
              border: isLast ? '1px dashed #6e6e73' : '1px solid rgba(0,0,0,0.08)',
            }}>
              <div style={styles.label}>{s.label}</div>
              <div style={styles.time}>{sec != null ? `${sec}s` : '—'}</div>
              <div style={styles.device}>{s.device}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/**
 * Scan recent log entries for per-stage latency. Messages the backend emits:
 *   "Intake complete (NNNms)" → deepgram-ish
 *   "Sales pitch ready (NNNms)" → claude
 *   "TTS ready (NNNms, ...)" → eleven
 *   "Wav2Lip pitch rendered (NNNms)" → wav2lip
 *   "Video received." → clip proxy
 */
function extractLatencies(log) {
  const out = {};
  for (const entry of log) {
    const msg = entry?.message || '';
    const m = msg.match(/\((\d+)ms/);
    if (!m) continue;
    const ms = parseInt(m[1], 10);
    const sec = (ms / 1000).toFixed(1);
    if (/Intake complete/.test(msg))              out.deepgram = sec;
    else if (/Sales pitch ready/.test(msg))        out.claude   = sec;
    else if (/TTS ready/.test(msg))                out.eleven   = sec;
    else if (/Wav2Lip.*?rendered/.test(msg))       out.wav2lip  = sec;
    else if (/Claude:|Analyzing product/.test(msg))out.claude   = out.claude || sec;
  }
  // Approximate Clip stage latency as a fixed small value if any later
  // stage has landed (phone upload happens "instantly" from the backend's
  // perspective — the real clip time is measured on the iPhone side).
  if (!out.clip && (out.deepgram || out.claude || out.wav2lip)) {
    out.clip = '0.4';
  }
  return out;
}

const styles = {
  container: {
    padding: 14,
    background: '#fff',
    border: '1px solid rgba(0,0,0,0.08)',
    borderRadius: 12,
    fontFamily: '-apple-system, "SF Pro Text", "Inter", system-ui, sans-serif',
  },
  headerRow: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    marginBottom: 10,
  },
  headerTitle: {
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    fontSize: 10, letterSpacing: 1.2,
    textTransform: 'uppercase', color: '#86868b', fontWeight: 600,
  },
  reintakeButton: {
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    fontSize: 10, letterSpacing: 0.6,
    border: '1px solid rgba(0,0,0,0.08)', background: '#fff',
    padding: '3px 7px', borderRadius: 6, cursor: 'pointer',
    color: '#1d1d1f',
  },
  grid: {
    display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)',
    gap: 6, alignItems: 'start',
  },
  cell: {
    padding: 8, borderRadius: 8,
    textAlign: 'center',
  },
  label: { fontSize: 11, fontWeight: 600, color: '#1d1d1f' },
  time: {
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    fontSize: 10, color: '#86868b',
  },
  device: { fontSize: 9, color: '#86868b', marginTop: 2 },
};
