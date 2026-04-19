import React, { useEffect, useRef, useState } from 'react';

/**
 * Live telemetry for the router. This is the demo's money shot — it tells
 * the local-first story without narration: "X% local, $Y saved this stream."
 *
 * Reads routingDecisions + routingStats from useEmpireSocket and renders:
 *   - A big percentage (local routing rate)
 *   - A big dollar amount (total cost avoided vs all-cloud)
 *   - The last 6 decisions with tool, latency, and cost — color-coded
 *     so you can see local vs cloud at a glance from 10ft.
 *
 * Every row pulses when it first lands so the eye tracks new activity.
 */

const TOOL_STYLES = {
  respond_locally:   { color: '#22c55e', label: 'local',  icon: '⚡' },
  play_canned_clip:  { color: '#7c3aed', label: 'canned', icon: '◆' },
  block_comment:     { color: '#a1a1aa', label: 'blocked',icon: '⛔' },
  escalate_to_cloud: { color: '#f59e0b', label: 'cloud',  icon: '☁' },
};

const TOOL_LABEL = {
  respond_locally:   'respond_locally',
  play_canned_clip:  'play_canned_clip',
  block_comment:     'block_comment',
  escalate_to_cloud: 'escalate_to_cloud',
};

export function RoutingPanel({ routingDecisions = [], routingStats }) {
  const { total = 0, local = 0, cost_saved_usd = 0 } = routingStats || {};
  const pctLocal = total > 0 ? Math.round((local / total) * 100) : 0;
  const recent = routingDecisions.slice(-6).reverse();  // newest first

  return (
    <div style={styles.container}>
      <div style={styles.headerRow}>
        <h3 style={styles.title}>Local Routing</h3>
        <span style={styles.badge}>on-device first</span>
      </div>

      <div style={styles.kpiRow}>
        <Kpi value={`${pctLocal}%`} label="local" color="#22c55e" />
        <Kpi value={formatUSD(cost_saved_usd)} label="saved" color="#fafafa" />
        <Kpi value={`${total}`} label="decisions" color="#a1a1aa" muted />
      </div>

      <div style={styles.feedHeader}>
        <span>Recent decisions</span>
        {total === 0 && <span style={styles.hint}>fire a comment to begin</span>}
      </div>

      <div style={styles.feed}>
        {recent.length === 0 ? (
          <div style={styles.empty}>No decisions yet.</div>
        ) : (
          recent.map((d, i) => <DecisionRow key={d.seq ?? `${d.receivedAt}-${d.tool}-${d.comment}`} decision={d} fresh={i === 0} />)
        )}
      </div>
    </div>
  );
}

function Kpi({ value, label, color, muted }) {
  return (
    <div style={styles.kpi}>
      <span style={{ ...styles.kpiValue, color, opacity: muted ? 0.6 : 1 }}>{value}</span>
      <span style={styles.kpiLabel}>{label}</span>
    </div>
  );
}

function DecisionRow({ decision, fresh }) {
  const t = TOOL_STYLES[decision.tool] || { color: '#a1a1aa', label: '', icon: '·' };
  const ms = decision.ms ?? 0;
  const saved = decision.cost_saved_usd ?? 0;
  const [pulse, setPulse] = useState(fresh);
  useEffect(() => {
    if (!fresh) return;
    setPulse(true);
    const h = setTimeout(() => setPulse(false), 900);
    return () => clearTimeout(h);
  }, [fresh, decision.receivedAt]);

  return (
    <div style={{ ...styles.row, borderLeftColor: t.color, background: pulse ? 'rgba(34,197,94,0.10)' : 'transparent' }}>
      <span style={{ ...styles.rowIcon, color: t.color }}>{t.icon}</span>
      <div style={styles.rowText}>
        <span style={styles.rowTool}>{TOOL_LABEL[decision.tool] || decision.tool}</span>
        <span style={styles.rowComment} title={decision.comment}>
          "{(decision.comment || '').slice(0, 34)}{(decision.comment || '').length > 34 ? '…' : ''}"
        </span>
      </div>
      <span style={{ ...styles.rowLatency, color: decision.was_local ? '#22c55e' : '#f59e0b' }}>
        {ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`}
      </span>
      <span style={styles.rowCost}>
        {saved > 0 ? `-${formatUSD(saved)}` : '—'}
      </span>
    </div>
  );
}

function formatUSD(v) {
  if (!v || v === 0) return '$0.00';
  if (v < 0.01) return `$${v.toFixed(5)}`;
  if (v < 1)    return `$${v.toFixed(4)}`;
  return `$${v.toFixed(2)}`;
}

const styles = {
  container: {
    display: 'flex', flexDirection: 'column',
    background: '#0f0f12',
    border: '1px solid #27272a',
    borderRadius: 12, padding: 14, gap: 10,
    minHeight: 0, overflow: 'hidden',
  },
  headerRow: {
    display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 12,
  },
  title: {
    margin: 0, fontSize: 14, fontWeight: 700, letterSpacing: 1.5,
    textTransform: 'uppercase', color: '#fafafa',
  },
  badge: {
    fontSize: 10, fontWeight: 700, letterSpacing: 1,
    color: '#22c55e', background: 'rgba(34,197,94,0.12)',
    border: '1px solid rgba(34,197,94,0.35)',
    borderRadius: 999, padding: '2px 8px',
  },
  kpiRow: {
    display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)',
    gap: 8, padding: '6px 0',
  },
  kpi: {
    display: 'flex', flexDirection: 'column', alignItems: 'center',
    gap: 2, padding: '6px 4px',
    background: '#09090b', borderRadius: 8, border: '1px solid #18181b',
  },
  kpiValue: {
    fontSize: 26, fontWeight: 800, letterSpacing: -0.5,
    fontVariantNumeric: 'tabular-nums',
  },
  kpiLabel: {
    fontSize: 10, fontWeight: 700, letterSpacing: 1.5,
    textTransform: 'uppercase', color: '#52525b',
  },
  feedHeader: {
    display: 'flex', justifyContent: 'space-between',
    fontSize: 10, fontWeight: 700, letterSpacing: 1.5,
    textTransform: 'uppercase', color: '#52525b',
    paddingTop: 4, borderTop: '1px dashed #27272a',
  },
  hint: { color: '#3f3f46', fontSize: 10, fontStyle: 'italic', textTransform: 'none', letterSpacing: 0 },
  feed: {
    display: 'flex', flexDirection: 'column', gap: 4,
    overflowY: 'auto', minHeight: 0, flex: 1,
  },
  empty: {
    color: '#3f3f46', fontSize: 12, fontStyle: 'italic',
    padding: '12px 0', textAlign: 'center',
  },
  row: {
    display: 'grid',
    gridTemplateColumns: '18px 1fr auto auto',
    alignItems: 'center', gap: 8,
    padding: '6px 8px', borderRadius: 6,
    borderLeft: '3px solid #27272a',
    transition: 'background 300ms ease',
  },
  rowIcon: { fontSize: 14, textAlign: 'center' },
  rowText: { display: 'flex', flexDirection: 'column', minWidth: 0 },
  rowTool: {
    fontSize: 11, fontWeight: 700, color: '#fafafa',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  },
  rowComment: {
    fontSize: 10, color: '#71717a',
    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
  },
  rowLatency: {
    fontSize: 12, fontWeight: 700,
    fontVariantNumeric: 'tabular-nums',
  },
  rowCost: {
    fontSize: 11, color: '#71717a',
    fontVariantNumeric: 'tabular-nums',
    minWidth: 44, textAlign: 'right',
  },
};
