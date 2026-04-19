import React, { useEffect, useState } from 'react';

/**
 * BRAIN — persistent telemetry panel.
 *
 * Polls /api/brain/stats every `pollIntervalMs`. Unlike RoutingPanel (live WS
 * stream of decisions in this session), BRAIN reads the SQLite-backed log of
 * EVERY decision across EVERY backend restart. The headline KPIs are:
 *   - Cost saved (cumulative)
 *   - % local (the moat number)
 *   - Total events processed
 * Plus two lists:
 *   - Top matched answers (which Q/A entries are doing the work)
 *   - Top misses (tokens that recur in escalate_to_cloud comments — what the
 *     local index is missing; feeds the next round of Q/A authoring)
 *
 * Roadmap: conversion-aware ranking will surface here too — which
 * answers correlate with add-to-cart, etc.
 */
const API_BASE = `http://${window.location.hostname}:8000`;

export function BrainPanel({ pollIntervalMs = 5000 }) {
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function fetchStats() {
      try {
        const r = await fetch(`${API_BASE}/api/brain/stats`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        if (cancelled) return;
        setStats(data);
        setError(null);
      } catch (e) {
        if (!cancelled) setError(e.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    fetchStats();
    const id = setInterval(fetchStats, pollIntervalMs);
    return () => { cancelled = true; clearInterval(id); };
  }, [pollIntervalMs]);

  if (loading && !stats) {
    return (
      <div style={styles.container}>
        <Header />
        <div style={styles.empty}>Loading…</div>
      </div>
    );
  }
  if (error && !stats) {
    return (
      <div style={styles.container}>
        <Header />
        <div style={styles.empty}>BRAIN unreachable: {error}</div>
      </div>
    );
  }

  const { total = 0, pct_local = 0, total_cost_saved_usd = 0,
          by_tool = {}, top_answers = [], top_misses = [] } = stats || {};

  return (
    <div style={styles.container}>
      <Header />
      <div style={styles.kpiRow}>
        <Kpi value={`${pct_local}%`} label="local" color="#22c55e" />
        <Kpi value={formatUSD(total_cost_saved_usd)} label="saved" color="#fafafa" />
        <Kpi value={`${total}`} label="events" color="#a1a1aa" muted />
      </div>

      <div style={styles.byToolRow}>
        {Object.entries(by_tool).map(([tool, n]) => (
          <ToolPill key={tool} tool={tool} count={n} />
        ))}
        {Object.keys(by_tool).length === 0 && (
          <span style={styles.hint}>fire a comment to begin</span>
        )}
      </div>

      <div style={styles.twoCol}>
        <ListSection
          title="TOP ANSWERS"
          subtitle="most-matched local Q/A"
          empty="no local matches yet"
          items={top_answers.map(a => ({ left: a.answer_id, right: a.count }))}
        />
        <ListSection
          title="TOP MISSES"
          subtitle="recurring tokens in cloud escalates — author Q/A for these"
          empty="no escalates yet"
          items={top_misses.map(m => ({ left: m.token, right: m.count }))}
        />
      </div>
    </div>
  );
}

function Header() {
  return (
    <div style={styles.headerRow}>
      <h3 style={styles.title}>BRAIN</h3>
      <span style={styles.badge}>persistent</span>
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

const TOOL_COLORS = {
  respond_locally:   '#22c55e',
  play_canned_clip:  '#7c3aed',
  block_comment:     '#a1a1aa',
  escalate_to_cloud: '#f59e0b',
};

function ToolPill({ tool, count }) {
  const color = TOOL_COLORS[tool] || '#a1a1aa';
  return (
    <span style={{
      ...styles.toolPill,
      color,
      borderColor: `${color}55`,
      background: `${color}1a`,
    }}>
      {tool}: {count}
    </span>
  );
}

function ListSection({ title, subtitle, empty, items }) {
  return (
    <div style={styles.listSection}>
      <div style={styles.listHeader}>
        <span style={styles.listTitle}>{title}</span>
        <span style={styles.listSubtitle}>{subtitle}</span>
      </div>
      <div style={styles.listBody}>
        {items.length === 0
          ? <div style={styles.empty}>{empty}</div>
          : items.map((it, i) => (
              <div key={`${it.left}-${i}`} style={styles.listRow}>
                <span style={styles.listLeft} title={it.left}>{it.left}</span>
                <span style={styles.listRight}>{it.right}</span>
              </div>
            ))
        }
      </div>
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
    color: '#7c3aed', background: 'rgba(124,58,237,0.12)',
    border: '1px solid rgba(124,58,237,0.35)',
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
  byToolRow: {
    display: 'flex', flexWrap: 'wrap', gap: 6,
    paddingTop: 4, borderTop: '1px dashed #27272a',
  },
  toolPill: {
    fontSize: 10, fontWeight: 700, letterSpacing: 0.5,
    border: '1px solid', borderRadius: 999,
    padding: '3px 8px',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  },
  twoCol: {
    flex: 1, display: 'grid', gridTemplateColumns: '1fr 1fr',
    gap: 10, minHeight: 0,
  },
  listSection: {
    display: 'flex', flexDirection: 'column',
    background: '#09090b', borderRadius: 8, border: '1px solid #18181b',
    padding: 10, gap: 6, minHeight: 0, overflow: 'hidden',
  },
  listHeader: {
    display: 'flex', flexDirection: 'column', gap: 1,
    paddingBottom: 4, borderBottom: '1px dashed #27272a',
  },
  listTitle: {
    fontSize: 10, fontWeight: 800, letterSpacing: 1.5,
    color: '#fafafa', textTransform: 'uppercase',
  },
  listSubtitle: {
    fontSize: 9, color: '#52525b', letterSpacing: 0.3,
  },
  listBody: {
    display: 'flex', flexDirection: 'column', gap: 3,
    overflowY: 'auto', flex: 1, minHeight: 0,
  },
  listRow: {
    display: 'grid', gridTemplateColumns: '1fr auto',
    alignItems: 'center', gap: 8,
    padding: '4px 6px', borderRadius: 4,
  },
  listLeft: {
    fontSize: 11, color: '#d4d4d8',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
  },
  listRight: {
    fontSize: 12, fontWeight: 700, color: '#fafafa',
    fontVariantNumeric: 'tabular-nums',
  },
  hint: { color: '#3f3f46', fontSize: 10, fontStyle: 'italic' },
  empty: {
    color: '#3f3f46', fontSize: 11, fontStyle: 'italic',
    padding: '12px 0', textAlign: 'center',
  },
};
