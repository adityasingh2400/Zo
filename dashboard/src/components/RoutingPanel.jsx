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

export function RoutingPanel({ routingDecisions = [], routingStats, compact = false }) {
  const { total = 0, local = 0, cost_saved_usd = 0 } = routingStats || {};
  const pctLocal = total > 0 ? Math.round((local / total) * 100) : 0;
  // Avg latency over the rolling window of decisions — gives the audience
  // a tight number for "sub-second routing" without over-claiming the
  // single-decision low. Falls back to "—" when no decisions yet.
  const avgMs = routingDecisions.length
    ? Math.round(routingDecisions.reduce((acc, d) => acc + (Number(d.ms) || 0), 0)
                 / routingDecisions.length)
    : null;
  const recent = routingDecisions.slice(-6).reverse();  // newest first
  const lastDecision = routingDecisions[routingDecisions.length - 1];

  // Compact mode: a single horizontal strip designed to live on the dark
  // bezel of the TikTok Shop overlay. Reads from 30+ feet — the moat made
  // visible without taking stage real estate. Pulses green on each new
  // local-path decision so the audience tracks the ticking.
  if (compact) {
    return (
      <CompactStrip
        local={local}
        total={total}
        pctLocal={pctLocal}
        costSaved={cost_saved_usd}
        avgMs={avgMs}
        lastDecision={lastDecision}
      />
    );
  }

  return (
    <div style={styles.container}>
      <div style={styles.headerRow}>
        <h3 style={styles.title}>Local Routing</h3>
        <span style={styles.badge}>on-device first</span>
      </div>

      {/* KPI order is deliberate. Serial Position Effects (Lidwell p178):
          items at the start (primacy) and end (recency) of a list are
          remembered better than the middle. Gutenberg Diagram (p100): for
          left-to-right readers, the top-left is the primary optical area.
          So we put the demo's punchline ($ saved vs all-cloud) FIRST, the
          supporting metric (% local) SECOND, and the noisy counter (total
          decisions) LAST and muted — context, not headline. */}
      <div style={styles.kpiRow}>
        <Kpi value={formatUSD(cost_saved_usd)} label="saved" color="#fafafa" />
        <Kpi value={`${pctLocal}%`} label="local" color="#22c55e" />
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

// CompactStrip — the stage-mode flavor of the routing panel. One row, three
// numbers + a status dot. Pulses each time a new local decision lands so
// the audience can clock the routing as it happens, not after.
function CompactStrip({ local, total, pctLocal, costSaved, avgMs, lastDecision }) {
  const [pulse, setPulse] = useState(false);
  const lastSeqRef = useRef(null);
  useEffect(() => {
    const seq = lastDecision?.seq;
    if (seq == null || seq === lastSeqRef.current) return;
    lastSeqRef.current = seq;
    setPulse(true);
    const h = setTimeout(() => setPulse(false), 700);
    return () => clearTimeout(h);
  }, [lastDecision?.seq]);

  // Color the dot based on whether the LATEST decision was local or cloud —
  // the audience reads the dot before the digits. Default green when nothing
  // has happened yet (the resting "we'd be local if a comment landed" tone).
  const wasLocal = lastDecision ? !!lastDecision.was_local : true;
  const dotColor = wasLocal ? '#22c55e' : '#f59e0b';

  return (
    <div
      // Liquid Glass surface — same recipe as the CostTicker so the two
      // bezel chrome pieces feel like one paired stack. --still flavor
      // because there's no content behind the bezel to refract; the dark
      // glass aesthetic still upgrades the look from "panel" to "lens".
      // Pulse driven by the latest routing decision (Common Fate, p40).
      className="lg-glass lg-glass--still"
      style={{
        ...stripStyles.container,
        ...(pulse && {
          borderColor: dotColor,
          boxShadow: `0 12px 40px rgba(0,0,0,0.45), 0 0 18px ${dotColor}66`,
        }),
        transition: 'border-color 240ms ease, box-shadow 240ms ease',
      }}
    >
      <div style={stripStyles.dotWrap}>
        <span style={{ ...stripStyles.dot, background: dotColor, boxShadow: `0 0 10px ${dotColor}` }} />
        <span style={stripStyles.dotLabel}>ROUTER</span>
      </div>
      <div style={stripStyles.divider} />
      {/* Stat order, same logic as the full panel above: Serial Position
          Effects (p178) puts the dollar figure in the primacy slot, then
          % local, then latency as quiet context in the recency slot. The
          stage strip is the audience's only telemetry surface — getting the
          read-order right is the difference between a story and a wall of
          numbers. */}
      <Stat label="SAVED" value={formatUSD(costSaved)} accent="#fafafa" />
      <div style={stripStyles.divider} />
      <Stat label="LOCAL" value={`${local} / ${total}`} sub={`${pctLocal}%`} accent="#22c55e" />
      <div style={stripStyles.divider} />
      <Stat
        label="AVG"
        value={avgMs == null ? '—' : (avgMs < 1000 ? `${avgMs}ms` : `${(avgMs / 1000).toFixed(1)}s`)}
        accent="#a1a1aa"
      />
    </div>
  );
}

function Stat({ label, value, sub, accent }) {
  return (
    <div style={stripStyles.stat}>
      <span style={stripStyles.statLabel}>{label}</span>
      <span style={{ ...stripStyles.statValue, color: accent }}>
        {value}
        {sub && <span style={stripStyles.statSub}>{` ${sub}`}</span>}
      </span>
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
    gap: 2, padding: '8px 4px',
    // Top-Down Lighting Bias (Lidwell p196): the brain assumes light comes
    // from above, so a thin white inset on the top edge + a dark gradient
    // body + soft outer shadow reads as a "raised" tile. Lifts the demo's
    // money numbers above the surrounding panel chrome. Mirrors the same
    // depth language used on the CostTicker so all telemetry surfaces feel
    // like one coordinated stack.
    background:
      'linear-gradient(rgba(255,255,255,0.05), rgba(255,255,255,0) 40%), #09090b',
    borderRadius: 8,
    border: '1px solid #18181b',
    boxShadow:
      '0 1px 0 rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05)',
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

// Stage-mode strip — designed for stadium readability. Lives on the dark
// frame next to the 9:16 overlay; same data as the full panel, distilled
// into a single horizontal row.
const stripStyles = {
  container: {
    // Layout/typography only — glass surface owned by .lg-glass utility.
    display: 'flex', alignItems: 'center', gap: 14,
    padding: '10px 16px',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  },
  dotWrap: {
    display: 'flex', alignItems: 'center', gap: 8,
  },
  dot: {
    width: 10, height: 10, borderRadius: 5,
    animation: 'pulse 1.2s ease-in-out infinite',
  },
  dotLabel: {
    fontSize: 10, fontWeight: 800, letterSpacing: 1.5,
    color: '#a1a1aa', textTransform: 'uppercase',
  },
  divider: {
    width: 1, height: 28, background: '#27272a',
  },
  stat: {
    display: 'flex', flexDirection: 'column', alignItems: 'flex-start',
    gap: 1, minWidth: 70,
  },
  statLabel: {
    fontSize: 9, fontWeight: 800, letterSpacing: 1.5,
    color: '#52525b', textTransform: 'uppercase',
  },
  statValue: {
    fontSize: 18, fontWeight: 800, letterSpacing: -0.3,
    fontVariantNumeric: 'tabular-nums', lineHeight: 1.1,
  },
  statSub: {
    fontSize: 11, fontWeight: 600, color: '#71717a',
    marginLeft: 2,
  },
};
