import React, { useEffect, useRef, useState } from 'react';

/**
 * CostTicker — the demo's kill shot.
 *
 * Tells the unit-economics story without a slide. Sits on the dark frame
 * outside the 9:16 stage area and reads in real time as the router emits
 * `routing_decision` events:
 *
 *   - Local tools (respond_locally / play_canned_clip / block_comment)
 *     add $0.00 — the ticker visibly stays parked at zero. That's the moat.
 *   - escalate_to_cloud adds $0.00035 (the observed Bedrock Haiku cost,
 *     grounded in backend/agents/router.COST_PER_CLOUD_COMMENT_USD). The
 *     count-up animation makes the increment unmissable from 30+ feet.
 *
 * Subscribes to the shared WebSocket directly (no prop drilling) so it can
 * be dropped into StageView next to LiveStage with no parent rewiring.
 *
 * Hotkeys (when this component is mounted):
 *   R — reset to $0 (demo restart between rehearsals or before stage)
 *
 * Visual contract:
 *   - Big monospace digits, $0.00000 format. Front three rows must read it.
 *   - 320ms count-up + flash on every increment so the eye tracks the ticks.
 *   - Comparison row underneath: "vs $9,628/mo human team" — the closing
 *     line of the demo. Fixed text, mirrors the design doc.
 */

// Bedrock Claude Haiku cost per cloud-escalated comment.
// Mirrors backend/agents/router.COST_PER_CLOUD_COMMENT_USD — change there
// first; this is the visible side of the same number.
const COST_PER_CLOUD = 0.00035;

// Kill-shot animation — short enough to feel sharp, long enough to be visible.
const COUNT_UP_MS = 320;

// Human comparison: a US-based live commerce host runs ~$9,628/mo at 40hr
// weeks. The contrast against $0.00 routing cost is the demo's closing line.
// Kept as a structured value so the Comparison (Lidwell p42) layout below can
// render it on a common scale with the live cost rather than as a footnote.
const HUMAN_COMPARISON = {
  amount: 9628,
  perPeriod: '/mo',
  label: 'human host team',
};

export function CostTicker({ wsRef, connected, comparison = HUMAN_COMPARISON }) {
  const [actual, setActual] = useState(0);   // truth: total cost incurred
  const [shown, setShown]   = useState(0);   // animated value the eye sees
  const [flash, setFlash]   = useState(false);
  // Common Fate (Lidwell p40): elements that move together are perceived as
  // related. The CostTicker and the RoutingPanel dot tell ONE story — "the
  // router stayed local, the cost held at zero." Pulsing them together makes
  // the cause→effect link readable from 30 ft. Color encodes which kind of
  // decision just landed: green = local (held the line), amber = cloud
  // (incurred a tick), null = resting.
  const [pulse, setPulse] = useState(null);  // 'local' | 'cloud' | null
  const animRef = useRef(null);
  const pulseTimerRef = useRef(null);

  // Subscribe to routing_decision events directly — same pattern LiveStage
  // uses (addEventListener, never replace the existing onmessage handler).
  //
  // Dep MUST include `connected` (not just `wsRef`). wsRef is a stable
  // React ref object whose identity never changes, and useEmpireSocket
  // creates the WebSocket inside its OWN useEffect — which fires AFTER
  // child useEffects (effects run bottom-up in React). So at the moment
  // this effect runs on mount, wsRef.current is still null, the listener
  // isn't attached, and the dep `[wsRef]` never re-triggers — the
  // ticker silently stays at $0 forever even on cloud escalates. Keying
  // off `connected` gives the re-trigger we need: ws.onopen flips it
  // true → effect re-runs → wsRef.current is now the live socket →
  // listener attaches. Same fix applies to TikTokShopOverlay's heart
  // listener and LiveStage's useVoiceStage subscription.
  useEffect(() => {
    const ws = wsRef?.current;
    if (!ws) return;
    function onMessage(e) {
      let msg;
      try { msg = JSON.parse(e.data); } catch { return; }
      if (msg.type !== 'routing_decision') return;
      // Only escalate_to_cloud actually costs money; everything else is the
      // local-first wedge and stays at $0. This mirrors COST_SAVED_USD_PER_TOOL
      // in backend/agents/router.py: local tools have was_local=true.
      if (msg.tool === 'escalate_to_cloud') {
        setActual(v => v + COST_PER_CLOUD);
        // Cloud pulse comes through the count-up animation effect below,
        // which sets pulse='cloud' when `actual` increases.
      } else {
        // Local-path decision — no money spent, but the ticker should
        // still react so the audience reads "stayed at zero" as an active
        // win rather than a passive flat line. (Common Fate, p40.)
        setPulse('local');
        if (pulseTimerRef.current) clearTimeout(pulseTimerRef.current);
        pulseTimerRef.current = setTimeout(() => setPulse(null), 700);
      }
    }
    ws.addEventListener('message', onMessage);
    return () => {
      ws.removeEventListener('message', onMessage);
      if (pulseTimerRef.current) clearTimeout(pulseTimerRef.current);
    };
  }, [wsRef, connected]);

  // Count-up animation: lerp from `shown` → `actual` over COUNT_UP_MS.
  // Cancels any in-flight animation if a new tick lands mid-flight so the
  // ticker stays responsive when several escalates happen back-to-back.
  useEffect(() => {
    if (actual === shown) return;
    if (animRef.current) cancelAnimationFrame(animRef.current);
    const start = performance.now();
    const from = shown;
    const to = actual;
    setFlash(true);
    // The cost only moves when an escalate_to_cloud lands, so any actual→shown
    // delta means "we just spent money." Pulse amber in lockstep with the
    // routing dot (Common Fate, Lidwell p40).
    setPulse('cloud');
    if (pulseTimerRef.current) clearTimeout(pulseTimerRef.current);
    pulseTimerRef.current = setTimeout(() => setPulse(null), 700);
    function step(now) {
      const t = Math.min(1, (now - start) / COUNT_UP_MS);
      // Smooth ease-out so the digits decelerate into place, not just snap.
      const eased = 1 - Math.pow(1 - t, 3);
      setShown(from + (to - from) * eased);
      if (t < 1) {
        animRef.current = requestAnimationFrame(step);
      } else {
        // Hold the flash a beat after the digits settle so the eye catches it.
        setTimeout(() => setFlash(false), 240);
      }
    }
    animRef.current = requestAnimationFrame(step);
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [actual]);  // shown intentionally omitted: the animation IS what catches up.

  // R hotkey: hard reset to $0 between rehearsals or right before stage.
  // Bound to the window so it works from anywhere on the page, including
  // when nothing has focus (default state when the dashboard first loads).
  useEffect(() => {
    function onKey(e) {
      if (e.key !== 'r' && e.key !== 'R') return;
      // Avoid hijacking when the user is typing into a text field.
      const tag = (e.target?.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'textarea') return;
      setActual(0);
      setShown(0);
      setFlash(false);
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  // Pulse → border + glow color. Common Fate (p40) ties this to the routing
  // panel: when its dot pulses green/amber, this surface pulses with it.
  const pulseColor = pulse === 'cloud' ? '#fbbf24'
                   : pulse === 'local' ? '#22c55e'
                   : null;

  return (
    <div
      // Liquid Glass surface — strong flavor (heavier blur, brighter rim)
      // because the cost ticker is the demo's most-watched single number.
      // --still flavor skips the SVG distortion (the bezel is flat black,
      // there's nothing visible to refract). Pulse-driven outline overrides
      // the utility's default rim via inline style — Common Fate (Lidwell
      // p40) ties the green/amber pulse here to the routing dot pulse.
      className="lg-glass lg-glass--strong lg-glass--still"
      style={{
        ...styles.container,
        ...(pulseColor && {
          borderColor: pulseColor,
          boxShadow: `0 12px 40px rgba(0,0,0,0.45), 0 0 24px ${pulseColor}55`,
        }),
        transition: 'border-color 240ms ease, box-shadow 240ms ease',
      }}
    >
      <div style={styles.eyebrow}>COST THIS STREAM</div>

      {/* Comparison (Lidwell p42): present Zo cost and the human-team cost on
          a common scale (same row, same type treatment, same label rhythm).
          The eye reads them as compared values, not as a number with a
          footnote. The vertical divider is the visual axis of comparison. */}
      <div style={styles.compareRow}>
        <div style={styles.col}>
          <div
            style={{
              ...styles.value,
              color: flash ? '#fbbf24' : '#fafafa',
              textShadow: flash ? '0 0 24px rgba(251,191,36,0.6)' : 'none',
              transition: 'color 200ms ease, text-shadow 200ms ease',
            }}
          >
            {formatCost(shown)}
          </div>
          <div style={styles.colLabel}>Zo · live now</div>
        </div>
        <div style={styles.compareDivider} />
        <div style={{ ...styles.col, ...styles.colMuted }}>
          <div style={styles.valueMuted}>
            {formatHuman(comparison.amount)}
            <span style={styles.valueUnit}>{comparison.perPeriod}</span>
          </div>
          <div style={styles.colLabel}>{comparison.label}</div>
        </div>
      </div>
    </div>
  );
}

// Always render five-decimal precision so the audience sees the
// "$0.00035" tick clearly. Numbers smaller than 0.00001 still show as
// $0.00000 so the resting state is visually clean.
function formatCost(v) {
  const safe = Number.isFinite(v) ? Math.max(0, v) : 0;
  return `$${safe.toFixed(5)}`;
}

// Whole-dollar comparison value with thousands separators ("$9,628"). The
// human comparison never gets fractional precision — pretending we measure
// payroll to five decimals would undercut the point.
function formatHuman(v) {
  const safe = Number.isFinite(v) ? Math.max(0, v) : 0;
  return `$${Math.round(safe).toLocaleString('en-US')}`;
}

const styles = {
  container: {
    // Layout/typography only. Background, border, backdrop-filter, shadow,
    // and inset highlight are all owned by the .lg-glass--strong utility
    // (lib/liquid-glass.css), which already encodes the Top-Down Lighting
    // Bias (Lidwell p196) inset rim + Apple WWDC '25 specular treatment.
    display: 'flex', flexDirection: 'column', alignItems: 'stretch',
    gap: 8, padding: '14px 16px',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    minWidth: 280,
  },
  eyebrow: {
    fontSize: 10, fontWeight: 800, letterSpacing: 2,
    color: '#52525b', textTransform: 'uppercase',
  },
  compareRow: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 1fr) auto minmax(0, 1fr)',
    alignItems: 'center', gap: 12,
  },
  col: {
    display: 'flex', flexDirection: 'column',
    gap: 2, minWidth: 0,
  },
  colMuted: {
    // Comparison (p42): same shape, dimmer color. Equal weight would lie
    // about live status; clearly muted "human team" preserves the truth
    // that one side is live cost, the other is the reference baseline.
    opacity: 0.85,
  },
  compareDivider: {
    width: 1, alignSelf: 'stretch', minHeight: 36,
    background:
      'linear-gradient(to bottom, transparent, rgba(82,82,91,0.8), transparent)',
  },
  value: {
    fontSize: 28, fontWeight: 900, letterSpacing: -0.5,
    fontVariantNumeric: 'tabular-nums',
    lineHeight: 1,
    paddingTop: 2,
  },
  valueMuted: {
    fontSize: 24, fontWeight: 800, letterSpacing: -0.4,
    fontVariantNumeric: 'tabular-nums',
    lineHeight: 1, color: '#a1a1aa',
    paddingTop: 2,
    display: 'inline-flex', alignItems: 'baseline', gap: 2,
  },
  valueUnit: {
    fontSize: 12, fontWeight: 700, color: '#71717a',
    letterSpacing: 0.5,
  },
  colLabel: {
    fontSize: 10, fontWeight: 700, letterSpacing: 1,
    color: '#52525b', textTransform: 'uppercase',
    paddingTop: 2,
    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
  },
};
