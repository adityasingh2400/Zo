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

export function CostTicker({ wsRef, comparison = '$9,628/mo human team' }) {
  const [actual, setActual] = useState(0);   // truth: total cost incurred
  const [shown, setShown]   = useState(0);   // animated value the eye sees
  const [flash, setFlash]   = useState(false);
  const animRef = useRef(null);

  // Subscribe to routing_decision events directly — same pattern LiveStage
  // uses (addEventListener, never replace the existing onmessage handler).
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
      }
    }
    ws.addEventListener('message', onMessage);
    return () => ws.removeEventListener('message', onMessage);
  }, [wsRef]);

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

  return (
    <div style={styles.container}>
      <div style={styles.label}>COST THIS STREAM</div>
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
      <div style={styles.comparison}>vs {comparison}</div>
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

const styles = {
  container: {
    display: 'flex', flexDirection: 'column', alignItems: 'flex-end',
    gap: 4, padding: '12px 18px',
    background: 'rgba(9,9,11,0.85)',
    border: '1px solid rgba(63,63,70,0.6)',
    borderRadius: 12,
    backdropFilter: 'blur(8px)',
    boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    minWidth: 220,
  },
  label: {
    fontSize: 10, fontWeight: 800, letterSpacing: 2,
    color: '#52525b', textTransform: 'uppercase',
  },
  value: {
    fontSize: 38, fontWeight: 900, letterSpacing: -0.5,
    fontVariantNumeric: 'tabular-nums',
    lineHeight: 1,
    paddingTop: 2,
  },
  comparison: {
    fontSize: 11, fontWeight: 600, letterSpacing: 0.5,
    color: '#a1a1aa', paddingTop: 2,
  },
};
