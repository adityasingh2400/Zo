import React from 'react';
import { LIVE_STAGES } from '../hooks/useEmpireSocket';

/** Slim 4-step bar: INTRO · BRIDGE · PITCH · LIVE. */
export function StageBar({ stage }) {
  const activeIdx = Math.max(0, LIVE_STAGES.indexOf(stage));
  return (
    <div style={styles.row}>
      {LIVE_STAGES.map((s, i) => {
        const done = i < activeIdx;
        const active = i === activeIdx;
        return (
          <React.Fragment key={s}>
            <div style={styles.step}>
              <span
                style={{
                  ...styles.dot,
                  background: active ? '#22c55e' : done ? '#3b82f6' : '#3f3f46',
                  boxShadow: active ? '0 0 8px rgba(34,197,94,0.6)' : 'none',
                }}
              />
              <span
                style={{
                  ...styles.label,
                  color: active ? '#fafafa' : done ? '#a1a1aa' : '#52525b',
                  fontWeight: active ? 700 : 500,
                }}
              >
                {s}
              </span>
            </div>
            {i < LIVE_STAGES.length - 1 && (
              <span
                style={{
                  ...styles.connector,
                  background: i < activeIdx ? '#3b82f6' : '#27272a',
                }}
              />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}

const styles = {
  row: {
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '6px 14px', background: '#0a0a0a',
    borderRadius: 999, border: '1px solid #27272a',
  },
  step: { display: 'flex', alignItems: 'center', gap: 6 },
  dot: { width: 8, height: 8, borderRadius: 4, transition: 'all 0.2s ease' },
  label: { fontSize: 11, letterSpacing: 1.5, textTransform: 'uppercase' },
  connector: { width: 28, height: 2, borderRadius: 1, transition: 'all 0.3s ease' },
};
