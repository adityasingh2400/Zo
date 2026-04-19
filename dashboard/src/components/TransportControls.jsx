import React, { useEffect, useState } from 'react';

/**
 * TransportControls — transport cluster + force-phase + On Air pill.
 *
 * Matches the empire-mac.jsx stage row below the metrics strip:
 *   [StageBar · INTRO → BRIDGE → PITCH → LIVE]
 *     ⏮⏸⏭   00:04 / 00:20   FORCE [INTRO][BRIDGE][PITCH][LIVE]   ● On Air
 *
 * Transport (⏮⏸⏭) is cosmetic in v1 — Director's stage machine auto-advances
 * based on pipeline progress. Force-phase POSTs to /api/director/force_phase
 * (new in Item 2) to manually jump the state machine. On Air toggle hits
 * /api/director/on_air to pause/resume all broadcasting (useful between takes).
 */
const API_BASE = `http://${window.location.hostname}:8000`;

const PHASES = ['INTRO', 'BRIDGE', 'PITCH', 'LIVE'];

export function TransportControls({ stage = 'INTRO', onAir: onAirProp = true }) {
  const [onAir, setOnAir] = useState(onAirProp);
  const [busy, setBusy] = useState(false);
  const [lang, setLang] = useState('en');
  const [supportedLangs, setSupportedLangs] = useState(null);

  // Hydrate supported languages from /api/live/language on mount. Graceful
  // degrade if the endpoint 404s (Item 6 not shipped).
  useEffect(() => {
    (async () => {
      try {
        const r = await fetch(`${API_BASE}/api/live/language`);
        if (!r.ok) return;
        const data = await r.json();
        setLang(data.active_language || 'en');
        setSupportedLangs(data.supported || null);
      } catch { /* silent — picker hidden when supportedLangs stays null */ }
    })();
  }, []);

  async function changeLang(e) {
    const next = e.target.value;
    setLang(next);
    try {
      const fd = new FormData();
      fd.append('lang', next);
      await fetch(`${API_BASE}/api/live/language`, { method: 'POST', body: fd });
    } catch { /* swallow — local state already updated */ }
  }

  async function forcePhase(phase) {
    if (busy) return;
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append('phase', phase);
      await fetch(`${API_BASE}/api/director/force_phase`, {
        method: 'POST', body: fd,
      });
    } catch { /* silent — backend will log */ }
    finally { setBusy(false); }
  }

  async function toggleOnAir() {
    const next = !onAir;
    setOnAir(next);
    try {
      const fd = new FormData();
      fd.append('on', String(next));
      await fetch(`${API_BASE}/api/director/on_air`, {
        method: 'POST', body: fd,
      });
    } catch {
      // Revert optimistic update if backend is unreachable.
      setOnAir(!next);
    }
  }

  const currentIdx = PHASES.indexOf(stage);

  return (
    <div style={styles.container}>
      {/* Inline StageBar — the mockup's phase pill chain */}
      <div style={styles.stageBar}>
        {PHASES.map((p, i) => {
          const active = i === currentIdx;
          const done = i < currentIdx;
          return (
            <React.Fragment key={p}>
              <div style={{
                ...styles.phasePill,
                background: active ? '#1d1d1f' : 'transparent',
                color: active ? '#fff' : (done ? '#1d1d1f' : '#86868b'),
                border: active ? 'none' : '1px solid rgba(0,0,0,0.08)',
              }}>
                <span style={{
                  ...styles.phaseDot,
                  background: active ? '#22c55e' : (done ? '#1d1d1f' : 'transparent'),
                  border: !done && !active ? '1px solid #86868b' : 'none',
                }} />
                {p}
              </div>
              {i < PHASES.length - 1 && (
                <div style={{
                  ...styles.phaseConnector,
                  background: done ? '#1d1d1f' : 'rgba(0,0,0,0.08)',
                }} />
              )}
            </React.Fragment>
          );
        })}
      </div>

      <div style={{ flex: 1 }} />

      {/* Transport: prev / play / next. Cosmetic in v1. */}
      <div style={styles.transport}>
        {[
          { label: '⏮', idx: 0 },
          { label: '⏸', idx: 1 },
          { label: '⏭', idx: 2 },
        ].map((b, i) => (
          <button key={b.label} type="button" style={{
            ...styles.transportButton,
            background: b.idx === 1 ? '#fff' : 'transparent',
            boxShadow: b.idx === 1 ? '0 1px 2px rgba(0,0,0,0.08)' : 'none',
          }}>
            {b.label}
          </button>
        ))}
      </div>

      <div style={styles.divider} />

      <span style={styles.forceLabel}>FORCE</span>
      {PHASES.map(p => (
        <button
          key={p}
          type="button"
          onClick={() => forcePhase(p)}
          disabled={busy}
          style={{
            ...styles.forcePill,
            opacity: busy ? 0.5 : 1,
          }}
        >
          {p}
        </button>
      ))}

      <div style={styles.divider} />

      {supportedLangs && Object.keys(supportedLangs).length > 1 && (
        <>
          <select
            value={lang}
            onChange={changeLang}
            style={styles.langSelect}
            title="Translation target — pitch text is Claude-translated before ElevenLabs TTS"
          >
            {Object.entries(supportedLangs).map(([code, meta]) => (
              <option key={code} value={code}>{code.toUpperCase()} · {meta.name}</option>
            ))}
          </select>
          <div style={styles.divider} />
        </>
      )}

      <button type="button" onClick={toggleOnAir} style={{
        ...styles.onAirButton,
        background: onAir ? '#1d1d1f' : '#fff',
        color: onAir ? '#fff' : '#1d1d1f',
      }}>
        <span style={{
          width: 7, height: 7, borderRadius: '50%',
          background: onAir ? '#22c55e' : '#86868b',
        }} />
        {onAir ? 'On Air' : 'Off Air'}
      </button>
    </div>
  );
}

const styles = {
  container: {
    display: 'flex', alignItems: 'center', gap: 10,
    padding: '10px 14px',
    background: '#fff',
    border: '1px solid rgba(0,0,0,0.08)',
    borderRadius: 12,
    fontFamily: '-apple-system, "SF Pro Text", "Inter", system-ui, sans-serif',
  },
  stageBar: {
    display: 'flex', alignItems: 'center', gap: 10,
  },
  phasePill: {
    display: 'flex', alignItems: 'center', gap: 6,
    padding: '5px 10px', borderRadius: 999,
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    fontSize: 10, fontWeight: 600, letterSpacing: 0.6,
    transition: 'all 220ms cubic-bezier(0.2,0.8,0.2,1)',
  },
  phaseDot: { width: 6, height: 6, borderRadius: '50%' },
  phaseConnector: { width: 14, height: 1 },
  transport: {
    display: 'inline-flex', alignItems: 'center', gap: 4,
    padding: 4, background: '#f5f5f7', borderRadius: 10,
  },
  transportButton: {
    width: 30, height: 26, borderRadius: 7, border: 'none',
    fontSize: 11, color: '#1d1d1f', cursor: 'pointer',
  },
  divider: { width: 1, height: 18, background: 'rgba(0,0,0,0.08)' },
  forceLabel: {
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    fontSize: 10, color: '#86868b', letterSpacing: 0.6,
  },
  forcePill: {
    padding: '5px 9px',
    border: '1px solid rgba(0,0,0,0.08)',
    borderRadius: 7, background: '#fff',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    fontSize: 10, fontWeight: 600, letterSpacing: 0.6,
    color: '#1d1d1f', cursor: 'pointer',
  },
  langSelect: {
    background: '#fff', color: '#1d1d1f',
    border: '1px solid rgba(0,0,0,0.08)', borderRadius: 7,
    padding: '5px 8px', fontSize: 11, fontWeight: 500,
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    cursor: 'pointer', outline: 'none',
    letterSpacing: 0.4,
  },
  onAirButton: {
    display: 'inline-flex', alignItems: 'center', gap: 6,
    padding: '7px 14px', borderRadius: 999, border: 'none',
    fontSize: 12, fontWeight: 500, cursor: 'pointer',
    fontFamily: 'inherit',
  },
};
