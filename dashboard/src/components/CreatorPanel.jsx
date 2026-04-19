import React, { useState } from 'react';

/**
 * CreatorPanel — one-click CREATOR demo for the stage.
 *
 * POSTs a bundled sample photo (public/sample_wallet.jpg) to /api/creator/build
 * with include_3d=false, then renders the 3 returned photos in a grid + the
 * promo video inline. Stays visible in the sideCol below ProductPanel so the
 * operator can fire it mid-pitch as a visible "watch the laptop generate
 * marketing assets in real time" beat.
 *
 * include_3d=false is intentional — TripoSR adds 15-30s and doesn't render
 * well on a dashboard anyway. Photos + promo are the demo-critical pieces.
 *
 * Second click in the same session reuses rembg warmup + ffmpeg cache — target
 * is ~2s for the re-build. First cold build is ~3-5s.
 */
const API_BASE = `http://${window.location.hostname}:8000`;
const SAMPLE_PHOTO = '/sample_wallet.jpg';

export function CreatorPanel() {
  const [buildState, setBuildState] = useState({ kind: 'idle' });

  async function onBuild() {
    const t0 = Date.now();
    setBuildState({ kind: 'loading' });
    try {
      const photoResp = await fetch(SAMPLE_PHOTO);
      if (!photoResp.ok) {
        throw new Error(`sample_wallet.jpg missing — HTTP ${photoResp.status}`);
      }
      const blob = await photoResp.blob();

      const fd = new FormData();
      fd.append('file', blob, 'sample_wallet.jpg');
      fd.append('include_3d', 'false');

      const res = await fetch(`${API_BASE}/api/creator/build`, {
        method: 'POST',
        body: fd,
      });
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`${res.status}: ${errText.slice(0, 200)}`);
      }
      const data = await res.json();
      setBuildState({ kind: 'done', data, clientElapsed: Date.now() - t0 });
    } catch (err) {
      setBuildState({ kind: 'error', message: String(err.message || err) });
    }
  }

  const isLoading = buildState.kind === 'loading';

  return (
    <div style={styles.container}>
      <div style={styles.headerRow}>
        <h3 style={styles.title}>CREATOR</h3>
        <span style={styles.badge}>v0 · laptop</span>
      </div>

      <button
        type="button"
        onClick={onBuild}
        disabled={isLoading}
        style={{
          ...styles.button,
          opacity: isLoading ? 0.6 : 1,
          cursor: isLoading ? 'wait' : 'pointer',
        }}
      >
        {isLoading ? 'BUILDING…' : '▶ BUILD 3 PHOTOS + 15s PROMO'}
      </button>

      {buildState.kind === 'loading' && (
        <div style={styles.hint}>
          ~3-5s cold · rembg + PIL + ffmpeg on laptop CPU
        </div>
      )}

      {buildState.kind === 'error' && (
        <div style={styles.error}>
          Failed: {buildState.message}
        </div>
      )}

      {buildState.kind === 'done' && buildState.data && (
        <div style={styles.output}>
          <div style={styles.timingPill}>
            {buildState.data.timing_ms.total}ms backend ·
            {' '}{buildState.data.photos.length} photos + promo
          </div>
          <div style={styles.photoGrid}>
            {buildState.data.photos.map((url, i) => (
              <img
                key={i}
                src={`${API_BASE}${url}`}
                alt={`CREATOR photo ${i + 1}`}
                style={styles.photo}
                loading="lazy"
              />
            ))}
          </div>
          <video
            src={`${API_BASE}${buildState.data.promo_video}`}
            controls
            muted
            autoPlay
            loop
            playsInline
            style={styles.video}
          />
        </div>
      )}
    </div>
  );
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
  button: {
    background: 'linear-gradient(135deg, #22c55e, #3b82f6)',
    color: '#09090b', fontWeight: 800,
    border: 'none', borderRadius: 8,
    padding: '14px 20px', fontSize: 13,
    letterSpacing: 1.5,
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  },
  hint: {
    fontSize: 10, color: '#71717a',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    textAlign: 'center', letterSpacing: 0.5,
  },
  error: {
    fontSize: 11, color: '#ef4444',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    padding: 8, background: 'rgba(239,68,68,0.08)',
    border: '1px solid rgba(239,68,68,0.3)',
    borderRadius: 6,
  },
  output: {
    display: 'flex', flexDirection: 'column', gap: 8,
    overflowY: 'auto', minHeight: 0,
  },
  timingPill: {
    fontSize: 10, fontWeight: 800, letterSpacing: 1,
    color: '#22c55e',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    textAlign: 'center', padding: '2px 0',
  },
  photoGrid: {
    display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 6,
  },
  photo: {
    width: '100%', aspectRatio: '1 / 1', objectFit: 'cover',
    borderRadius: 6, border: '1px solid #27272a', background: '#09090b',
  },
  video: {
    width: '100%', maxHeight: 220,
    borderRadius: 8, background: '#000',
  },
};
