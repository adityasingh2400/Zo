import React, { useEffect, useRef, useState } from 'react';
import { Spin3D } from './Spin3D';
import { HeroGallery } from './HeroGallery';

const API_BASE = `http://${window.location.hostname}:8000`;

/**
 * CarouselTester — dedicated page for iterating on the 3D extraction pipeline.
 * Hits ONLY /api/build_carousel — no Claude, no TTS, no avatar, no WebSocket.
 *
 * Drop a video → see the carousel render with the new shader pipeline + every
 * orbit-quality stat from the backend. Tweak n_frames / out_size / rembg_model
 * inline so you can A/B without restarting.
 */
export function CarouselTester() {
  const [view, setView] = useState(null);          // { kind, frames, stats, ... } from /api/build_carousel
  const [stats, setStats] = useState(null);
  const [params, setParams] = useState({
    // Bumped from 36 → 48 — the carousel_from_video function default. The
    // first test render came back fast enough that there's headroom to
    // capture more angles without exceeding the 5-6s demo budget.
    n_frames: 48,
    out_size: 1024,
    clean_bg: true,
    // Switched default from u2net → isnet-general-use. Newer (2022 vs 2020),
    // generally better edge fidelity on small product details (logos,
    // textures, jewelry), comparable speed since both models internally
    // resize to 320x320. ~170MB one-time download — backend prewarms it
    // at startup alongside u2net so the first render isn't paying for it.
    rembg_model: 'isnet-general-use',
    stabilize: true,
    remove_skin: false,
    keep_central: true,
    // Hard trim of seconds from the head/tail of the video BEFORE frames
    // are extracted. Predictable + explicit — the user-preferred way to
    // handle "operator filmed the product but panned onto the MacBook
    // in the last 1-2 seconds". Default tail trim of 1.0s catches that
    // common case without accidentally killing real product frames.
    // Future clean videos: dial both to 0.
    trim_head_seconds: 0.0,
    trim_tail_seconds: 1.0,
    // Auto-detection of a wandering subject (different mean color/coverage
    // than the median) — kept available but defaulted OFF in favor of the
    // explicit trim above. Flip on for belt-and-suspenders if a video has
    // unpredictable bad frames mid-shot, not just at the ends.
    subject_continuity: false,
  });
  const [busy, setBusy] = useState(false);
  const [busyMs, setBusyMs] = useState(0);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);     // [{name, view, stats, ms}]
  const [dragging, setDragging] = useState(false);
  const [spinState, setSpinState] = useState('idle');
  const [productLabel, setProductLabel] = useState('Test product');
  const [theme, setTheme] = useState('studio_light');
  const fileInputRef = useRef(null);
  const busyTimerRef = useRef(null);

  // Tick a millisecond counter while a render is in flight so the UI feels alive.
  useEffect(() => {
    if (!busy) {
      if (busyTimerRef.current) clearInterval(busyTimerRef.current);
      return;
    }
    const t0 = performance.now();
    busyTimerRef.current = setInterval(() => setBusyMs(performance.now() - t0), 100);
    return () => clearInterval(busyTimerRef.current);
  }, [busy]);

  async function uploadFile(file) {
    if (!file) return;
    setError(null);
    setBusy(true);
    setBusyMs(0);
    setStats({ status: 'uploading', filename: file.name, size_mb: (file.size / 1e6).toFixed(2) });
    try {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('n_frames', String(params.n_frames));
      fd.append('out_size', String(params.out_size));
      fd.append('clean_bg', String(params.clean_bg));
      fd.append('rembg_model', params.rembg_model);
      fd.append('stabilize', String(params.stabilize));
      fd.append('remove_skin', String(params.remove_skin));
      fd.append('keep_central', String(params.keep_central));
      fd.append('subject_continuity', String(params.subject_continuity));
      fd.append('trim_head_seconds', String(params.trim_head_seconds));
      fd.append('trim_tail_seconds', String(params.trim_tail_seconds));
      const t0 = performance.now();
      const res = await fetch(`${API_BASE}/api/build_carousel`, { method: 'POST', body: fd });
      const elapsed = Math.round(performance.now() - t0);
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`HTTP ${res.status}: ${txt.slice(0, 200)}`);
      }
      const data = await res.json();
      setView(data);
      setStats({ ...data.stats, timings: data.timings, ms: data.ms, slug: data.slug,
                 cached: data.cached, frames: data.frames?.length, request_ms: elapsed });
      setHistory(h => [{ name: file.name, view: data, stats: data.stats, ms: data.ms, t: Date.now() },
                       ...h].slice(0, 8));
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  }

  function onDrop(e) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
  }

  return (
    <div
      style={{ ...styles.app, ...(dragging ? styles.appDragging : {}) }}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      {dragging && (
        <div style={styles.dropOverlay}>
          <div style={{ fontSize: 80 }}>🎬</div>
          <div style={{ fontSize: 22, fontWeight: 700 }}>Drop video to rebuild the carousel</div>
        </div>
      )}

      {/* Header */}
      <header style={styles.header}>
        <h1 style={styles.logo}>CAROUSEL TESTER</h1>
        <div style={styles.subtitle}>
          isolated <code style={styles.code}>/api/build_carousel</code> — no Claude, no TTS, no avatar
        </div>
      </header>

      <div style={styles.grid}>
        {/* Carousel viewer */}
        <section style={styles.viewerCard}>
          <div style={styles.cardHeader}>
            <span style={styles.cardLabel}>RENDER</span>
            <div style={styles.cardHeaderRight}>
              <button
                onClick={() => fileInputRef.current?.click()}
                style={styles.uploadBtn}
                disabled={busy}
              >
                {busy ? `building… ${(busyMs / 1000).toFixed(1)}s` : '🎬 Upload video'}
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                style={{ display: 'none' }}
                onChange={(e) => { const f = e.target.files[0]; if (f) uploadFile(f); }}
              />
            </div>
          </div>
          <div style={styles.viewerStage}>
            {view?.frames?.length ? (
              <div style={styles.viewerStack}>
                {view.heroes?.length > 0 && (
                  <HeroGallery
                    heroes={view.heroes}
                    heroMeta={view.hero_meta || []}
                    theme={theme}
                  />
                )}
                <div style={styles.spinSquare}>
                  <Spin3D
                    view={view}
                    height={'100%'}
                    label={productLabel || undefined}
                    state={spinState}
                    accent="#7c3aed"
                    theme={theme}
                  />
                </div>
              </div>
            ) : (
              <div style={styles.viewerEmpty}>
                <div style={{ fontSize: 80, opacity: 0.4 }}>🎬</div>
                <div style={{ marginTop: 16, color: '#71717a', fontSize: 16 }}>
                  Drag a video here or click upload
                </div>
                <div style={{ marginTop: 6, color: '#52525b', fontSize: 13 }}>
                  Best: 1080p · 12-15s slow orbit · plain background · bright light
                </div>
              </div>
            )}
          </div>
          {view?.frames?.length > 0 && (
            <>
              <div style={styles.stateRow}>
                <span style={{ color: '#71717a', fontSize: 11 }}>THEME</span>
                {['studio_light', 'studio_dark'].map(t => (
                  <button
                    key={t}
                    onClick={() => setTheme(t)}
                    style={{
                      ...styles.stateBtn,
                      background: theme === t ? '#7c3aed' : '#27272a',
                      color: theme === t ? '#fff' : '#a1a1aa',
                    }}
                  >
                    {t.replace('studio_', '')}
                  </button>
                ))}
                <span style={{ color: '#52525b', fontSize: 11, marginLeft: 'auto' }}>
                  {theme === 'studio_light' ? 'white seamless · floor reflection' : 'dark stage · rim light'}
                </span>
              </div>
              <div style={styles.stateRow}>
                <span style={{ color: '#71717a', fontSize: 11 }}>STATE</span>
                {['idle', 'listening', 'thinking', 'responding'].map(s => (
                  <button
                    key={s}
                    onClick={() => setSpinState(s)}
                    style={{
                      ...styles.stateBtn,
                      background: spinState === s ? '#7c3aed' : '#27272a',
                      color: spinState === s ? '#fff' : '#a1a1aa',
                    }}
                  >
                    {s}
                  </button>
                ))}
                <input
                  type="text"
                  value={productLabel}
                  onChange={(e) => setProductLabel(e.target.value)}
                  placeholder="Product label"
                  style={styles.labelInput}
                />
              </div>
            </>
          )}
        </section>

        {/* Right column: params + stats + history */}
        <section style={styles.sidebar}>
          {/* Params */}
          <div style={styles.card}>
            <div style={styles.cardLabel}>PARAMS</div>
            <div style={styles.paramsGrid}>
              <Param label="n_frames" type="number" value={params.n_frames} min={6} max={48}
                     onChange={(v) => setParams(p => ({ ...p, n_frames: parseInt(v) || 24 }))} />
              <Param label="out_size" type="number" value={params.out_size} min={256} max={1024} step={64}
                     onChange={(v) => setParams(p => ({ ...p, out_size: parseInt(v) || 640 }))} />
              <Param label="rembg_model" type="select" value={params.rembg_model}
                     options={['u2net', 'u2netp', 'isnet-general-use']}
                     onChange={(v) => setParams(p => ({ ...p, rembg_model: v }))} />
              <Param label="stabilize" type="checkbox" value={params.stabilize}
                     onChange={(v) => setParams(p => ({ ...p, stabilize: v }))} />
              <Param label="clean_bg" type="checkbox" value={params.clean_bg}
                     onChange={(v) => setParams(p => ({ ...p, clean_bg: v }))} />
              <Param label="remove_skin" type="checkbox" value={params.remove_skin}
                     onChange={(v) => setParams(p => ({ ...p, remove_skin: v }))} />
              <Param label="keep_central" type="checkbox" value={params.keep_central}
                     onChange={(v) => setParams(p => ({ ...p, keep_central: v }))} />
              <Param label="trim_head_seconds" type="number" value={params.trim_head_seconds}
                     min={0} max={5} step={0.5}
                     onChange={(v) => setParams(p => ({ ...p, trim_head_seconds: parseFloat(v) || 0 }))} />
              <Param label="trim_tail_seconds" type="number" value={params.trim_tail_seconds}
                     min={0} max={5} step={0.5}
                     onChange={(v) => setParams(p => ({ ...p, trim_tail_seconds: parseFloat(v) || 0 }))} />
              <Param label="subject_continuity" type="checkbox" value={params.subject_continuity}
                     onChange={(v) => setParams(p => ({ ...p, subject_continuity: v }))} />
            </div>
            <div style={styles.paramsHint}>
              <strong style={{color:'#a1a1aa'}}>trim_head/tail_seconds</strong>{' '}
              chops time off either end of the video before frame extraction.
              Use for "operator panned onto the MacBook at the end" (1.0s default).
              Set both 0 for clean videos that don't need trimming.{' '}
              <strong style={{color:'#a1a1aa'}}>keep_central</strong> drops
              notebook / stand props from the alpha — leave on for shoots where
              the product sits on something.{' '}
              <strong style={{color:'#a1a1aa'}}>remove_skin</strong> kills hand
              pixels (don't enable for tan-leather products).{' '}
              <strong style={{color:'#a1a1aa'}}>subject_continuity</strong>{' '}
              auto-detects wandering subjects (off by default in favor of explicit trim).
            </div>
          </div>

          {/* Stats */}
          <div style={styles.card}>
            <div style={styles.cardLabel}>SHOOT QUALITY</div>
            {stats ? <StatsTable stats={stats} /> : (
              <div style={{ color: '#52525b', fontSize: 13, padding: 14 }}>
                Upload a video to see orbit stats.
              </div>
            )}
          </div>

          {/* Error */}
          {error && (
            <div style={{ ...styles.card, borderColor: '#7f1d1d', background: 'rgba(127,29,29,0.15)' }}>
              <div style={{ ...styles.cardLabel, color: '#fca5a5' }}>ERROR</div>
              <div style={{ padding: 14, color: '#fecaca', fontFamily: 'ui-monospace, monospace', fontSize: 12 }}>
                {error}
              </div>
            </div>
          )}

          {/* History */}
          {history.length > 0 && (
            <div style={styles.card}>
              <div style={styles.cardLabel}>HISTORY</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4, padding: 8 }}>
                {history.map((h, i) => (
                  <button
                    key={`${h.t}_${i}`}
                    onClick={() => { setView(h.view); setStats({ ...h.stats, ms: h.ms, frames: h.view.frames?.length }); }}
                    style={styles.historyRow}
                  >
                    <span style={{ flex: 1, fontSize: 12, color: '#e4e4e7', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {h.name}
                    </span>
                    <span style={{ fontSize: 11, color: '#71717a', fontFamily: 'ui-monospace, monospace' }}>
                      {h.stats?.kept || 0}f · {(h.ms / 1000).toFixed(1)}s
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </section>
      </div>

      <footer style={styles.footer}>
        Drop video anywhere · Carousel uses the WebGL shader · Backend: <code style={styles.code}>{API_BASE}</code>
      </footer>
    </div>
  );
}

function Param({ label, type, value, options, onChange, ...rest }) {
  if (type === 'checkbox') {
    return (
      <label style={styles.paramRow}>
        <span style={styles.paramLabel}>{label}</span>
        <input type="checkbox" checked={value} onChange={(e) => onChange(e.target.checked)} />
      </label>
    );
  }
  if (type === 'select') {
    return (
      <label style={styles.paramRow}>
        <span style={styles.paramLabel}>{label}</span>
        <select value={value} onChange={(e) => onChange(e.target.value)} style={styles.select}>
          {options.map(o => <option key={o} value={o}>{o}</option>)}
        </select>
      </label>
    );
  }
  return (
    <label style={styles.paramRow}>
      <span style={styles.paramLabel}>{label}</span>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={styles.input}
        {...rest}
      />
    </label>
  );
}

function StatsTable({ stats }) {
  // Some stats from /api/build_carousel; some derived. Display the orbit ones
  // prominently with health colors.
  const orbit = stats.orbit || {};
  const kept = stats.kept;
  const cands = stats.candidates;
  const dropped = stats.dropped;
  const keptPct = (kept != null && cands) ? (100 * kept / cands) : null;

  function band(value, good, ok) {
    if (value == null) return '#71717a';
    if (value <= good) return '#22c55e';
    if (value <= ok) return '#fbbf24';
    return '#ef4444';
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', padding: 4 }}>
      <Row label="frames kept" value={`${kept ?? '—'}/${cands ?? '—'}`}
           hint={keptPct != null ? `${keptPct.toFixed(0)}%` : ''}
           color={keptPct == null ? '#fafafa' : band(-keptPct, -80, -50)} />
      <Row label="dropped" value={`${dropped ?? 0}`} />
      {stats.subject_outliers_dropped > 0 && (
        <Row
          label="  ↳ wrong subject"
          value={`${stats.subject_outliers_dropped}`}
          color="#fbbf24"
          hint={
            stats.subject_outlier_debug?.dropped_indices?.length > 0
              ? `frames ${stats.subject_outlier_debug.dropped_indices.join(',')}`
              : 'camera wandered off the product'
          }
        />
      )}
      {(stats.trim_head_seconds > 0 || stats.trim_tail_seconds > 0) && (
        <Row
          label="trim window"
          value={`-${stats.trim_head_seconds}s / -${stats.trim_tail_seconds}s`}
          color="#a78bfa"
          hint={`${stats.video_duration_sec}s → ${stats.effective_duration_sec}s effective`}
        />
      )}
      <Row label="rembg model" value={stats.rembg_model || '—'} />
      <Row label="median crop side" value={`${orbit.median_side_px ?? '—'} px`} />
      <Row label="center drift" value={`${orbit.center_stddev_px ?? '—'} px`}
           hint={orbit.center_stddev_px == null ? '' :
             orbit.center_stddev_px < 20 ? 'rock steady' :
             orbit.center_stddev_px < 80 ? 'normal handheld' : 'shaky — try slower orbit'}
           color={band(orbit.center_stddev_px, 20, 80)} />
      <Row label="size drift" value={`${orbit.size_stddev_px ?? '—'} px`}
           hint={orbit.size_stddev_px == null ? '' :
             orbit.size_stddev_px < 30 ? 'consistent distance' :
             orbit.size_stddev_px < 100 ? 'mild walk-in' : 'big distance change — fix radius'}
           color={band(orbit.size_stddev_px, 30, 100)} />
      <Row label="clipped frames" value={`${orbit.clipped_frames ?? 0}`}
           hint={orbit.clipped_frames > 0 ? 'product near edge — pull camera back' : 'all in frame'}
           color={band(orbit.clipped_frames, 2, 5)} />
      <hr style={styles.hr} />
      <Row label="total render" value={`${stats.ms ? (stats.ms / 1000).toFixed(2) : '—'}s`} />
      {stats.timings && (
        <>
          <Row label="  ffmpeg" value={`${(stats.timings.extract_ms / 1000).toFixed(2)}s`} dim />
          <Row label="  pick sharpest" value={`${(stats.timings.pick_ms / 1000).toFixed(2)}s`} dim />
          <Row label="  rembg + crop" value={`${(stats.timings.process_ms / 1000).toFixed(2)}s`} dim />
        </>
      )}
      {stats.cached && <Row label="cache" value="HIT" color="#22c55e" />}
      {stats.slug && <Row label="slug" value={stats.slug.slice(0, 10)} dim />}
    </div>
  );
}

function Row({ label, value, hint, color, dim }) {
  return (
    <div style={styles.statsRow}>
      <span style={{ ...styles.statsLabel, opacity: dim ? 0.6 : 1 }}>{label}</span>
      <span style={{
        ...styles.statsValue,
        color: color || '#fafafa',
        opacity: dim ? 0.7 : 1,
      }}>
        {value}
      </span>
      {hint && <span style={styles.statsHint}>{hint}</span>}
    </div>
  );
}

const styles = {
  app: {
    minHeight: '100vh',
    padding: 24,
    display: 'flex', flexDirection: 'column', gap: 18,
    maxWidth: 1500, margin: '0 auto',
  },
  appDragging: { outline: '3px dashed #7c3aed', outlineOffset: -3 },
  dropOverlay: {
    position: 'fixed', inset: 0, background: 'rgba(124,58,237,0.18)',
    display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
    gap: 14, zIndex: 999, backdropFilter: 'blur(6px)',
  },
  header: { display: 'flex', flexDirection: 'column', gap: 4 },
  logo: {
    fontSize: 28, fontWeight: 900, letterSpacing: 4, margin: 0,
    background: 'linear-gradient(135deg, #7c3aed, #3b82f6)',
    WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
  },
  subtitle: { color: '#71717a', fontSize: 13 },
  code: {
    background: '#18181b', border: '1px solid #27272a', borderRadius: 4,
    padding: '2px 6px', fontFamily: 'ui-monospace, monospace', fontSize: 11, color: '#a1a1aa',
  },
  codeInline: {
    background: 'rgba(124,58,237,0.18)', borderRadius: 3, padding: '1px 5px',
    fontFamily: 'ui-monospace, monospace', fontSize: 11, color: '#c4b5fd',
  },
  grid: {
    display: 'grid', gap: 16,
    gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 1fr)',
  },
  viewerCard: {
    background: '#0f0f12', border: '1px solid #27272a', borderRadius: 14,
    display: 'flex', flexDirection: 'column', overflow: 'hidden',
  },
  cardHeader: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '12px 16px', borderBottom: '1px solid #27272a',
  },
  cardLabel: {
    color: '#a1a1aa', fontSize: 11, fontWeight: 800, letterSpacing: 2,
  },
  cardHeaderRight: { display: 'flex', alignItems: 'center', gap: 8 },
  uploadBtn: {
    background: '#7c3aed', color: '#fff', border: 'none',
    padding: '8px 16px', borderRadius: 8, fontWeight: 700, fontSize: 13,
    cursor: 'pointer',
  },
  viewerStage: {
    padding: 16, display: 'flex',
    alignItems: 'center', justifyContent: 'center',
  },
  viewerStack: {
    display: 'flex', flexDirection: 'column', gap: 16,
    width: '100%', maxWidth: 720,
  },
  spinSquare: {
    width: '100%', aspectRatio: '1 / 1',
    display: 'flex', alignItems: 'stretch', justifyContent: 'stretch',
  },
  viewerEmpty: {
    flex: 1, display: 'flex', flexDirection: 'column',
    alignItems: 'center', justifyContent: 'center',
    background: 'radial-gradient(ellipse at center, #18181b 0%, #09090b 100%)',
    border: '1px dashed #27272a', borderRadius: 10,
  },
  stateRow: {
    display: 'flex', alignItems: 'center', gap: 6, padding: '10px 16px',
    borderTop: '1px solid #27272a', flexWrap: 'wrap',
  },
  stateBtn: {
    border: 'none', borderRadius: 6, padding: '5px 10px',
    fontSize: 11, fontWeight: 700, letterSpacing: 1, cursor: 'pointer',
    textTransform: 'uppercase',
  },
  labelInput: {
    marginLeft: 'auto',
    background: '#18181b', border: '1px solid #27272a', borderRadius: 6,
    padding: '5px 10px', fontSize: 12, color: '#fafafa', outline: 'none',
    width: 200,
  },
  sidebar: { display: 'flex', flexDirection: 'column', gap: 12 },
  card: {
    background: '#0f0f12', border: '1px solid #27272a', borderRadius: 12,
    overflow: 'hidden',
  },
  paramsGrid: {
    display: 'flex', flexDirection: 'column', padding: 8,
  },
  paramRow: {
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '6px 8px', borderRadius: 6,
  },
  paramLabel: {
    flex: 1, fontSize: 12, color: '#a1a1aa', fontWeight: 600,
    fontFamily: 'ui-monospace, monospace',
  },
  input: {
    background: '#18181b', border: '1px solid #27272a', borderRadius: 6,
    padding: '4px 8px', color: '#fafafa', fontSize: 12, width: 80, outline: 'none',
  },
  select: {
    background: '#18181b', border: '1px solid #27272a', borderRadius: 6,
    padding: '4px 8px', color: '#fafafa', fontSize: 12, outline: 'none',
  },
  paramsHint: {
    padding: '8px 14px 12px', color: '#52525b', fontSize: 11, lineHeight: 1.5,
  },
  statsRow: {
    display: 'grid', gridTemplateColumns: 'minmax(110px, auto) 1fr auto',
    alignItems: 'baseline', gap: 8, padding: '5px 12px',
  },
  statsLabel: { color: '#71717a', fontSize: 11, fontWeight: 600 },
  statsValue: {
    color: '#fafafa', fontFamily: 'ui-monospace, monospace', fontSize: 13,
    fontWeight: 700, fontVariantNumeric: 'tabular-nums',
  },
  statsHint: { color: '#52525b', fontSize: 10, fontStyle: 'italic' },
  hr: { border: 'none', borderTop: '1px solid #27272a', margin: '6px 12px' },
  historyRow: {
    background: 'transparent', border: 'none', cursor: 'pointer',
    padding: '8px 10px', borderRadius: 6,
    display: 'flex', alignItems: 'center', gap: 12, color: '#fafafa',
    textAlign: 'left',
  },
  footer: {
    color: '#52525b', fontSize: 11, textAlign: 'center', padding: '8px 0',
  },
};
