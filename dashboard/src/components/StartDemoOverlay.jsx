import React, { useEffect, useRef, useState } from 'react';

/**
 * StartDemoOverlay — one-tap autoplay-unlock ceremony.
 *
 * The audio-first build creates `<audio>` elements at WebSocket-message
 * time (e.g. when comment_response_audio fires). Browser autoplay policy
 * blocks `.play()` calls on elements that exist before any user gesture
 * has touched the document. The existing LiveStage unmute hook only
 * targets the Tier 1 video element refs — a brand-new <audio> created
 * later is muted-by-policy.
 *
 * The fix is REVISIONS §3 from the design doc: open the demo behind a
 * "Start Demo" button. The button click is the user gesture. We `play()`
 * a 100ms silent MP3 inside the synchronous click handler to bank the
 * unlock. Once banked, every subsequent <audio>.play() call from any WS
 * handler succeeds without rejection.
 *
 * One-time cost: a single click before the demo starts. After that the
 * overlay is gone forever for the session.
 *
 * The component takes `onStart` so the parent can do its own boot work
 * (e.g. focus the mic, show the keyboard hint, send a "stage_ready" WS
 * message) once the user has acknowledged the demo is starting.
 */

const API_BASE = `http://${window.location.hostname}:8000`;
const STORAGE_KEY = 'zo:demo-unlocked';

export function StartDemoOverlay({ onStart }) {
  // Persist the unlock across page reloads in the same session — operators
  // refresh the dashboard mid-rehearsal more often than they think, and
  // the second prompt reads as broken UX. sessionStorage clears when the
  // tab closes so a fresh judge session still gets the deliberate ceremony.
  const [done, setDone] = useState(() => {
    try { return sessionStorage.getItem(STORAGE_KEY) === '1'; }
    catch { return false; }
  });
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const audioRef = useRef(null);

  useEffect(() => {
    // Pre-instantiate (without play()) so the silent file is preloaded
    // by the time the operator clicks. Saves ~50ms on slow connections.
    const a = new Audio();
    a.src = `${API_BASE}/static/silent_unlock.mp3`;
    a.preload = 'auto';
    audioRef.current = a;
  }, []);

  // If the parent component already considers us done (e.g. parent persisted
  // its own flag), make sure onStart fires once so the parent's downstream
  // setup runs. Using a ref guard so we only fire once per mount.
  const onStartFiredRef = useRef(false);
  useEffect(() => {
    if (done && !onStartFiredRef.current) {
      onStartFiredRef.current = true;
      onStart?.();
    }
  }, [done, onStart]);

  async function handleStart() {
    if (busy) return;
    setBusy(true);
    setErr(null);
    try {
      // The user gesture: this click. Browser will permit .play() inside
      // the synchronous descendant of this handler. We play, immediately
      // pause, and reset to t=0 so nothing is audible.
      const a = audioRef.current;
      if (a) {
        await a.play();
        a.pause();
        a.currentTime = 0;
      }
      try { sessionStorage.setItem(STORAGE_KEY, '1'); } catch {}
      setDone(true);
      onStart?.();
    } catch (e) {
      // Common cases: user has system audio muted at OS level, or browser
      // refuses for unrelated reasons. We let them through anyway — the
      // unlock failure mostly only hurts the karaoke <audio> path; video
      // tier-1 already has its own click-anywhere unmute hook in LiveStage.
      console.warn('[start-demo] silent unlock failed (continuing):', e);
      setErr(e?.message || String(e));
      try { sessionStorage.setItem(STORAGE_KEY, '1'); } catch {}
      setDone(true);
      onStart?.();
    } finally {
      setBusy(false);
    }
  }

  if (done) return null;

  return (
    <div style={styles.overlay} onClick={handleStart} role="dialog" aria-label="Start demo">
      <div style={styles.card}>
        <div style={styles.eyebrow}>READY</div>
        <h1 style={styles.title}>Zo</h1>
        <p style={styles.sub}>
          Live commerce, on-device.<br />
          Click anywhere to start the demo.
        </p>
        <button
          type="button"
          style={styles.btn}
          onClick={(e) => { e.stopPropagation(); handleStart(); }}
          disabled={busy}
          autoFocus
        >
          {busy ? 'Starting…' : 'Start Demo'}
        </button>
        <div style={styles.foot}>
          One click banks audio playback for the rest of the session.
        </div>
        {err && (
          <div style={styles.errChip}>
            audio unlock had a hiccup ({err.slice(0, 64)}) — proceeding anyway
          </div>
        )}
      </div>
    </div>
  );
}

const styles = {
  overlay: {
    position: 'fixed', inset: 0, zIndex: 9999,
    background: 'radial-gradient(circle at 50% 40%, rgba(56,29,118,0.7), rgba(0,0,0,0.95))',
    backdropFilter: 'blur(12px)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    cursor: 'pointer',
    animation: 'startDemoFadeIn 220ms ease-out',
  },
  card: {
    background: 'rgba(9,9,11,0.92)',
    border: '1px solid #3f3f46',
    borderRadius: 20,
    padding: '40px 56px',
    textAlign: 'center',
    color: '#fafafa',
    maxWidth: 480,
    boxShadow: '0 30px 80px rgba(0,0,0,0.6)',
  },
  eyebrow: {
    fontSize: 11, fontWeight: 800, letterSpacing: 4,
    color: '#a78bfa', textTransform: 'uppercase',
    marginBottom: 12,
  },
  title: {
    fontSize: 64, fontWeight: 900, letterSpacing: 4,
    background: 'linear-gradient(135deg, #c4b5fd, #38bdf8)',
    WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
    margin: '0 0 18px',
    fontFamily: 'ui-sans-serif, system-ui, sans-serif',
  },
  sub: {
    fontSize: 16, color: '#a1a1aa', lineHeight: 1.5,
    margin: '0 0 28px',
  },
  btn: {
    background: 'linear-gradient(135deg, #7c3aed, #2563eb)',
    color: '#fff',
    border: 'none', borderRadius: 12,
    padding: '14px 36px', fontSize: 16, fontWeight: 800,
    letterSpacing: 1.5, cursor: 'pointer',
    boxShadow: '0 8px 24px rgba(124,58,237,0.45)',
    transition: 'transform 120ms ease',
  },
  foot: {
    marginTop: 24, fontSize: 11,
    color: '#52525b',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    letterSpacing: 0.5,
  },
  errChip: {
    marginTop: 16,
    background: 'rgba(120,53,15,0.6)',
    border: '1px solid #d97706',
    color: '#fbbf24',
    padding: '6px 12px',
    borderRadius: 6,
    fontSize: 11,
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  },
};

if (typeof document !== 'undefined' && !document.getElementById('start-demo-keyframes')) {
  const s = document.createElement('style');
  s.id = 'start-demo-keyframes';
  s.innerHTML = `
    @keyframes startDemoFadeIn { from { opacity: 0 } to { opacity: 1 } }
  `;
  document.head.appendChild(s);
}
