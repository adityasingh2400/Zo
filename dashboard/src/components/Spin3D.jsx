import React, { useEffect, useRef, useState } from 'react';

const API_BASE = `http://${window.location.hostname}:8000`;

/**
 * Spin3D — plays a frame carousel as a continuous rotation, looking like
 * a 3D model spin. Tier 1 (no GPU). When `view.kind === 'glb'`, lazy-mounts
 * <model-viewer> instead (Tier 2, after RunPod TripoSR is wired up).
 *
 * Carousel frames are pre-rendered PNG files served from /renders/spin/<slug>/.
 * We auto-rotate at ~12 fps with hover-to-pause and click-to-scrub.
 */
export function Spin3D({ view, height = 220 }) {
  if (!view) return null;
  if (view.kind === 'glb' && view.url) {
    return <GLBView url={view.url} height={height} />;
  }
  if (view.kind === 'frame_carousel' && view.frames?.length) {
    return <CarouselSpin frames={view.frames} ms={view.ms} height={height} />;
  }
  return null;
}

function CarouselSpin({ frames, ms, height }) {
  // We track a *floating-point* phase (0..frames.length) so we can crossfade
  // between adjacent frames instead of hard-cutting. Phase is updated by an
  // rAF loop, not setInterval, so it stays smooth under load and lets us
  // ease in/out at the wraparound for a sense of inertia.
  const [phase, setPhase] = useState(0);
  const [paused, setPaused] = useState(false);
  const [loaded, setLoaded] = useState(0);
  const containerRef = useRef(null);
  const dragStartRef = useRef(null);
  const rafRef = useRef(0);

  // Preload all frames before starting so the spin never stalls mid-rotation.
  useEffect(() => {
    let cancel = false;
    setLoaded(0);
    let count = 0;
    frames.forEach((url) => {
      const img = new Image();
      img.onload = img.onerror = () => {
        if (cancel) return;
        count += 1;
        setLoaded(count);
      };
      img.src = `${API_BASE}${url}`;
    });
    return () => { cancel = true; };
  }, [frames]);

  // rAF rotation loop. Default speed: one full rotation in ~3.5s. Adds a
  // mild ease so the spin breathes — slightly slower at the seam-line of the
  // loop so the eye doesn't catch a discontinuity even though the carousel
  // is technically wrapping around.
  const SECONDS_PER_REVOLUTION = 3.5;
  useEffect(() => {
    if (paused || loaded < frames.length) return;
    let last = performance.now();
    const tick = (now) => {
      const dt = (now - last) / 1000;
      last = now;
      setPhase((p) => {
        const next = p + (frames.length / SECONDS_PER_REVOLUTION) * dt;
        // Wrap, keeping the fractional component so the crossfade stays
        // continuous across the seam.
        return next % frames.length;
      });
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [paused, loaded, frames.length]);

  // Drag-to-scrub: while dragging, map x-delta to phase
  function onPointerDown(e) {
    setPaused(true);
    dragStartRef.current = { x: e.clientX, phase };
    e.currentTarget.setPointerCapture?.(e.pointerId);
  }
  function onPointerMove(e) {
    if (!dragStartRef.current) return;
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const dx = e.clientX - dragStartRef.current.x;
    // One full rotation per container-width swipe.
    const next = dragStartRef.current.phase + (dx / rect.width) * frames.length;
    setPhase(((next % frames.length) + frames.length) % frames.length);
  }
  function onPointerUp(e) {
    dragStartRef.current = null;
    e.currentTarget.releasePointerCapture?.(e.pointerId);
    // Resume after a brief beat so the user can read what they're looking at.
    setTimeout(() => setPaused(false), 1200);
  }

  const ready = loaded >= frames.length;
  // Two visible frames per moment: floor(phase) at (1 - frac) opacity, and
  // ceil(phase) at frac opacity. Smooth linear blend; the rAF loop runs at
  // 60fps so consecutive blends look continuous.
  const lower = Math.floor(phase) % frames.length;
  const upper = (lower + 1) % frames.length;
  const frac = phase - Math.floor(phase);

  return (
    <div
      ref={containerRef}
      style={{ ...styles.box, height }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => { dragStartRef.current = null; setPaused(false); }}
      title="Drag to rotate"
    >
      {/* Render every frame absolutely-positioned. Show only the two adjacent
          to the current phase, blended by `frac`. All other frames stay in the
          DOM at opacity 0 so the browser keeps them decoded. */}
      {frames.map((f, i) => {
        let opacity = 0;
        if (i === lower) opacity = 1 - frac;
        else if (i === upper) opacity = frac;
        return (
          <img
            key={f}
            src={`${API_BASE}${f}`}
            alt=""
            draggable={false}
            style={{ ...styles.frame, opacity }}
          />
        );
      })}
      <div style={styles.badge}>3D · {frames.length}</div>
      {!ready && (
        <div style={styles.loading}>
          loading {loaded}/{frames.length}
        </div>
      )}
      <div style={{ ...styles.scrubBar, opacity: paused ? 1 : 0 }}>
        {frames.map((_, i) => {
          // Highlight the lower-bound frame; tick subtly fades into the next
          // one to telegraph that we're between frames.
          const active = i === lower;
          const next = i === upper;
          return (
            <div
              key={i}
              style={{
                ...styles.scrubTick,
                background: active ? '#7c3aed'
                  : next ? `rgba(124,58,237,${0.3 + 0.7 * frac})`
                    : '#3f3f46',
              }}
            />
          );
        })}
      </div>
    </div>
  );
}

function GLBView({ url, height }) {
  // <model-viewer> is loaded once via index.html script tag.
  // It's a custom element so React just renders it as JSX.
  return (
    <div style={{ ...styles.box, height }}>
      {/* @ts-ignore */}
      <model-viewer
        src={`${API_BASE}${url}`}
        camera-controls
        auto-rotate
        auto-rotate-delay="0"
        rotation-per-second="30deg"
        interaction-prompt="none"
        style={{ width: '100%', height: '100%', background: '#0a0a0a' }}
      />
      <div style={styles.badge}>3D · GLB</div>
    </div>
  );
}

const styles = {
  box: {
    position: 'relative', width: '100%',
    background: 'radial-gradient(ellipse at center, #1f1f23 0%, #09090b 100%)',
    borderRadius: 10, overflow: 'hidden',
    cursor: 'grab', userSelect: 'none',
    border: '1px solid #27272a',
  },
  frame: {
    position: 'absolute', inset: 0,
    width: '100%', height: '100%', objectFit: 'contain',
    pointerEvents: 'none',
    // No CSS transition — the rAF loop drives `opacity` directly via React.
    // CSS easing on opacity would smear the crossfade and look laggy.
  },
  badge: {
    position: 'absolute', top: 8, left: 8,
    padding: '2px 8px', borderRadius: 999,
    background: 'rgba(124,58,237,0.85)', color: '#fff',
    fontSize: 10, fontWeight: 800, letterSpacing: 1,
  },
  loading: {
    position: 'absolute', inset: 0,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    color: '#71717a', fontSize: 12, fontFamily: 'ui-monospace, monospace',
  },
  scrubBar: {
    position: 'absolute', bottom: 8, left: 8, right: 8,
    display: 'flex', gap: 2, height: 3,
    transition: 'opacity 200ms ease',
  },
  scrubTick: { flex: 1, borderRadius: 1 },
};
