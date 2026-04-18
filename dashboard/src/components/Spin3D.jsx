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
  const [idx, setIdx] = useState(0);
  const [paused, setPaused] = useState(false);
  const [loaded, setLoaded] = useState(0);
  const totalRef = useRef(frames.length);
  totalRef.current = frames.length;
  const containerRef = useRef(null);
  const dragStartRef = useRef(null);

  // Preload all frames before starting the spin so it never stalls mid-rotation.
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

  // Auto-rotate at ~12 fps, only after all frames are loaded
  useEffect(() => {
    if (paused) return;
    if (loaded < frames.length) return;
    const id = setInterval(() => {
      setIdx((i) => (i + 1) % totalRef.current);
    }, 1000 / 12);
    return () => clearInterval(id);
  }, [paused, loaded, frames.length]);

  // Drag-to-scrub: while dragging, map x-delta to frame index
  function onPointerDown(e) {
    setPaused(true);
    dragStartRef.current = { x: e.clientX, idx };
    e.currentTarget.setPointerCapture?.(e.pointerId);
  }
  function onPointerMove(e) {
    if (!dragStartRef.current) return;
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const dx = e.clientX - dragStartRef.current.x;
    const stride = rect.width / frames.length;
    const next = (dragStartRef.current.idx + Math.round(dx / stride)) % frames.length;
    setIdx((next + frames.length) % frames.length);
  }
  function onPointerUp(e) {
    dragStartRef.current = null;
    e.currentTarget.releasePointerCapture?.(e.pointerId);
    setTimeout(() => setPaused(false), 800); // brief pause before auto-resume
  }

  const ready = loaded >= frames.length;
  const url = `${API_BASE}${frames[idx]}`;

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
      {/* Render all frames absolutely-positioned, hide all but current.
          Browser holds them in cache so swaps are instant. */}
      {frames.map((f, i) => (
        <img
          key={f}
          src={`${API_BASE}${f}`}
          alt=""
          draggable={false}
          style={{ ...styles.frame, opacity: i === idx ? 1 : 0 }}
        />
      ))}
      <div style={styles.badge}>3D · {frames.length}</div>
      {!ready && (
        <div style={styles.loading}>
          loading {loaded}/{frames.length}
        </div>
      )}
      <div style={{ ...styles.scrubBar, opacity: paused ? 1 : 0 }}>
        {frames.map((_, i) => (
          <div
            key={i}
            style={{
              ...styles.scrubTick,
              background: i === idx ? '#7c3aed' : '#3f3f46',
            }}
          />
        ))}
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
    transition: 'opacity 60ms linear',
    pointerEvents: 'none',
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
