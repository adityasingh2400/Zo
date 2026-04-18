import React, { useState } from 'react';

const API_BASE = `http://${window.location.hostname}:8000`;

/**
 * HeroGallery — the "magazine cover" companion to the spinning carousel.
 *
 * Shows N hand-picked hero frames (one per rotation quadrant, scored by
 * sharpness + compactness + coverage) in a clean 2×N/2 grid. Each tile
 * uses the studio_light aesthetic via CSS: white card, subtle floor
 * shadow, soft hover lift. Clicking a tile opens it in a lightbox.
 *
 * The hero PNGs themselves are transparent — the white "stage" comes from
 * the card background, and the soft elliptical shadow under the product is
 * drawn via a CSS radial-gradient pseudo-element. This keeps the assets
 * themable: drop the same PNG on dark and the card flips dark.
 */
export function HeroGallery({ heroes = [], heroMeta = [], theme = 'studio_light' }) {
  const [active, setActive] = useState(null); // index in lightbox
  if (!heroes.length) return null;

  const isLight = theme === 'studio_light';
  const cardBg = isLight ? '#ffffff' : '#0f0f12';
  const cardBorder = isLight ? '1px solid #e4e4e7' : '1px solid #27272a';
  const cardShadow = isLight
    ? '0 1px 3px rgba(15,23,42,0.06), 0 8px 24px rgba(15,23,42,0.08)'
    : '0 1px 3px rgba(0,0,0,0.4), 0 8px 24px rgba(0,0,0,0.5)';
  const tagBg = isLight ? 'rgba(15,15,18,0.78)' : 'rgba(124,58,237,0.92)';

  return (
    <div style={styles.wrap}>
      <div style={styles.header}>
        <span style={styles.label}>HERO SHOTS</span>
        <span style={styles.sub}>
          best frames per rotation quadrant · 1536px · unsharp + studio comp
        </span>
      </div>

      <div style={styles.grid(heroes.length)}>
        {heroes.map((url, i) => {
          const meta = heroMeta[i] || {};
          return (
            <button
              key={url}
              onClick={() => setActive(i)}
              style={{
                ...styles.card,
                background: cardBg,
                border: cardBorder,
                boxShadow: cardShadow,
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = isLight
                  ? '0 4px 12px rgba(15,23,42,0.10), 0 16px 40px rgba(15,23,42,0.14)'
                  : '0 4px 12px rgba(0,0,0,0.5), 0 16px 40px rgba(0,0,0,0.6)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = cardShadow;
              }}
              aria-label={`Hero shot at ${meta.angle_deg ?? '?'}°`}
            >
              {/* The actual product image */}
              <img
                src={`${API_BASE}${url}`}
                alt=""
                style={styles.img}
                draggable={false}
              />

              {/* Soft elliptical "contact shadow" rendered in CSS so it scales
                  with the card and matches the studio look. */}
              <div style={{
                ...styles.cardShadow,
                background: isLight
                  ? 'radial-gradient(ellipse 50% 6% at 50% 92%, rgba(15,23,42,0.22) 0%, rgba(15,23,42,0) 70%)'
                  : 'radial-gradient(ellipse 50% 6% at 50% 92%, rgba(0,0,0,0.5) 0%, rgba(0,0,0,0) 70%)',
              }} />

              {/* Angle tag — top-left */}
              <div style={{ ...styles.tag, background: tagBg }}>
                {meta.angle_deg != null ? `${Math.round(meta.angle_deg)}°` : `#${i + 1}`}
              </div>
            </button>
          );
        })}
      </div>

      {active !== null && (
        <Lightbox
          url={heroes[active]}
          meta={heroMeta[active]}
          theme={theme}
          onClose={() => setActive(null)}
          onPrev={() => setActive((a) => (a - 1 + heroes.length) % heroes.length)}
          onNext={() => setActive((a) => (a + 1) % heroes.length)}
        />
      )}
    </div>
  );
}

function Lightbox({ url, meta, theme, onClose, onPrev, onNext }) {
  const isLight = theme === 'studio_light';
  React.useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') onClose();
      else if (e.key === 'ArrowLeft') onPrev();
      else if (e.key === 'ArrowRight') onNext();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose, onPrev, onNext]);

  return (
    <div style={styles.lightboxBackdrop} onClick={onClose}>
      <div
        style={{
          ...styles.lightboxCard,
          background: isLight ? '#fff' : '#0f0f12',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <img src={`${API_BASE}${url}`} alt="" style={styles.lightboxImg} />
        <div style={{
          ...styles.lightboxShadow,
          background: isLight
            ? 'radial-gradient(ellipse 45% 5% at 50% 92%, rgba(15,23,42,0.22) 0%, rgba(15,23,42,0) 70%)'
            : 'radial-gradient(ellipse 45% 5% at 50% 92%, rgba(0,0,0,0.5) 0%, rgba(0,0,0,0) 70%)',
        }} />
        <div style={styles.lightboxMeta}>
          <span>angle <strong>{meta?.angle_deg ?? '?'}°</strong></span>
          <span>· sharpness <strong>{meta?.sharpness ?? '?'}</strong></span>
          <span>· coverage <strong>{meta?.coverage != null ? `${(meta.coverage * 100).toFixed(1)}%` : '?'}</strong></span>
        </div>
        <button onClick={onClose} style={styles.lightboxClose} aria-label="Close">×</button>
        <button onClick={onPrev} style={{ ...styles.lightboxArrow, left: 16 }} aria-label="Previous">‹</button>
        <button onClick={onNext} style={{ ...styles.lightboxArrow, right: 16 }} aria-label="Next">›</button>
      </div>
    </div>
  );
}

const styles = {
  wrap: {
    display: 'flex', flexDirection: 'column', gap: 12,
  },
  header: {
    display: 'flex', alignItems: 'baseline', gap: 12,
  },
  label: {
    color: '#a1a1aa', fontSize: 11, fontWeight: 800, letterSpacing: 2,
  },
  sub: {
    color: '#52525b', fontSize: 11,
  },
  // Grid columns scale with hero count: 2 = single row, 4 = 2x2, 6 = 3x2.
  grid: (n) => ({
    display: 'grid',
    gap: 12,
    gridTemplateColumns: `repeat(${Math.min(n, 4)}, minmax(0, 1fr))`,
  }),
  card: {
    position: 'relative', overflow: 'hidden',
    borderRadius: 12, padding: 0, cursor: 'zoom-in',
    aspectRatio: '1 / 1',
    transition: 'transform 200ms cubic-bezier(0.4, 0.0, 0.2, 1), box-shadow 200ms cubic-bezier(0.4, 0.0, 0.2, 1)',
    appearance: 'none',
  },
  img: {
    position: 'absolute', inset: 0,
    width: '100%', height: '100%', objectFit: 'contain',
    pointerEvents: 'none',
  },
  cardShadow: {
    position: 'absolute', inset: 0, pointerEvents: 'none',
  },
  tag: {
    position: 'absolute', top: 8, left: 8,
    padding: '3px 8px', borderRadius: 999,
    color: '#fff', fontSize: 10, fontWeight: 800, letterSpacing: 1.2,
    backdropFilter: 'blur(4px)',
  },
  lightboxBackdrop: {
    position: 'fixed', inset: 0, zIndex: 9999,
    background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(8px)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    animation: 'lightboxIn 180ms ease-out',
  },
  lightboxCard: {
    position: 'relative',
    width: 'min(85vw, 85vh)', height: 'min(85vw, 85vh)',
    borderRadius: 16, overflow: 'hidden',
    boxShadow: '0 30px 80px rgba(0,0,0,0.5)',
  },
  lightboxImg: {
    position: 'absolute', inset: 0,
    width: '100%', height: '100%', objectFit: 'contain',
  },
  lightboxShadow: {
    position: 'absolute', inset: 0, pointerEvents: 'none',
  },
  lightboxMeta: {
    position: 'absolute', bottom: 16, left: 16, right: 16,
    color: '#27272a', fontSize: 12,
    display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'center',
    fontFamily: 'ui-monospace, SFMono-Regular, monospace',
    background: 'rgba(255,255,255,0.85)', padding: '8px 14px', borderRadius: 999,
    backdropFilter: 'blur(6px)', width: 'fit-content', margin: '0 auto',
  },
  lightboxClose: {
    position: 'absolute', top: 12, right: 12,
    width: 36, height: 36, borderRadius: 18, border: 'none',
    background: 'rgba(15,15,18,0.75)', color: '#fff',
    fontSize: 22, lineHeight: '34px', cursor: 'pointer',
    backdropFilter: 'blur(6px)',
  },
  lightboxArrow: {
    position: 'absolute', top: '50%', transform: 'translateY(-50%)',
    width: 44, height: 44, borderRadius: 22, border: 'none',
    background: 'rgba(15,15,18,0.65)', color: '#fff',
    fontSize: 28, lineHeight: '40px', cursor: 'pointer',
    backdropFilter: 'blur(6px)',
  },
};

if (typeof document !== 'undefined' && !document.getElementById('hero-gallery-keyframes')) {
  const s = document.createElement('style');
  s.id = 'hero-gallery-keyframes';
  s.innerHTML = `
    @keyframes lightboxIn {
      from { opacity: 0 }
      to   { opacity: 1 }
    }
  `;
  document.head.appendChild(s);
}
