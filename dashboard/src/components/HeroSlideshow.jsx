import React, { useEffect, useRef, useState } from 'react';

const API_BASE = `http://${window.location.hostname}:8000`;

/**
 * HeroSlideshow — auto-rotating companion to the Spin3D carousel.
 *
 * The product ingestion pipeline (backend/agents/threed.py → carousel_from_video)
 * scores the candidate video frames by sharpness + compactness + coverage and
 * picks N hero shots, one per rotation quadrant. The Spin3D shader uses the
 * dense `frames` array; this component uses the curated `heroes` array to
 * give each best shot ~3.5s of dedicated screen time so the audience reads
 * the product in detail while the avatar narrates over the top.
 *
 * Visual character: the same studio aesthetic as Spin3D — transparent product
 * PNG (rembg cleaned), soft elliptical floor shadow rendered in CSS, rounded
 * card. Crossfades between shots are pure CSS opacity transitions; no canvas,
 * no GL, no rAF — just a setInterval ticking the active index. Total cost
 * per slot change: one repaint of two <img> elements.
 *
 * The static HeroGallery (used by /test) shows all heroes in a 2×N grid for
 * picking a thumbnail; HeroSlideshow shows one at a time as a passive ambient
 * surface that loops while the avatar talks. Different surfaces, same data.
 *
 * Props:
 *   heroes      : array of hero PNG URLs (relative to API_BASE)
 *   heroMeta    : optional [{angle_deg, sharpness, coverage}, ...] aligned with heroes
 *   intervalMs  : dwell time per shot (default 3500 — user spec'd "3 to 4 seconds")
 *   fadeMs      : crossfade duration (default 600)
 *   theme       : 'studio_dark' (default) | 'studio_light' — matches Spin3D themes
 *   paused      : skip the auto-advance interval (e.g. on hover)
 *   showBadge   : render the "HERO i/N" pill in the top-left (default true)
 *   showDots    : render the progress dots strip in the bottom (default true)
 *   ariaLabel   : accessible name; defaults to "Product hero shots"
 */
export function HeroSlideshow({
  heroes = [],
  heroMeta = [],
  intervalMs = 3500,
  fadeMs = 600,
  theme = 'studio_dark',
  paused = false,
  showBadge = true,
  showDots = true,
  ariaLabel = 'Product hero shots',
}) {
  const [activeIdx, setActiveIdx] = useState(0);
  const [hovered, setHovered] = useState(false);

  // Reset to the first hero whenever the URL list changes (new product loaded).
  // Joined into a single key string so equal arrays don't trigger a reset.
  const heroesKey = heroes.join('|');
  useEffect(() => {
    setActiveIdx(0);
  }, [heroesKey]);

  // Auto-advance. Skipped when paused, hovered, or there's only 0–1 heroes
  // (no rotation needed). interval lives in a ref-light closure so we don't
  // re-create it on every render — only when the dependencies actually change.
  useEffect(() => {
    if (heroes.length <= 1) return;
    if (paused || hovered) return;
    const id = setInterval(() => {
      setActiveIdx(i => (i + 1) % heroes.length);
    }, intervalMs);
    return () => clearInterval(id);
  }, [heroes.length, intervalMs, paused, hovered]);

  if (!heroes.length) return null;

  const isLight = theme === 'studio_light';
  const cardBg = isLight ? '#ffffff' : '#0a0a0a';
  const cardBorder = isLight ? '1px solid #e4e4e7' : '1px solid #27272a';
  const cardShadow = isLight
    ? '0 1px 3px rgba(15,23,42,0.06), 0 8px 24px rgba(15,23,42,0.08)'
    : '0 1px 3px rgba(0,0,0,0.4), 0 8px 24px rgba(0,0,0,0.5)';
  const badgeBg = isLight ? 'rgba(15,15,18,0.78)' : 'rgba(124,58,237,0.92)';
  const dotIdle = isLight ? 'rgba(15,15,18,0.22)' : 'rgba(255,255,255,0.28)';
  const dotActive = isLight ? '#0a0a0a' : '#fafafa';

  const activeMeta = heroMeta[activeIdx] || {};

  return (
    <div
      style={{
        ...styles.wrap,
        background: cardBg,
        border: cardBorder,
        boxShadow: cardShadow,
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      role="img"
      aria-label={ariaLabel}
    >
      {/* All hero images stacked; only the active one is visible. We mount
          them all so the browser can decode + cache once and crossfades are
          a pure GPU opacity transition (no decode hitch on switch). */}
      <div style={styles.stack}>
        {heroes.map((url, i) => (
          <img
            key={url}
            src={`${API_BASE}${url}`}
            alt=""
            draggable={false}
            // eager-load the first hero, lazy-load the rest so the panel
            // shows something immediately even on slow links.
            loading={i === 0 ? 'eager' : 'lazy'}
            decoding="async"
            style={{
              ...styles.img,
              opacity: i === activeIdx ? 1 : 0,
              transition: `opacity ${fadeMs}ms ease-in-out`,
            }}
          />
        ))}
      </div>

      {/* Soft elliptical floor shadow — same trick as HeroGallery / Spin3D
          so the three product surfaces all read as if the item is sitting on
          the same studio paper. Rendered in CSS so it scales with the card. */}
      <div
        style={{
          ...styles.floorShadow,
          background: isLight
            ? 'radial-gradient(ellipse 50% 6% at 50% 92%, rgba(15,23,42,0.22) 0%, rgba(15,23,42,0) 70%)'
            : 'radial-gradient(ellipse 50% 7% at 50% 92%, rgba(0,0,0,0.65) 0%, rgba(0,0,0,0) 70%)',
        }}
      />

      {/* Hero count badge — top-left, mirrors Spin3D's "ON-DEVICE 3D" pill so
          the two stacked panels feel like one unit. */}
      {showBadge && (
        <div style={styles.badgeStack}>
          <div style={{ ...styles.badge, background: badgeBg }}>
            HERO {activeIdx + 1}/{heroes.length}
          </div>
          {activeMeta.angle_deg != null && (
            <div style={{ ...styles.angleTag, background: isLight ? 'rgba(255,255,255,0.85)' : 'rgba(9,9,11,0.85)',
                          color: isLight ? '#27272a' : '#a1a1aa' }}>
              {Math.round(activeMeta.angle_deg)}°
            </div>
          )}
        </div>
      )}

      {/* Progress dots — gives a subtle "i/N" affordance and previews the
          next-cycle moment so the audience's eye knows it's a slideshow,
          not a static photo. Active dot is wider (pill) than the others. */}
      {showDots && heroes.length > 1 && (
        <div style={styles.dots}>
          {heroes.map((_, i) => (
            <div
              key={i}
              style={{
                ...styles.dot,
                background: i === activeIdx ? dotActive : dotIdle,
                width: i === activeIdx ? 16 : 5,
              }}
            />
          ))}
        </div>
      )}
    </div>
  );
}

const styles = {
  wrap: {
    position: 'relative',
    width: '100%',
    aspectRatio: '1 / 1',
    borderRadius: 12,
    overflow: 'hidden',
  },
  stack: {
    position: 'absolute', inset: 0,
  },
  img: {
    position: 'absolute', inset: 0,
    width: '100%', height: '100%',
    objectFit: 'contain',
    pointerEvents: 'none',
  },
  floorShadow: {
    position: 'absolute', inset: 0, pointerEvents: 'none',
  },
  badgeStack: {
    position: 'absolute', top: 8, left: 8, zIndex: 2,
    display: 'flex', flexDirection: 'column', gap: 4, alignItems: 'flex-start',
    pointerEvents: 'none',
  },
  badge: {
    padding: '3px 9px', borderRadius: 999,
    color: '#fff', fontSize: 10, fontWeight: 800, letterSpacing: 1.2,
    backdropFilter: 'blur(4px)',
    fontFamily: 'ui-monospace, SFMono-Regular, monospace',
  },
  angleTag: {
    padding: '2px 7px', borderRadius: 999,
    fontSize: 9, fontWeight: 700, letterSpacing: 0.8,
    backdropFilter: 'blur(4px)',
    fontFamily: 'ui-monospace, SFMono-Regular, monospace',
  },
  dots: {
    position: 'absolute', bottom: 10, left: 0, right: 0, zIndex: 2,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    gap: 4,
    pointerEvents: 'none',
  },
  dot: {
    height: 4,
    borderRadius: 2,
    transition: 'width 240ms ease, background 240ms ease',
  },
};
