import React from 'react';

/**
 * TranslationChip — top-right corner badge that frames the audio output as
 * "auto-captioned". Sells two things:
 *
 *  1. The streaming aesthetic. TikTok Live / IG Live audiences expect
 *     captions on speaker video; their absence reads as "studio recording",
 *     their presence reads as "this is a real live stream right now".
 *  2. The audio-first architecture's micro-asynchrony cover. When the
 *     30s pitch's pre-rendered video loops underneath the audio, the lip
 *     motion isn't perfectly phoneme-locked. The chip primes the audience
 *     to interpret any motion mismatch as the captioning layer doing its
 *     job, not as bad lip-sync.
 *
 * Per REVISIONS §9 we ship the auto-caption framing (not "translated from
 * Mandarin") because we don't render an actual Mandarin pitch tonight.
 * Same psychological effect, no false claim.
 *
 * Props:
 *   visible: render only when there's audio playing (mounted by LiveStage
 *     when audioPlaying is set).
 *   variant: 'live' (default) or 'pitch' — pitch variant uses a brighter
 *     accent so the demo's hero moment reads more deliberate.
 */
export function TranslationChip({ visible = true, variant = 'live' }) {
  if (!visible) return null;

  const tone = variant === 'pitch'
    ? styles.tonePitch
    : styles.toneLive;

  return (
    <div style={{ ...styles.chip, ...tone }} aria-hidden>
      <span style={styles.dotWrap}>
        <span style={styles.dotInner} />
      </span>
      <span style={styles.label}>LIVE</span>
      <span style={styles.divider} />
      <span style={styles.sub}>auto-captioned</span>
    </div>
  );
}

const styles = {
  chip: {
    position: 'absolute',
    top: 14,
    right: 14,
    zIndex: 9,
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: '6px 12px 6px 8px',
    borderRadius: 999,
    border: '1px solid',
    backdropFilter: 'blur(6px)',
    WebkitBackdropFilter: 'blur(6px)',
    color: '#fafafa',
    boxShadow: '0 4px 16px rgba(0,0,0,0.45)',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    animation: 'translationChipIn 280ms cubic-bezier(0.4, 0, 0.2, 1)',
  },
  toneLive: {
    background: 'rgba(127, 29, 29, 0.85)',
    borderColor: 'rgba(248,113,113,0.85)',
  },
  tonePitch: {
    background: 'rgba(67, 20, 7, 0.92)',
    borderColor: 'rgba(252,165,79,0.85)',
  },
  dotWrap: {
    width: 22, height: 22, borderRadius: 11,
    display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
    background: 'rgba(0,0,0,0.35)',
  },
  dotInner: {
    width: 10, height: 10, borderRadius: 5,
    background: '#ef4444',
    boxShadow: '0 0 12px #ef4444, 0 0 6px #fca5a5',
    animation: 'translationChipPulse 1.4s ease-in-out infinite',
  },
  label: {
    fontSize: 12, fontWeight: 900, letterSpacing: 1.6,
  },
  divider: {
    width: 1, height: 14,
    background: 'rgba(255,255,255,0.32)',
    margin: '0 2px',
  },
  sub: {
    fontSize: 11, fontWeight: 600, letterSpacing: 0.8,
    color: 'rgba(254,243,199,0.92)',
  },
};

if (typeof document !== 'undefined' && !document.getElementById('translation-chip-keyframes')) {
  const s = document.createElement('style');
  s.id = 'translation-chip-keyframes';
  s.innerHTML = `
    @keyframes translationChipIn {
      from { transform: translateX(8px); opacity: 0 }
      to   { transform: translateX(0);   opacity: 1 }
    }
    @keyframes translationChipPulse {
      0%, 100% { opacity: 1; transform: scale(1) }
      50%      { opacity: 0.55; transform: scale(0.85) }
    }
  `;
  document.head.appendChild(s);
}
