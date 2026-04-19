import React, { useEffect, useRef, useState } from 'react';

/**
 * LanguagePicker — operator-facing live-language selector.
 *
 * Surfaces the backend's `pipeline_state["active_language"]` as a tappable
 * pill row (one tile per supported language). On change we POST
 * `/api/live/language` with `lang=<code>`; the server flips its in-memory
 * state and broadcasts a `language_changed` WS event so every connected
 * dashboard stays in sync (useEmpireSocket subscribes + updates the
 * `activeLanguage` it hands back to App).
 *
 * Flow once the operator picks a language:
 *   1. POST /api/live/language → server updates pipeline_state, broadcasts.
 *   2. Operator drops the product video.
 *   3. run_video_sell_pipeline → run_sell_pipeline reads active_language
 *      and feeds it into translator.translate(script, lang) +
 *      text_to_speech(..., language_code=lang). ElevenLabs flash_v2_5 is
 *      multilingual so the avatar speaks in the chosen language with the
 *      same voice id.
 *
 * Two presentations driven by `compact`:
 *   • compact=false (pre-upload) — a horizontal grid of six tiles with
 *     flag + native label + ISO code, so the operator can pick before they
 *     drop a video. Sits beneath the "Drop a product video" hint.
 *   • compact=true (post-upload) — a single corner chip showing the active
 *     flag + code; clicking it pops a small grid out so the language can
 *     still be changed mid-stream without leaving the stage.
 *
 * Languages mirror agents/translator.py SUPPORTED. To add a 7th language,
 * append a row to SUPPORTED + add a row here. (Same row order — the demo
 * voice expects en first because that's the un-translated baseline.)
 */
export const LANGUAGES = [
  { code: 'en', label: 'English',  flag: '🇺🇸' },
  { code: 'es', label: 'Español',  flag: '🇪🇸' },
  { code: 'fr', label: 'Français', flag: '🇫🇷' },
  { code: 'de', label: 'Deutsch',  flag: '🇩🇪' },
  { code: 'zh', label: '中文',     flag: '🇨🇳' },
  { code: 'tl', label: 'Tagalog',  flag: '🇵🇭' },
];

export function LanguagePicker({
  activeLanguage = 'en',
  onChange,
  compact = false,
  disabled = false,
}) {
  const [pending, setPending] = useState(null);
  const [expanded, setExpanded] = useState(false);
  const containerRef = useRef(null);
  const apiBase = `http://${window.location.hostname}:8000`;

  // Auto-collapse the post-upload popout when the operator clicks
  // anywhere outside the picker (consistent with chip/menu UX patterns
  // — feels less intrusive than an explicit close button).
  useEffect(() => {
    if (!compact || !expanded) return;
    function handleClick(e) {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setExpanded(false);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [compact, expanded]);

  async function pick(code) {
    if (code === activeLanguage || pending || disabled) return;
    setPending(code);
    try {
      const fd = new FormData();
      fd.append('lang', code);
      const r = await fetch(`${apiBase}/api/live/language`, {
        method: 'POST',
        body: fd,
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      // Server broadcasts language_changed over WS — useEmpireSocket
      // updates activeLanguage from there (single source of truth).
      // We also call onChange optimistically so the local UI updates
      // even if the WS round-trip lags.
      onChange?.(code);
    } catch (e) {
      console.warn('[language] set failed', e);
    } finally {
      setPending(null);
      // Collapse the popout after a successful pick so the chip
      // returns to its compact form and the operator sees the new
      // selection reflected in one motion.
      if (compact) setExpanded(false);
    }
  }

  const active = LANGUAGES.find(l => l.code === activeLanguage) || LANGUAGES[0];

  // Compact variant: collapsed chip + click-to-expand popout. Used
  // post-upload so the picker doesn't dominate the stage but stays
  // reachable for mid-stream language changes.
  if (compact) {
    return (
      <div ref={containerRef} style={styles.compactRoot}>
        <button
          type="button"
          onClick={() => setExpanded(v => !v)}
          style={{
            ...styles.compactChip,
            ...(expanded ? styles.compactChipOpen : null),
          }}
          aria-label={`Language: ${active.label}. Click to change.`}
        >
          <span style={styles.compactFlag}>{active.flag}</span>
          <span style={styles.compactCode}>{active.code.toUpperCase()}</span>
          <span style={styles.compactCaret}>{expanded ? '▴' : '▾'}</span>
        </button>

        {expanded && (
          <div style={styles.compactGrid}>
            {LANGUAGES.map(lang => {
              const isActive = lang.code === activeLanguage;
              const isPending = lang.code === pending;
              return (
                <button
                  type="button"
                  key={lang.code}
                  onClick={() => pick(lang.code)}
                  disabled={isPending || disabled}
                  style={{
                    ...styles.compactTile,
                    ...(isActive ? styles.compactTileActive : null),
                    ...(isPending ? styles.tilePending : null),
                  }}
                >
                  <span style={styles.compactTileFlag}>{lang.flag}</span>
                  <span style={styles.compactTileLabel}>{lang.label}</span>
                </button>
              );
            })}
          </div>
        )}
      </div>
    );
  }

  // Full variant: prominent six-tile row, used pre-upload as a "step 1:
  // pick language" affordance directly under the "Drop a product video"
  // hint. Matches the empty-state's visual language (monospace, dim
  // chrome) so it doesn't compete with the drop-zone callout.
  return (
    <div style={styles.fullRoot}>
      <p style={styles.fullCaption}>STEP 1 · LANGUAGE</p>
      <div style={styles.fullGrid}>
        {LANGUAGES.map(lang => {
          const isActive = lang.code === activeLanguage;
          const isPending = lang.code === pending;
          return (
            <button
              type="button"
              key={lang.code}
              onClick={() => pick(lang.code)}
              disabled={isPending || disabled}
              style={{
                ...styles.fullTile,
                ...(isActive ? styles.fullTileActive : null),
                ...(isPending ? styles.tilePending : null),
              }}
              aria-pressed={isActive}
            >
              <span style={styles.fullFlag}>{lang.flag}</span>
              <span style={styles.fullLabel}>{lang.label}</span>
              <span style={styles.fullCode}>{lang.code.toUpperCase()}</span>
            </button>
          );
        })}
      </div>
      <p style={styles.fullSub}>
        avatar will speak in {active.label.toLowerCase()} · same voice, translated script
      </p>
    </div>
  );
}

const styles = {
  // ── Full (pre-upload) ──────────────────────────────────────────────
  fullRoot: {
    display: 'flex', flexDirection: 'column',
    alignItems: 'center', gap: 14,
    pointerEvents: 'auto',
  },
  fullCaption: {
    fontSize: 11, fontWeight: 700, letterSpacing: 2,
    color: '#52525b', margin: 0,
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    textTransform: 'uppercase',
  },
  fullGrid: {
    display: 'flex', gap: 10, flexWrap: 'wrap',
    justifyContent: 'center',
  },
  fullTile: {
    display: 'flex', flexDirection: 'column',
    alignItems: 'center', justifyContent: 'center',
    gap: 4, minWidth: 84, padding: '12px 14px',
    background: 'rgba(15,15,18,0.85)',
    border: '1px solid #27272a',
    borderRadius: 12,
    color: '#a1a1aa',
    cursor: 'pointer',
    transition: 'transform 120ms ease, border-color 120ms ease, color 120ms ease, background 120ms ease',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  },
  fullTileActive: {
    background: 'rgba(124,58,237,0.18)',
    borderColor: 'rgba(167,139,250,0.85)',
    color: '#fafafa',
    transform: 'translateY(-1px)',
    boxShadow: '0 6px 20px rgba(124,58,237,0.35), 0 0 0 1px rgba(167,139,250,0.4) inset',
  },
  fullFlag: {
    fontSize: 28, lineHeight: 1,
  },
  fullLabel: {
    fontSize: 11, fontWeight: 700, letterSpacing: 0.6,
    fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif',
  },
  fullCode: {
    fontSize: 9, fontWeight: 800, letterSpacing: 1.4,
    color: '#71717a',
  },
  fullSub: {
    fontSize: 10, fontWeight: 600, letterSpacing: 1.2,
    color: '#3f3f46', margin: 0,
    textTransform: 'uppercase',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  },

  // ── Compact (post-upload) ──────────────────────────────────────────
  compactRoot: {
    position: 'relative',
    pointerEvents: 'auto',
  },
  compactChip: {
    display: 'flex', alignItems: 'center', gap: 6,
    background: 'rgba(15,15,18,0.85)',
    backdropFilter: 'blur(8px)',
    WebkitBackdropFilter: 'blur(8px)',
    border: '1px solid #27272a',
    borderRadius: 999,
    padding: '5px 10px 5px 8px',
    color: '#fafafa',
    cursor: 'pointer',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    fontSize: 11,
    transition: 'border-color 120ms ease, background 120ms ease',
  },
  compactChipOpen: {
    borderColor: 'rgba(167,139,250,0.85)',
    background: 'rgba(35,20,55,0.92)',
  },
  compactFlag: {
    fontSize: 14, lineHeight: 1,
  },
  compactCode: {
    fontWeight: 800, letterSpacing: 1.4,
  },
  compactCaret: {
    fontSize: 9, color: '#a1a1aa',
  },
  compactGrid: {
    position: 'absolute',
    top: 'calc(100% + 6px)',
    right: 0,
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: 6,
    background: 'rgba(15,15,18,0.94)',
    backdropFilter: 'blur(10px)',
    WebkitBackdropFilter: 'blur(10px)',
    border: '1px solid #27272a',
    borderRadius: 12,
    padding: 8,
    minWidth: 220,
    boxShadow: '0 14px 40px rgba(0,0,0,0.6)',
  },
  compactTile: {
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '8px 10px',
    background: 'rgba(24,24,27,0.6)',
    border: '1px solid #27272a',
    borderRadius: 8,
    color: '#a1a1aa',
    cursor: 'pointer',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    fontSize: 11,
    transition: 'background 120ms ease, border-color 120ms ease, color 120ms ease',
  },
  compactTileActive: {
    background: 'rgba(124,58,237,0.22)',
    borderColor: 'rgba(167,139,250,0.85)',
    color: '#fafafa',
  },
  compactTileFlag: {
    fontSize: 16, lineHeight: 1,
  },
  compactTileLabel: {
    fontSize: 11, fontWeight: 700,
    fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif',
  },

  // Shared "request in flight" affordance.
  tilePending: {
    opacity: 0.55, cursor: 'wait',
  },
};
