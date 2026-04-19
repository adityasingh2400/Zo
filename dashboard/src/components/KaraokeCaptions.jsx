import React, { useEffect, useRef, useState } from 'react';

/**
 * KaraokeCaptions — word-by-word caption overlay synced to a playing
 * `<audio>` element. Highlights the active word in brand color, trails
 * past words in white, hides future words.
 *
 * Why captions: design doc thesis 3 — "Captions divert eye gaze. ~80% of
 * viewer fixation is on text when text is present. Mouth gets <15%."
 * The 30s pitch reuses an 8s looping Veo pitching-pose video; the loop
 * is invisible because the audience is reading captions, not watching
 * lips.
 *
 * Word timings are SYNTHESIZED on the backend (whitespace split + audio
 * duration probe) since we ship without Cartesia today. Accuracy is
 * ~80-95%; the rAF loop re-syncs every frame from
 * audioRef.current.currentTime so any drift inside a word is invisible
 * at the active-word boundary.
 *
 * Sliding window: shows ~10 words at once centred loosely on the active
 * word. Past words trail back, future words come in, scroll feels
 * continuous without ever showing the whole 75-word transcript at once.
 *
 * Props:
 *   audioRef: React ref to the playing <audio> element
 *   wordTimings: [{word, start, end}, ...] in seconds
 *   windowSize: visible word count (default 10)
 *   visible: boolean override (default true). False = render null.
 */
export function KaraokeCaptions({
  audioRef,
  wordTimings,
  windowSize = 10,
  visible = true,
}) {
  const [activeIdx, setActiveIdx] = useState(-1);
  const rafRef = useRef(null);

  useEffect(() => {
    if (!visible) {
      setActiveIdx(-1);
      return;
    }
    if (!audioRef?.current || !wordTimings?.length) {
      setActiveIdx(-1);
      return;
    }
    const audio = audioRef.current;

    function tick() {
      const t = audio.currentTime;
      // Binary search for the word whose [start, end) interval contains t.
      let lo = 0;
      let hi = wordTimings.length - 1;
      let idx = -1;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        const w = wordTimings[mid];
        if (t >= w.start && t < w.end) { idx = mid; break; }
        if (t < w.start) hi = mid - 1;
        else lo = mid + 1;
      }
      // If t is past a word's end (in the gap before next word), highlight
      // the last completed word so the caption never blanks mid-pitch.
      if (idx === -1 && lo > 0) idx = lo - 1;
      // Past the last word: clamp to last index so the trail stays visible
      // briefly after audio ends (parent will unmount us soon either way).
      if (t >= (wordTimings[wordTimings.length - 1]?.end || 0)) {
        idx = wordTimings.length - 1;
      }
      setActiveIdx(idx);
      rafRef.current = requestAnimationFrame(tick);
    }
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [audioRef, wordTimings, visible]);

  if (!visible || !wordTimings?.length) return null;
  if (activeIdx < 0) return null;

  // Sliding window: centre loosely on activeIdx, clamp to ends so the
  // last few words don't shrink the visible row prematurely.
  const half = Math.floor(windowSize / 2);
  let start = Math.max(0, activeIdx - half);
  const end = Math.min(wordTimings.length, start + windowSize);
  // If we're near the end, slide the window left so the active word
  // stays in the right portion (reads as "the pitch is wrapping up").
  if (end - start < windowSize) {
    start = Math.max(0, end - windowSize);
  }
  const visibleWords = wordTimings.slice(start, end);

  return (
    <div style={styles.container} aria-hidden>
      <div style={styles.row}>
        {visibleWords.map((w, i) => {
          const realIdx = start + i;
          const isActive = realIdx === activeIdx;
          const isPast = realIdx < activeIdx;
          return (
            <span
              key={`${realIdx}-${w.word}`}
              style={{
                ...styles.word,
                ...(isPast ? styles.past : {}),
                ...(isActive ? styles.active : {}),
              }}
            >
              {w.word}
            </span>
          );
        })}
      </div>
    </div>
  );
}

const styles = {
  container: {
    position: 'absolute',
    bottom: 90,
    left: '5%',
    right: '5%',
    pointerEvents: 'none',
    zIndex: 8,
    display: 'flex',
    justifyContent: 'center',
  },
  row: {
    display: 'flex',
    flexWrap: 'wrap',
    justifyContent: 'center',
    alignItems: 'baseline',
    gap: '0.35em',
    fontSize: 'clamp(28px, 4.4vw, 56px)',
    fontWeight: 900,
    lineHeight: 1.18,
    fontFamily: '"Inter", "Helvetica Neue", system-ui, sans-serif',
    textAlign: 'center',
    letterSpacing: -0.5,
    // Heavy multi-stop text-shadow does the legibility work — readable on
    // any background without needing a dark plate behind the words.
    textShadow:
      '0 0 4px rgba(0,0,0,0.95),' +
      '0 0 12px rgba(0,0,0,0.85),' +
      '3px 3px 0 rgba(0,0,0,0.95),' +
      '-1px -1px 0 rgba(0,0,0,0.9)',
    color: 'rgba(255,255,255,0.36)',
  },
  word: {
    transition:
      'color 180ms ease-out, transform 180ms cubic-bezier(0.34, 1.56, 0.64, 1), text-shadow 180ms ease',
    display: 'inline-block',
    transformOrigin: 'center bottom',
    willChange: 'transform, color',
  },
  past: {
    color: '#ffffff',
  },
  active: {
    color: '#fcd34d',
    transform: 'scale(1.12)',
    textShadow:
      '0 0 6px rgba(0,0,0,0.95),' +
      '0 0 14px rgba(252,211,77,0.55),' +
      '0 0 28px rgba(252,211,77,0.35),' +
      '3px 3px 0 rgba(0,0,0,0.95)',
  },
};
