import React, { useEffect, useRef, useState } from 'react';
import { subscribe } from '../lib/dlog';

/**
 * EventLogHUD — bottom-right floating live trace of every dlog() call.
 *
 * The historical pain when something breaks during a phone upload is
 * "the avatar froze and I have no idea where in the chain it failed."
 * This HUD shows every state transition (WS connect, phone status,
 * play_clip received, listener attach/detach, tier1 driver events) in
 * one rolling list with timestamps + color coding, so a 5-second
 * glance tells you exactly which step the chain dropped.
 *
 * Color key:
 *   green  — lifecycle success (connect, listener_attached, play_started)
 *   yellow — in-flight (recording, uploading, event_received)
 *   red    — error (failed, stalled, error, closed)
 *   purple — phone-specific events
 *   blue   — websocket events
 *   gray   — informational
 *
 * Toggle visibility with the ✕ in the top-right of the panel; the
 * subscription stays alive so re-opening shows the live feed instantly.
 */
export function EventLogHUD() {
  const [events, setEvents] = useState([]);
  const [visible, setVisible] = useState(true);
  const ringRef = useRef([]);
  const MAX_EVENTS = 30;

  useEffect(() => {
    return subscribe((evt) => {
      ringRef.current.push(evt);
      if (ringRef.current.length > MAX_EVENTS) {
        ringRef.current.shift();
      }
      setEvents([...ringRef.current]);
    });
  }, []);

  if (!visible) {
    return (
      <button
        onClick={() => setVisible(true)}
        style={styles.openBtn}
        title="show event log"
      >
        ▾ events
      </button>
    );
  }

  return (
    <div style={styles.panel}>
      <div style={styles.header}>
        <span style={styles.title}>EVENT LOG · {events.length}/{MAX_EVENTS}</span>
        <button
          onClick={() => { ringRef.current = []; setEvents([]); }}
          style={styles.btn}
          title="clear"
        >
          ⟲
        </button>
        <button
          onClick={() => setVisible(false)}
          style={styles.btn}
          title="hide"
        >
          ✕
        </button>
      </div>
      <div style={styles.list}>
        {events.length === 0 && (
          <div style={styles.empty}>waiting for events…</div>
        )}
        {events.map((evt, i) => (
          <EventRow key={i} evt={evt} />
        ))}
      </div>
    </div>
  );
}

function EventRow({ evt }) {
  const palette = colorFor(evt);
  const time = formatTime(evt.ts);
  const dataPreview = formatData(evt.data);
  return (
    <div style={{ ...styles.row, borderLeft: `3px solid ${palette}` }}>
      <span style={styles.time}>{time}</span>
      <span style={{ ...styles.src, color: palette }}>{evt.src}</span>
      <span style={styles.msg}>{evt.msg}</span>
      {dataPreview && <span style={styles.data}>{dataPreview}</span>}
    </div>
  );
}

// Pick a color per event so the visual flow at-a-glance shows phase changes.
// Falls through to a neutral gray for anything we don't explicitly classify.
function colorFor(evt) {
  const m = evt.msg || '';
  // Errors first — they should always shout.
  if (/(error|failed|stalled|closed|disconnect)/i.test(m)) return '#ef4444';
  // Lifecycle successes — green.
  if (/(connected|attached|complete|started|play_started|preload_hit)/i.test(m)) return '#22c55e';
  // In-flight transitions — yellow.
  if (/(recording|uploading|event_received|received|reconnect)/i.test(m)) return '#fbbf24';
  // Source-based default colors.
  if (evt.src === 'phone') return '#a855f7';
  if (evt.src === 'ws') return '#3b82f6';
  if (evt.src === 'avatarStream') return '#06b6d4';
  if (evt.src === 'tier1' || evt.src === 'tier0') return '#f97316';
  return '#71717a';
}

function formatTime(ts) {
  const d = new Date(ts);
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  const ss = String(d.getSeconds()).padStart(2, '0');
  const ms = String(d.getMilliseconds()).padStart(3, '0');
  return `${hh}:${mm}:${ss}.${ms}`;
}

// Compact one-liner of the data payload — pick the 2-3 most relevant
// keys so the row stays scannable. Full payload is in console.info if
// the operator wants the rest.
function formatData(data) {
  if (!data || typeof data !== 'object') return '';
  const interesting = [
    'intent', 'url', 'layer', 'status', 'connected',
    'code', 'reason', 'bytes', 'chunks', 'mime_type',
  ];
  const parts = [];
  for (const k of interesting) {
    if (data[k] != null) {
      let v = data[k];
      if (typeof v === 'string' && v.length > 40) v = v.slice(0, 38) + '…';
      parts.push(`${k}=${v}`);
    }
  }
  return parts.join(' · ');
}

const styles = {
  panel: {
    position: 'fixed', bottom: 14, right: 14, zIndex: 90,
    width: 460, maxWidth: '40vw', maxHeight: '50vh',
    background: 'rgba(15,15,18,0.92)',
    backdropFilter: 'blur(12px)',
    border: '1px solid #27272a',
    borderRadius: 8,
    boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
    display: 'flex', flexDirection: 'column',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    fontSize: 10,
    color: '#fafafa',
  },
  header: {
    display: 'flex', alignItems: 'center', gap: 6,
    padding: '6px 10px',
    borderBottom: '1px solid #27272a',
    background: '#0a0a0b',
    borderRadius: '8px 8px 0 0',
  },
  title: {
    flex: 1,
    fontSize: 9, fontWeight: 800,
    color: '#a1a1aa',
    letterSpacing: 1.2,
  },
  btn: {
    background: 'transparent',
    border: '1px solid #27272a',
    color: '#a1a1aa',
    borderRadius: 4,
    padding: '2px 6px',
    fontSize: 11, lineHeight: 1,
    cursor: 'pointer',
    minWidth: 22,
  },
  list: {
    flex: 1, overflowY: 'auto',
    padding: '4px 0',
    display: 'flex', flexDirection: 'column-reverse', // newest at top
  },
  empty: {
    padding: '10px 12px',
    color: '#52525b', fontStyle: 'italic',
  },
  row: {
    display: 'flex', alignItems: 'baseline', gap: 6,
    padding: '2px 8px 2px 6px',
    lineHeight: 1.35,
    borderBottom: '1px solid rgba(39,39,42,0.5)',
  },
  time: {
    color: '#52525b',
    fontSize: 9,
    minWidth: 80,
    flexShrink: 0,
  },
  src: {
    fontWeight: 800,
    minWidth: 80,
    flexShrink: 0,
  },
  msg: {
    color: '#fafafa',
    fontWeight: 600,
    minWidth: 130,
    flexShrink: 0,
  },
  data: {
    color: '#a1a1aa',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  openBtn: {
    position: 'fixed', bottom: 14, right: 14, zIndex: 90,
    background: 'rgba(15,15,18,0.85)',
    border: '1px solid #27272a',
    color: '#a1a1aa',
    borderRadius: 999,
    padding: '6px 12px',
    fontSize: 10, fontWeight: 800,
    letterSpacing: 1,
    cursor: 'pointer',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  },
};
