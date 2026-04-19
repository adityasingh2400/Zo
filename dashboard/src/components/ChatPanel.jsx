import React, { useState, useRef, useEffect } from 'react';

/**
 * ChatPanel — TikTok-Live-style overlay (no container chrome).
 *
 * Messages float directly over whatever's behind them — no panel
 * background, no header, no border. Each bubble has its own translucent
 * dark fill so it stays readable against any frame, but the surrounding
 * dock is invisible. Preset chips are tiny floating pills (no
 * containing box). Input is a single rounded pill at the bottom.
 *
 * Why no container: the operator wanted the chat to feel like ambient
 * audience reaction layered over the live stage, not a Slack-style
 * sidebar. Anything that reads as "chrome" pulls the eye off the
 * avatar.
 */
export function ChatPanel({ onSendComment, commentResponse, pendingComments = [] }) {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, pendingComments]);

  // Append a response row whenever a new commentResponse arrives.
  // Keyed on response text + total_ms so duplicates skip.
  useEffect(() => {
    if (!commentResponse?.response) return;
    setMessages(prev => {
      const lastResp = [...prev].reverse().find(m => m.type === 'response');
      if (lastResp && lastResp.text === commentResponse.response &&
          lastResp.totalMs === commentResponse.total_ms) {
        return prev;
      }
      return [...prev, {
        type: 'response',
        text: commentResponse.response,
        comment: commentResponse.comment,
        totalMs: commentResponse.total_ms,
        timestamp: Date.now(),
      }];
    });
  }, [commentResponse]);

  function handleSend() {
    if (!input.trim()) return;
    setMessages(prev => [...prev, {
      type: 'comment',
      text: input.trim(),
      timestamp: Date.now(),
    }]);
    onSendComment(input.trim());
    setInput('');
  }

  const presetComments = [
    "Is it real leather?",
    "What colors does it come in?",
    "How much does it weigh?",
    "Can I return it?",
    "Does it ship internationally?",
  ];

  return (
    <div style={styles.root}>
      {/* Floating message stream — bubbles only, no container background.
          Reverse-stacked at the bottom so the latest message sits above
          the input row. Older messages scroll up + out. */}
      <div style={styles.stream}>
        {messages.map((msg, i) => (
          <div key={i} style={{
            ...styles.bubble,
            alignSelf: msg.type === 'comment' ? 'flex-end' : 'flex-start',
            background: msg.type === 'comment'
              ? 'rgba(59,130,246,0.85)'
              : 'rgba(0,0,0,0.55)',
          }}>
            <div style={styles.bubbleHead}>
              <span style={{
                ...styles.bubbleLabel,
                color: msg.type === 'comment' ? '#dbeafe' : '#86efac',
              }}>
                {msg.type === 'comment' ? 'Viewer' : 'AI Seller'}
              </span>
              {msg.totalMs != null && (
                <span style={styles.latencyChip}>
                  ⚡ {(msg.totalMs / 1000).toFixed(1)}s
                </span>
              )}
            </div>
            <span style={styles.bubbleText}>{msg.text}</span>
          </div>
        ))}
        {pendingComments.map(p => (
          <div key={p.id} style={{
            ...styles.bubble,
            alignSelf: 'flex-start',
            background: 'rgba(0,0,0,0.45)',
            borderColor: 'rgba(253,224,71,0.35)',
          }}>
            <span style={{ ...styles.bubbleLabel, color: '#fde68a' }}>
              AI Seller (rendering…)
            </span>
            <span style={{ ...styles.bubbleText, color: '#d4d4d8', fontStyle: 'italic' }}>
              responding to "{p.text}"
            </span>
          </div>
        ))}
        <div ref={endRef} />
      </div>

      {/* Floating preset chips — tiny pills, semi-transparent, no border.
          One-tap to populate the input field for fast testing iteration. */}
      <div style={styles.presets}>
        {presetComments.map((c, i) => (
          <button
            key={i}
            onClick={() => setInput(c)}
            style={styles.preset}
          >
            {c}
          </button>
        ))}
      </div>

      {/* Floating input + send. Input is a rounded pill that visually
          anchors to the bottom edge but stays free of any container.
          Send button is the only opaque element — purple cue per the
          existing brand palette. */}
      <div style={styles.inputRow}>
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSend()}
          placeholder="Type a comment…"
          style={styles.input}
        />
        <button onClick={handleSend} style={styles.sendBtn}>
          Send <kbd style={styles.sendKbd}>↵</kbd>
        </button>
      </div>
    </div>
  );
}

const styles = {
  // Transparent root — no background, no border, no shadow. Children
  // float directly over whatever's behind. Pointer-events on so we can
  // still interact with the input + chips.
  root: {
    height: '100%', width: '100%',
    display: 'flex', flexDirection: 'column',
    gap: 6,
    background: 'transparent',
  },
  // Message stream — flex column, scrollable when overflowing. Pushed
  // toward the bottom so initial state has the chips + input visible
  // without a wall of empty grid.
  stream: {
    flex: 1, overflowY: 'auto',
    display: 'flex', flexDirection: 'column',
    gap: 6, paddingRight: 4,
    justifyContent: 'flex-end',
  },
  // Single message — translucent dark pill with subtle border so it
  // reads against any background frame the avatar puts behind.
  // backdrop-filter blurs whatever's underneath for an iOS-style glass
  // bubble that still feels light.
  bubble: {
    padding: '6px 10px',
    borderRadius: 12,
    maxWidth: '90%',
    display: 'flex', flexDirection: 'column',
    gap: 2,
    border: '1px solid rgba(255,255,255,0.08)',
    backdropFilter: 'blur(10px)',
    boxShadow: '0 2px 10px rgba(0,0,0,0.35)',
  },
  bubbleHead: {
    display: 'flex', justifyContent: 'space-between',
    alignItems: 'baseline', gap: 8,
  },
  bubbleLabel: {
    fontSize: 10, fontWeight: 800, letterSpacing: 0.6,
    textTransform: 'uppercase',
  },
  bubbleText: {
    color: '#fafafa', fontSize: 13, lineHeight: 1.35,
  },
  // Preset chips — tiny semi-transparent pills, no border. Wrap so a
  // long list still fits in narrow docks without horizontal overflow.
  presets: {
    display: 'flex', flexWrap: 'wrap', gap: 4,
  },
  preset: {
    background: 'rgba(0,0,0,0.45)',
    color: '#d4d4d8',
    border: 'none',
    borderRadius: 999,
    padding: '3px 9px',
    fontSize: 10, fontWeight: 600,
    letterSpacing: 0.2,
    cursor: 'pointer',
    backdropFilter: 'blur(8px)',
    transition: 'background 120ms ease',
  },
  // Input row — pill-shaped input + the existing purple send button.
  // No container, just two adjacent elements.
  inputRow: {
    display: 'flex', gap: 6, alignItems: 'center',
  },
  input: {
    flex: 1,
    background: 'rgba(0,0,0,0.55)',
    border: '1px solid rgba(255,255,255,0.12)',
    borderRadius: 999,
    padding: '8px 14px',
    color: '#fafafa',
    fontSize: 13,
    outline: 'none',
    backdropFilter: 'blur(10px)',
    boxShadow: '0 2px 8px rgba(0,0,0,0.35)',
  },
  sendBtn: {
    background: '#7c3aed',
    color: '#fff',
    border: 'none',
    borderRadius: 999,
    padding: '8px 14px',
    fontWeight: 700,
    fontSize: 12,
    cursor: 'pointer',
    display: 'inline-flex', alignItems: 'center', gap: 6,
    boxShadow: '0 2px 10px rgba(124,58,237,0.45)',
  },
  sendKbd: {
    display: 'inline-block', minWidth: 16, padding: '1px 5px',
    background: 'rgba(255,255,255,0.22)',
    border: '1px solid rgba(255,255,255,0.3)',
    borderRadius: 4, fontSize: 10, fontWeight: 700,
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    color: '#fff', textAlign: 'center',
  },
  latencyChip: {
    fontSize: 9, fontWeight: 800, color: '#bbf7d0',
    background: 'rgba(22,163,74,0.28)',
    padding: '1px 6px', borderRadius: 999,
    fontVariantNumeric: 'tabular-nums',
  },
};
