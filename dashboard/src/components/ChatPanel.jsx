import React, { useState, useRef, useEffect } from 'react';

export function ChatPanel({ onSendComment, commentResponse, pendingComments = [] }) {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, pendingComments]);

  // Append a response row whenever a new commentResponse arrives.
  // We key on response text + total_ms so duplicates are skipped.
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
    <div style={styles.container}>
      <h3 style={styles.title}>Live Chat</h3>

      <div style={styles.messages}>
        {messages.map((msg, i) => (
          <div key={i} style={{
            ...styles.message,
            alignSelf: msg.type === 'comment' ? 'flex-end' : 'flex-start',
            background: msg.type === 'comment' ? '#3b82f6' : '#27272a',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 8 }}>
              <span style={{ fontSize: 11, color: msg.type === 'comment' ? '#93c5fd' : '#22c55e', fontWeight: 700 }}>
                {msg.type === 'comment' ? 'Viewer' : 'AI Seller'}
              </span>
              {msg.totalMs && (
                <span style={styles.latencyChip}>⚡ {(msg.totalMs / 1000).toFixed(1)}s</span>
              )}
            </div>
            <span style={{ color: '#fafafa', fontSize: 13 }}>{msg.text}</span>
          </div>
        ))}
        {pendingComments.map(p => (
          <div key={p.id} style={{ ...styles.message, alignSelf: 'flex-start', background: '#3f3f46', opacity: 0.85 }}>
            <span style={{ fontSize: 11, color: '#fde68a', fontWeight: 700 }}>AI Seller (rendering…)</span>
            <span style={{ color: '#a1a1aa', fontSize: 13, fontStyle: 'italic' }}>responding to "{p.text}"</span>
          </div>
        ))}
        <div ref={endRef} />
      </div>

      <div style={styles.presets}>
        {presetComments.map((c, i) => (
          <button
            key={i}
            onClick={() => { setInput(c); }}
            style={styles.preset}
          >
            {c}
          </button>
        ))}
      </div>

      <div style={styles.inputRow}>
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSend()}
          placeholder="Type a comment..."
          style={styles.input}
        />
        <button onClick={handleSend} style={styles.sendBtn}>
          {/* Recognition Over Recall (Lidwell p164): Enter-to-send is wired,
              but invisible behavior is no behavior at all to a first-time
              user. The kbd chip surfaces the shortcut at the exact moment
              the user is deciding how to commit. */}
          Send <kbd style={styles.sendKbd}>↵</kbd>
        </button>
      </div>
    </div>
  );
}

const styles = {
  container: { background: '#18181b', borderRadius: 12, padding: 16, height: '100%', display: 'flex', flexDirection: 'column' },
  title: { color: '#a1a1aa', fontSize: 14, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 },
  messages: { flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 12 },
  message: { padding: '8px 12px', borderRadius: 10, maxWidth: '80%', display: 'flex', flexDirection: 'column', gap: 2 },
  presets: { display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 8 },
  preset: {
    background: '#27272a', color: '#a1a1aa', border: '1px solid #3f3f46', borderRadius: 16,
    padding: '4px 10px', fontSize: 11, cursor: 'pointer',
  },
  inputRow: { display: 'flex', gap: 8 },
  input: {
    flex: 1, background: '#27272a', border: '1px solid #3f3f46', borderRadius: 8,
    padding: '10px 12px', color: '#fafafa', fontSize: 14, outline: 'none',
  },
  sendBtn: {
    background: '#7c3aed', color: '#fff', border: 'none', borderRadius: 8,
    padding: '10px 16px', fontWeight: 700, cursor: 'pointer',
    display: 'inline-flex', alignItems: 'center', gap: 8,
  },
  sendKbd: {
    display: 'inline-block', minWidth: 18, padding: '1px 6px',
    background: 'rgba(255,255,255,0.2)', border: '1px solid rgba(255,255,255,0.3)',
    borderRadius: 4, fontSize: 11, fontWeight: 700,
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    color: '#fff', textAlign: 'center',
  },
  latencyChip: {
    fontSize: 10, fontWeight: 800, color: '#bbf7d0',
    background: 'rgba(22,163,74,0.25)', padding: '2px 6px', borderRadius: 999,
    fontVariantNumeric: 'tabular-nums',
  },
};
