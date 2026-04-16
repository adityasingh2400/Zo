import React, { useState, useRef, useEffect } from 'react';

export function ChatPanel({ onSendComment, commentResponse }) {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (!commentResponse) return;
    setMessages(prev => [...prev, {
      type: 'response',
      text: commentResponse.response,
      timestamp: Date.now(),
    }]);
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
            <span style={{ fontSize: 11, color: msg.type === 'comment' ? '#93c5fd' : '#22c55e', fontWeight: 700 }}>
              {msg.type === 'comment' ? 'Viewer' : 'AI Seller'}
            </span>
            <span style={{ color: '#fafafa', fontSize: 13 }}>{msg.text}</span>
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
        <button onClick={handleSend} style={styles.sendBtn}>Send</button>
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
    padding: '10px 20px', fontWeight: 700, cursor: 'pointer',
  },
};
