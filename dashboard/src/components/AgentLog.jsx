import React, { useRef, useEffect } from 'react';

export function AgentLog({ log }) {
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [log]);

  const agentColors = {
    EYES: '#3b82f6',
    CREATOR: '#a855f7',
    SELLER: '#22c55e',
    SYSTEM: '#f59e0b',
  };

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Agent Activity</h3>
      <div style={styles.logArea}>
        {log.map((entry, i) => {
          const ts = new Date(entry.timestamp * 1000);
          const timeStr = ts.toLocaleTimeString('en', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
          const color = agentColors[entry.agent] || '#71717a';

          return (
            <div key={i} style={styles.entry}>
              <span style={{ color: '#52525b', fontFamily: 'monospace', fontSize: 12 }}>{timeStr}</span>
              <span style={{ color, fontWeight: 700, fontSize: 13, minWidth: 70 }}>{entry.agent}</span>
              <span style={{ color: '#d4d4d8', fontSize: 13, flex: 1 }}>{entry.message}</span>
            </div>
          );
        })}
        <div ref={endRef} />
      </div>
    </div>
  );
}

const styles = {
  container: { background: '#18181b', borderRadius: 12, padding: 16, height: '100%', display: 'flex', flexDirection: 'column' },
  title: { color: '#a1a1aa', fontSize: 14, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 },
  logArea: { flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 6 },
  entry: { display: 'flex', gap: 8, alignItems: 'baseline' },
};
