import React, { useRef, useEffect, useState } from 'react';

export function AvatarPanel({ status, latestAudio, salesScript }) {
  const audioRef = useRef(null);
  const [speaking, setSpeaking] = useState(false);

  useEffect(() => {
    if (!latestAudio?.audio) return;
    const blob = base64ToBlob(latestAudio.audio, 'audio/mp3');
    const url = URL.createObjectURL(blob);

    if (audioRef.current) {
      audioRef.current.src = url;
      audioRef.current.play().then(() => setSpeaking(true)).catch(() => {});
      audioRef.current.onended = () => setSpeaking(false);
    }

    return () => URL.revokeObjectURL(url);
  }, [latestAudio]);

  const statusColors = {
    idle: '#52525b',
    analyzing: '#3b82f6',
    creating: '#a855f7',
    selling: '#22c55e',
    live: '#22c55e',
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h3 style={styles.title}>AI Seller</h3>
        <div style={{
          padding: '2px 10px', borderRadius: 20, fontSize: 12, fontWeight: 700,
          background: statusColors[status] || '#52525b', color: '#fff',
          textTransform: 'uppercase',
        }}>
          {status}
        </div>
      </div>

      <div style={styles.avatarArea}>
        {status === 'idle' ? (
          <div style={styles.placeholderAvatar}>
            <span style={{ fontSize: 80 }}>🤖</span>
            <p style={{ color: '#52525b', marginTop: 8 }}>Waiting for product...</p>
          </div>
        ) : (
          <div style={styles.activeAvatar}>
            <div style={{
              ...styles.avatarCircle,
              boxShadow: speaking ? '0 0 40px rgba(34,197,94,0.5)' : '0 0 20px rgba(59,130,246,0.3)',
              borderColor: speaking ? '#22c55e' : '#3b82f6',
            }}>
              <span style={{ fontSize: 80 }}>🧑‍💼</span>
            </div>
            {speaking && (
              <div style={styles.speakingIndicator}>
                <div style={styles.soundBar} />
                <div style={{ ...styles.soundBar, animationDelay: '0.1s' }} />
                <div style={{ ...styles.soundBar, animationDelay: '0.2s' }} />
                <div style={{ ...styles.soundBar, animationDelay: '0.3s' }} />
                <div style={{ ...styles.soundBar, animationDelay: '0.4s' }} />
              </div>
            )}
            {salesScript && (
              <p style={styles.scriptPreview}>
                "{salesScript.slice(0, 120)}..."
              </p>
            )}
          </div>
        )}
      </div>

      <audio ref={audioRef} style={{ display: 'none' }} />

      <style>{`
        @keyframes soundbar {
          0%, 100% { height: 8px; }
          50% { height: 24px; }
        }
      `}</style>
    </div>
  );
}

function base64ToBlob(b64, type) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type });
}

const styles = {
  container: { background: '#18181b', borderRadius: 12, padding: 16, height: '100%', display: 'flex', flexDirection: 'column' },
  header: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 },
  title: { color: '#a1a1aa', fontSize: 14, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 1 },
  avatarArea: { flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' },
  placeholderAvatar: { textAlign: 'center' },
  activeAvatar: { textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16 },
  avatarCircle: {
    width: 160, height: 160, borderRadius: 80, border: '3px solid #3b82f6',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    background: '#27272a', transition: 'all 0.3s ease',
  },
  speakingIndicator: { display: 'flex', gap: 4, alignItems: 'flex-end', height: 28 },
  soundBar: {
    width: 4, background: '#22c55e', borderRadius: 2,
    animation: 'soundbar 0.5s ease-in-out infinite',
  },
  scriptPreview: { color: '#71717a', fontSize: 12, fontStyle: 'italic', maxWidth: 300 },
};
