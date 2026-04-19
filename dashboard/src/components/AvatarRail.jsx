import React, { useEffect, useState, useCallback } from 'react';

/**
 * AvatarRail — vertical rail of avatar cards matching the empire-mac.jsx
 * mockup. Four avatars: Maya (en·es·tl), Leo (en·es), Mei (zh·en),
 * Jasper (en), with a stub "+ New avatar" CTA at the bottom.
 *
 * v1 data source: tries GET /api/avatars first. Falls back to a static
 * 4-avatar manifest so this renders even before Item 4 ships the backend
 * catalog. When Item 4 lands, the component automatically picks up real
 * data + wires POST /api/avatars/active on select.
 */
const API_BASE = `http://${window.location.hostname}:8000`;

const FALLBACK_AVATARS = [
  { id: 'maya',   name: 'Maya',   language_tags: ['en','es','tl'] },
  { id: 'leo',    name: 'Leo',    language_tags: ['en','es'] },
  { id: 'mei',    name: 'Mei',    language_tags: ['zh','en'] },
  { id: 'jasper', name: 'Jasper', language_tags: ['en'] },
];

export function AvatarRail() {
  const [avatars, setAvatars] = useState(FALLBACK_AVATARS);
  const [activeId, setActiveId] = useState('maya');
  const [busy, setBusy] = useState(false);
  const [hasBackend, setHasBackend] = useState(false);

  const refetch = useCallback(async () => {
    try {
      const r = await fetch(`${API_BASE}/api/avatars`);
      if (!r.ok) return;
      const data = await r.json();
      if (Array.isArray(data.avatars) && data.avatars.length > 0) {
        setAvatars(data.avatars);
        setActiveId(data.active_avatar_id || data.avatars[0]?.id || 'maya');
        setHasBackend(true);
      }
    } catch {
      // Backend /api/avatars not live yet — v1 renders the static list.
    }
  }, []);

  useEffect(() => { refetch(); }, [refetch]);

  async function selectAvatar(id) {
    if (busy || id === activeId) return;
    setActiveId(id);  // optimistic
    if (!hasBackend) return;
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append('avatar_id', id);
      const r = await fetch(`${API_BASE}/api/avatars/active`, {
        method: 'POST', body: fd,
      });
      if (!r.ok) refetch();  // revert
    } catch {
      refetch();
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>Avatars</div>
      <div style={styles.list}>
        {avatars.map(a => {
          const sel = a.id === activeId;
          return (
            <button
              key={a.id}
              type="button"
              onClick={() => selectAvatar(a.id)}
              style={{
                ...styles.card,
                border: sel ? '1px solid #1d1d1f' : '1px solid rgba(0,0,0,0.08)',
                background: sel ? 'rgba(0,0,0,0.03)' : '#fff',
              }}
            >
              <div style={styles.thumbnail}>
                {sel && (
                  <div style={styles.selectedDot}>●</div>
                )}
              </div>
              <div style={styles.labelRow}>
                <span style={styles.name}>{a.name}</span>
                <span style={styles.tags}>
                  {(a.language_tags || []).join('·')}
                </span>
              </div>
            </button>
          );
        })}
      </div>
      <button type="button" style={styles.newAvatar}>+ New avatar</button>
    </div>
  );
}

const styles = {
  container: {
    padding: 10,
    background: '#fff',
    border: '1px solid rgba(0,0,0,0.08)',
    borderRadius: 12,
    display: 'flex', flexDirection: 'column', gap: 8,
    minHeight: 0, overflow: 'hidden',
    width: 128,
    fontFamily: '-apple-system, "SF Pro Text", "Inter", system-ui, sans-serif',
  },
  header: {
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    fontSize: 9, letterSpacing: 1,
    textTransform: 'uppercase', color: '#86868b', fontWeight: 600,
    padding: '2px 2px 4px',
  },
  list: {
    display: 'flex', flexDirection: 'column', gap: 7, flex: 1,
    overflowY: 'auto', minHeight: 0,
  },
  card: {
    padding: 6,
    borderRadius: 10,
    cursor: 'pointer', textAlign: 'left',
    display: 'flex', flexDirection: 'column', gap: 4,
    fontFamily: 'inherit',
  },
  thumbnail: {
    width: '100%', aspectRatio: '1', borderRadius: 7,
    background: 'repeating-linear-gradient(135deg,#f4f4f6 0 8px,#ededf0 8px 9px)',
    position: 'relative',
  },
  selectedDot: {
    position: 'absolute', top: 4, right: 4,
    width: 14, height: 14, borderRadius: '50%',
    background: '#1d1d1f', color: '#fff',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    fontSize: 8,
  },
  labelRow: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
    padding: '0 2px',
  },
  name: { fontSize: 11, fontWeight: 600, color: '#1d1d1f' },
  tags: {
    fontSize: 9, color: '#86868b',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  },
  newAvatar: {
    padding: '7px 8px',
    border: '1px dashed rgba(0,0,0,0.08)',
    borderRadius: 8, background: 'transparent',
    fontSize: 11, color: '#6e6e73', cursor: 'pointer',
    fontFamily: 'inherit',
  },
};
