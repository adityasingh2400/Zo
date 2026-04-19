import React from 'react';

/**
 * StudioSidebar — left-rail navigation matching the empire-mac.jsx mockup.
 *
 * Four sections: STUDIO (the operating surfaces), DISTRIBUTION (platforms
 * a response can fan out to), AGENTS (the swarm from the PDF), INFRA (the
 * compute substrate).
 *
 * v1 scope: pure visual nav. Selected item is local state; selection doesn't
 * currently change what renders — the cinema grid is always "Live Control
 * Room." Item 5+6 (Hands + multi-language) will wire the distribution and
 * agents sections to real toggles.
 */
const SECTIONS = [
  {
    title: 'STUDIO',
    items: [
      { id: 'live',    label: 'Live Control Room' },
      { id: 'intake',  label: 'Intake Queue' },
      { id: 'catalog', label: 'Catalog · 2' },
      { id: 'avatars', label: 'Avatars · 1' },
    ],
  },
  {
    title: 'DISTRIBUTION',
    items: [
      { id: 'tiktok',    label: 'TikTok Shop' },
      { id: 'shopify',   label: 'Shopify Live' },
      { id: 'etsy',      label: 'Etsy' },
      { id: 'global',    label: 'Global Launch' },
      { id: 'ugc',       label: 'UGC Mode' },
    ],
  },
  {
    title: 'AGENTS',
    items: [
      { id: 'eyes',     label: 'Eyes · perception' },
      { id: 'seller',   label: 'Seller · spoken' },
      { id: 'director', label: 'Director · stage' },
      { id: 'hands',    label: 'Hands · actuation' },
    ],
  },
  {
    title: 'INFRA',
    items: [
      { id: 'pod',    label: 'Pod · 5090' },
      { id: 'eleven', label: 'Voice · Eleven' },
      { id: 'gemma',  label: 'On-device · Gemma 4' },
    ],
  },
];

export function StudioSidebar({ selectedId = 'live', onSelect }) {
  return (
    <div style={styles.container}>
      {SECTIONS.map(section => (
        <div key={section.title} style={styles.section}>
          <div style={styles.sectionHeader}>{section.title}</div>
          {section.items.map(item => {
            const selected = item.id === selectedId;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onSelect?.(item.id)}
                style={{
                  ...styles.item,
                  ...(selected ? styles.itemSelected : {}),
                }}
              >
                <span style={{
                  ...styles.dot,
                  background: selected ? '#22c55e' : '#3f3f46',
                }} />
                {item.label}
              </button>
            );
          })}
        </div>
      ))}
    </div>
  );
}

const styles = {
  container: {
    display: 'flex', flexDirection: 'column',
    padding: '16px 10px',
    background: '#0a0a0b',
    borderRight: '1px solid #18181b',
    fontFamily: '-apple-system, "SF Pro Text", "Inter", system-ui, sans-serif',
    color: '#fafafa',
    width: 200, minWidth: 200, flexShrink: 0,
    gap: 16,
    overflowY: 'auto',
  },
  section: {
    display: 'flex', flexDirection: 'column', gap: 2,
  },
  sectionHeader: {
    fontSize: 10, fontWeight: 700, letterSpacing: 1.5,
    color: '#52525b', padding: '4px 10px 8px',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  },
  item: {
    display: 'flex', alignItems: 'center', gap: 8,
    background: 'transparent', border: 'none',
    padding: '7px 10px', borderRadius: 7,
    color: '#d4d4d8', fontSize: 13,
    textAlign: 'left', cursor: 'pointer',
    fontFamily: 'inherit',
  },
  itemSelected: {
    background: 'rgba(255,255,255,0.06)',
    color: '#fafafa', fontWeight: 500,
  },
  dot: {
    width: 6, height: 6, borderRadius: '50%', flexShrink: 0,
  },
};
