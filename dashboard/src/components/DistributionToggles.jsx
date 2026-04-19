import React, { useEffect, useState } from 'react';

/**
 * DistributionToggles — TikTok Shop / Shopify Live / Etsy / Instagram Live
 * toggles matching the empire-mac.jsx mockup's distribution section.
 *
 * v1: local state only. Toggles don't actually fan out — Item 5 (Hands
 * agent) wires these to real /api/hands/publish calls with a mock adapter
 * set (TikTok/Etsy/Instagram) + real Shopify Storefront API.
 *
 * When Item 5 ships, this component will:
 *   - GET /api/hands/state on mount to hydrate enabled platforms
 *   - POST /api/hands/publish on toggle
 *   - Subscribe via useEmpireSocket to hands_published events
 *
 * For now: localStorage persistence so toggles survive reloads.
 */
const API_BASE = `http://${window.location.hostname}:8000`;
const STORAGE_KEY = 'EMPIRE_DISTRIBUTION';

const PLATFORMS = [
  { k: 'tiktok',    label: 'TikTok Shop',    sub: '@maya.makes · 12,408 watching' },
  { k: 'shopify',   label: 'Shopify Live',   sub: 'anagamastudio.myshopify.com' },
  { k: 'etsy',      label: 'Etsy',           sub: 'mirror to UGC only (no live api)' },
  { k: 'instagram', label: 'Instagram Live', sub: 'UGC fanout · 9:16' },
];

const DEFAULTS = {
  tiktok: true, shopify: true, etsy: false, instagram: false,
};

export function DistributionToggles() {
  const [state, setState] = useState(DEFAULTS);
  const [handsReady, setHandsReady] = useState(false);

  // Hydrate from localStorage on mount, then try real Hands state.
  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) setState({ ...DEFAULTS, ...JSON.parse(saved) });
    } catch { /* corrupt localStorage entry — ignore */ }

    (async () => {
      try {
        const r = await fetch(`${API_BASE}/api/hands/state`);
        if (!r.ok) return;
        const data = await r.json();
        if (data?.platforms) {
          const next = { ...DEFAULTS };
          for (const [k, v] of Object.entries(data.platforms)) {
            next[k] = !!v.enabled;
          }
          setState(next);
          setHandsReady(true);
        }
      } catch { /* Hands not live yet — v1 mode */ }
    })();
  }, []);

  function toggle(k) {
    const next = { ...state, [k]: !state[k] };
    setState(next);
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(next)); }
    catch { /* ignore quota errors */ }

    if (handsReady) {
      // Fire-and-forget: the real actuation call. Backend emits a
      // hands_published event which MetricsStrip consumes.
      const fd = new FormData();
      fd.append('platform', k);
      fd.append('enabled', String(next[k]));
      fetch(`${API_BASE}/api/hands/toggle`, { method: 'POST', body: fd })
        .catch(() => { /* swallow — toggle state already set locally */ });
    }
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>Distribution</div>
      {PLATFORMS.map((p, i) => (
        <div key={p.k} style={{
          ...styles.row,
          borderTop: i === 0 ? 'none' : '1px solid rgba(0,0,0,0.05)',
        }}>
          <div style={{ flex: 1 }}>
            <div style={styles.label}>{p.label}</div>
            <div style={styles.sub}>{p.sub}</div>
          </div>
          <button
            type="button"
            onClick={() => toggle(p.k)}
            style={{
              ...styles.switch,
              background: state[p.k] ? '#1d1d1f' : '#d4d4d7',
              justifyContent: state[p.k] ? 'flex-end' : 'flex-start',
            }}
            aria-pressed={state[p.k]}
          >
            <span style={styles.switchKnob} />
          </button>
        </div>
      ))}
    </div>
  );
}

const styles = {
  container: {
    padding: 14,
    background: '#fff',
    border: '1px solid rgba(0,0,0,0.08)',
    borderRadius: 12,
    fontFamily: '-apple-system, "SF Pro Text", "Inter", system-ui, sans-serif',
  },
  header: {
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    fontSize: 10, letterSpacing: 1.2,
    textTransform: 'uppercase', color: '#86868b', fontWeight: 600,
    marginBottom: 6,
  },
  row: {
    display: 'flex', alignItems: 'center', gap: 12,
    padding: '8px 0',
  },
  label: { fontSize: 13, fontWeight: 500, color: '#1d1d1f' },
  sub: {
    fontSize: 11, color: '#86868b',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  },
  switch: {
    width: 36, height: 20, borderRadius: 999, border: 'none', padding: 2,
    cursor: 'pointer', display: 'flex',
    transition: 'all 220ms',
  },
  switchKnob: {
    width: 16, height: 16, borderRadius: '50%',
    background: '#fff', boxShadow: '0 1px 2px rgba(0,0,0,0.2)',
  },
};
