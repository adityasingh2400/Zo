import React, { useEffect, useState, useCallback } from 'react';

/**
 * ProductSelector — dropdown to switch the active product.
 *
 * Fetches /api/state on mount to populate the catalog + active id. On
 * change, POSTs /api/state/active_product. The backend broadcasts an
 * `active_product_changed` WS event; we don't subscribe here (App-level
 * useEmpireSocket would re-pull state) — for v0 we just refetch our own
 * state after the POST so the dropdown reflects the change immediately.
 *
 * Limitation: doesn't reflect catalog changes pushed from another
 * dashboard tab. The intended use is a single operator at the demo
 * machine; this is fine until multi-operator (roadmap).
 */
const API_BASE = `http://${window.location.hostname}:8000`;

export function ProductSelector() {
  const [products, setProducts] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [busy, setBusy] = useState(false);

  const refetch = useCallback(async () => {
    try {
      const r = await fetch(`${API_BASE}/api/state`);
      if (!r.ok) return;
      const data = await r.json();
      setProducts(data.products || []);
      setActiveId(data.active_product_id || null);
    } catch {
      // Silent — selector is non-critical chrome.
    }
  }, []);

  useEffect(() => { refetch(); }, [refetch]);

  async function onChange(e) {
    const newId = e.target.value;
    if (!newId || newId === activeId) return;
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append('product_id', newId);
      const r = await fetch(`${API_BASE}/api/state/active_product`, {
        method: 'POST', body: fd,
      });
      if (r.ok) {
        const data = await r.json();
        setActiveId(data.active_product_id);
      }
    } finally {
      setBusy(false);
    }
  }

  // Hide the selector entirely if there's only one product (no decision
  // for the operator to make). Returns once multi-product onboarding lands
  // (roadmap) and a real seller will have multiple products to choose from.
  if (products.length <= 1) return null;

  return (
    <div style={styles.wrap}>
      <span style={styles.label}>PRODUCT</span>
      <select
        value={activeId || ''}
        onChange={onChange}
        disabled={busy}
        style={{ ...styles.select, opacity: busy ? 0.6 : 1 }}
      >
        {products.map(p => (
          <option key={p.id} value={p.id}>
            {p.name} ({p.qa_count} Q/A)
          </option>
        ))}
      </select>
    </div>
  );
}

const styles = {
  wrap: {
    display: 'flex', alignItems: 'center', gap: 8,
    background: '#18181b', border: '1px solid #3f3f46',
    borderRadius: 8, padding: '8px 12px',
  },
  label: {
    fontSize: 10, fontWeight: 800, letterSpacing: 1.5,
    color: '#71717a', textTransform: 'uppercase',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  },
  select: {
    background: '#09090b', color: '#fafafa',
    border: '1px solid #3f3f46', borderRadius: 6,
    padding: '6px 10px', fontSize: 13, outline: 'none',
    fontFamily: 'inherit',
    cursor: 'pointer',
  },
};
