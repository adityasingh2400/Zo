import React from 'react';
import { Spin3D } from './Spin3D';

export function ProductPanel({ productData, productPhoto, transcript, view3d, transcriptExtract }) {
  if (!productData && !productPhoto && !view3d && !transcriptExtract) {
    return (
      <div style={styles.container}>
        <h3 style={styles.title}>Product Intelligence</h3>
        <div style={styles.empty}>
          <span style={{ fontSize: 48 }}>📦</span>
          <p style={{ color: '#52525b' }}>Point at a product and say "sell this"</p>
        </div>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Product Intelligence</h3>
      <div style={styles.content}>
        {view3d ? (
          <Spin3D view={view3d} height={220} />
        ) : productPhoto && (
          <img
            src={`data:image/png;base64,${productPhoto}`}
            alt="Product"
            style={styles.photo}
          />
        )}
        {productData ? (
          <div style={styles.dataGrid}>
            <DataRow label="Name" value={productData.name} />
            <DataRow label="Category" value={productData.category} />
            <DataRow label="Materials" value={Array.isArray(productData.materials) ? productData.materials.join(', ') : productData.materials} />
            <DataRow label="Price Range" value={productData.suggested_price_range} />
            <DataRow label="Target" value={productData.target_audience} />
            {productData.selling_points?.map((pt, i) => (
              <DataRow key={i} label={`Point ${i + 1}`} value={pt} />
            ))}
          </div>
        ) : transcriptExtract && (
          <ExtractPreview extract={transcriptExtract} />
        )}
        {transcript && (
          <div style={{ marginTop: 12, padding: '10px 12px', background: '#27272a', borderRadius: 8 }}>
            <span style={{ color: '#a855f7', fontSize: 12, fontWeight: 700 }}>SELLER NARRATION</span>
            <p style={{ color: '#d4d4d8', fontSize: 13, marginTop: 4, fontStyle: 'italic' }}>"{transcript}"</p>
          </div>
        )}
      </div>
    </div>
  );
}

function DataRow({ label, value }) {
  if (!value) return null;
  return (
    <div style={{ display: 'flex', gap: 8, fontSize: 13 }}>
      <span style={{ color: '#71717a', minWidth: 80, fontWeight: 600 }}>{label}</span>
      <span style={{ color: '#e4e4e7' }}>{value}</span>
    </div>
  );
}

// Lightweight preview rendered while Claude vision is still in flight.
// Shows on-device structured signals so the panel isn't empty for ~3s.
function ExtractPreview({ extract }) {
  const { name_hint, category_hint, claims, selling_points,
          target_audience_hint, price_hint, source, latency_ms } = extract;
  return (
    <div style={styles.dataGrid}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11 }}>
        <span style={{ color: '#a855f7', fontWeight: 700, letterSpacing: 0.5 }}>ON-DEVICE PREVIEW</span>
        <span style={{ color: '#52525b' }}>· {source} · {latency_ms}ms</span>
      </div>
      <DataRow label="Name" value={name_hint} />
      <DataRow label="Category" value={category_hint} />
      <DataRow label="Price" value={price_hint} />
      <DataRow label="Target" value={target_audience_hint} />
      {claims?.slice(0, 3).map((c, i) => (
        <DataRow key={`c${i}`} label={i === 0 ? 'Claims' : ''} value={c} />
      ))}
      {selling_points?.slice(0, 3).map((s, i) => (
        <DataRow key={`s${i}`} label={i === 0 ? 'Hooks' : ''} value={s} />
      ))}
      <div style={{ color: '#52525b', fontSize: 11, marginTop: 4, fontStyle: 'italic' }}>
        Claude vision will refine this in a moment...
      </div>
    </div>
  );
}

const styles = {
  container: { background: '#18181b', borderRadius: 12, padding: 16, height: '100%', display: 'flex', flexDirection: 'column' },
  title: { color: '#a1a1aa', fontSize: 14, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 },
  empty: { flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12 },
  content: { flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 16 },
  photo: { width: '100%', maxHeight: 200, objectFit: 'contain', borderRadius: 8, background: '#fff' },
  dataGrid: { display: 'flex', flexDirection: 'column', gap: 8 },
};
