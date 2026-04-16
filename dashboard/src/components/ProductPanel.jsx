import React from 'react';

export function ProductPanel({ productData, productPhoto }) {
  if (!productData && !productPhoto) {
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
        {productPhoto && (
          <img
            src={`data:image/png;base64,${productPhoto}`}
            alt="Product"
            style={styles.photo}
          />
        )}
        {productData && (
          <div style={styles.dataGrid}>
            <DataRow label="Name" value={productData.name} />
            <DataRow label="Category" value={productData.category} />
            <DataRow label="Materials" value={productData.materials?.join(', ')} />
            <DataRow label="Price Range" value={productData.suggested_price_range} />
            <DataRow label="Target" value={productData.target_audience} />
            {productData.selling_points?.map((pt, i) => (
              <DataRow key={i} label={`Point ${i + 1}`} value={pt} />
            ))}
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

const styles = {
  container: { background: '#18181b', borderRadius: 12, padding: 16, height: '100%', display: 'flex', flexDirection: 'column' },
  title: { color: '#a1a1aa', fontSize: 14, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 },
  empty: { flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12 },
  content: { flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 16 },
  photo: { width: '100%', maxHeight: 200, objectFit: 'contain', borderRadius: 8, background: '#fff' },
  dataGrid: { display: 'flex', flexDirection: 'column', gap: 8 },
};
