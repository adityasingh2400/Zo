// empire-components.jsx — shared primitives for the three variations

// ─────────────────────────────────────────────────────────────
// Tokens — Apple-spare, white, hairline
// ─────────────────────────────────────────────────────────────
const E = {
  bg: '#fbfbfd',
  card: '#ffffff',
  ink: '#1d1d1f',
  ink2: '#6e6e73',
  ink3: '#86868b',
  hair: 'rgba(0,0,0,0.08)',
  hair2: 'rgba(0,0,0,0.05)',
  live: 'oklch(0.70 0.14 155)', // calm green
  aura: 'oklch(0.72 0.12 220)', // cool blue aura for speaking
  warn: 'oklch(0.72 0.12 60)',
  font: '-apple-system, "SF Pro Text", "SF Pro Display", "Inter", system-ui, sans-serif',
  mono: '"SF Mono", "JetBrains Mono", ui-monospace, Menlo, monospace',
  shadow: '0 1px 2px rgba(0,0,0,0.04), 0 8px 24px rgba(0,0,0,0.04)',
  shadowHi: '0 1px 2px rgba(0,0,0,0.06), 0 20px 48px rgba(0,0,0,0.08)',
};

// ─────────────────────────────────────────────────────────────
// Avatar placeholder — striped portrait w/ speaking pulse aura
// ─────────────────────────────────────────────────────────────
function AvatarPortrait({ speaking, caption, label = 'Avatar · Maya', size = 'lg' }) {
  const h = size === 'lg' ? '100%' : size === 'md' ? 360 : 220;
  return (
    <div style={{
      position: 'relative',
      width: '100%', height: h,
      borderRadius: 18,
      overflow: 'hidden',
      background: '#fff',
      boxShadow: 'inset 0 0 0 1px rgba(0,0,0,0.06)',
    }}>
      {/* diagonal-stripe placeholder */}
      <div style={{
        position: 'absolute', inset: 0,
        backgroundImage: `repeating-linear-gradient(
          135deg,
          #f4f4f6 0 14px,
          #ededf0 14px 15px
        )`,
      }} />
      {/* soft portrait vignette to suggest a head/shoulders */}
      <svg viewBox="0 0 400 500" preserveAspectRatio="xMidYMid slice"
        style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}>
        <defs>
          <radialGradient id="head" cx="50%" cy="36%" r="28%">
            <stop offset="0%" stopColor="rgba(255,255,255,0.95)" />
            <stop offset="70%" stopColor="rgba(255,255,255,0.35)" />
            <stop offset="100%" stopColor="rgba(255,255,255,0)" />
          </radialGradient>
          <radialGradient id="torso" cx="50%" cy="95%" r="55%">
            <stop offset="0%" stopColor="rgba(255,255,255,0.7)" />
            <stop offset="100%" stopColor="rgba(255,255,255,0)" />
          </radialGradient>
        </defs>
        <rect width="400" height="500" fill="url(#torso)" />
        <ellipse cx="200" cy="180" rx="110" ry="130" fill="url(#head)" />
      </svg>
      {/* speaking aura */}
      {speaking && (
        <div style={{
          position: 'absolute', inset: 0,
          background: `radial-gradient(60% 50% at 50% 45%, ${E.aura} 0%, transparent 70%)`,
          mixBlendMode: 'multiply',
          opacity: 0.18,
          animation: 'empireAura 2.2s ease-in-out infinite',
          pointerEvents: 'none',
        }} />
      )}
      {/* center label */}
      <div style={{
        position: 'absolute', left: '50%', top: '50%',
        transform: 'translate(-50%,-50%)',
        fontFamily: E.mono, fontSize: 11, letterSpacing: 0.8,
        textTransform: 'uppercase',
        color: 'rgba(0,0,0,0.42)',
        textAlign: 'center', lineHeight: 1.6,
      }}>
        {label}<br/>
        <span style={{ opacity: 0.5 }}>1080p · Veo 3.1 · Wav2Lip</span>
      </div>
      {/* live pill */}
      <div style={{
        position: 'absolute', top: 14, left: 14,
        display: 'inline-flex', alignItems: 'center', gap: 6,
        padding: '5px 10px 5px 8px',
        background: 'rgba(255,255,255,0.9)',
        backdropFilter: 'blur(10px)',
        WebkitBackdropFilter: 'blur(10px)',
        borderRadius: 999,
        fontFamily: E.mono, fontSize: 10, fontWeight: 600,
        letterSpacing: 0.8,
        color: E.ink,
      }}>
        <span style={{
          width: 7, height: 7, borderRadius: '50%',
          background: E.live,
          boxShadow: `0 0 0 4px ${E.live}22`,
          animation: 'empirePulse 1.4s ease-in-out infinite',
        }} />
        LIVE
      </div>
      {/* caption — what the avatar is saying */}
      {caption && (
        <div style={{
          position: 'absolute', left: 14, right: 14, bottom: 14,
          padding: '12px 14px',
          background: 'rgba(0,0,0,0.72)',
          color: '#fff',
          backdropFilter: 'blur(12px)',
          WebkitBackdropFilter: 'blur(12px)',
          borderRadius: 12,
          fontSize: 14, lineHeight: 1.45, fontWeight: 500,
          letterSpacing: -0.1,
        }}>
          {caption}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Stage bar — INTRO → BRIDGE → PITCH → LIVE
// ─────────────────────────────────────────────────────────────
function StageBar({ phase, compact }) {
  const phases = ['INTRO', 'BRIDGE', 'PITCH', 'LIVE'];
  const idx = phases.indexOf(phase);
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: compact ? 10 : 16,
    }}>
      {phases.map((p, i) => {
        const active = i === idx;
        const done = i < idx;
        return (
          <React.Fragment key={p}>
            <div style={{
              display: 'flex', alignItems: 'center', gap: 8,
              padding: compact ? '5px 10px' : '7px 14px',
              borderRadius: 999,
              background: active ? E.ink : 'transparent',
              color: active ? '#fff' : (done ? E.ink : E.ink3),
              border: active ? 'none' : `1px solid ${E.hair}`,
              fontFamily: E.mono,
              fontSize: compact ? 10 : 11,
              fontWeight: 600, letterSpacing: 0.8,
              transition: 'all 300ms cubic-bezier(.2,.8,.2,1)',
            }}>
              <span style={{
                width: 6, height: 6, borderRadius: '50%',
                background: active ? E.live : (done ? E.ink : 'transparent'),
                border: done && !active ? 'none' : active ? 'none' : `1px solid ${E.ink3}`,
              }} />
              {p}
            </div>
            {i < phases.length - 1 && (
              <div style={{
                width: compact ? 14 : 22, height: 1,
                background: i < idx ? E.ink : E.hair,
                transition: 'background 300ms',
              }} />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Latency badge (monospace, corner placement)
// ─────────────────────────────────────────────────────────────
function LatencyBadge({ ms = 5600, label = 'p50' }) {
  return (
    <div style={{
      display: 'inline-flex', alignItems: 'baseline', gap: 6,
      padding: '6px 10px',
      background: E.card,
      border: `1px solid ${E.hair}`,
      borderRadius: 8,
      fontFamily: E.mono,
    }}>
      <span style={{ fontSize: 10, color: E.ink3, letterSpacing: 0.6 }}>
        {label.toUpperCase()}
      </span>
      <span style={{ fontSize: 12, color: E.ink, fontWeight: 600 }}>
        {(ms / 1000).toFixed(1)}s
      </span>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Comment chip — floats in from the right
// ─────────────────────────────────────────────────────────────
function CommentChip({ handle, text, replying, when }) {
  return (
    <div style={{
      padding: '12px 14px',
      background: E.card,
      border: `1px solid ${E.hair}`,
      borderRadius: 14,
      boxShadow: E.shadow,
      display: 'flex', gap: 12,
    }}>
      <div style={{
        flexShrink: 0,
        width: 30, height: 30, borderRadius: '50%',
        background: `repeating-linear-gradient(45deg, #eee 0 4px, #e4e4e7 4px 5px)`,
        border: `1px solid ${E.hair}`,
      }} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          marginBottom: 3,
        }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: E.ink }}>
            {handle}
          </div>
          <div style={{ fontFamily: E.mono, fontSize: 10, color: E.ink3 }}>
            {when}
          </div>
        </div>
        <div style={{ fontSize: 13, color: E.ink, lineHeight: 1.4 }}>
          {text}
        </div>
        {replying && (
          <div style={{
            marginTop: 8,
            display: 'inline-flex', alignItems: 'center', gap: 6,
            fontFamily: E.mono, fontSize: 10,
            color: E.aura, fontWeight: 600, letterSpacing: 0.6,
          }}>
            <span style={{
              width: 6, height: 6, borderRadius: '50%', background: E.aura,
              animation: 'empirePulse 1.2s ease-in-out infinite',
            }} />
            MAYA REPLYING · WAV2LIP RENDERING
          </div>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Agent log — WebSocket bus trace, monospace, 4 agents
// ─────────────────────────────────────────────────────────────
const AGENT_COLORS = {
  eyes:     'oklch(0.72 0.12 260)',
  seller:   'oklch(0.70 0.14 155)',
  director: 'oklch(0.72 0.13 310)',
  hands:    'oklch(0.74 0.12 60)',
};

function AgentLine({ agent, message, t }) {
  return (
    <div style={{
      display: 'grid', gridTemplateColumns: '52px 78px 1fr',
      gap: 12, alignItems: 'baseline',
      padding: '7px 0',
      borderBottom: `1px solid ${E.hair2}`,
      fontFamily: E.mono, fontSize: 11, lineHeight: 1.5,
    }}>
      <span style={{ color: E.ink3 }}>{t}</span>
      <span style={{
        color: AGENT_COLORS[agent] || E.ink,
        textTransform: 'uppercase', fontWeight: 600, letterSpacing: 0.6,
      }}>{agent}</span>
      <span style={{ color: E.ink, whiteSpace: 'pre-wrap' }}>{message}</span>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Product card — item the avatar is pitching, with visual details
// ─────────────────────────────────────────────────────────────
function ProductCard({ name = 'Matte Black Pour-Over', sku = 'ANAGAMA-01', price = '$48', details }) {
  return (
    <div style={{
      padding: 18,
      background: E.card,
      border: `1px solid ${E.hair}`,
      borderRadius: 16,
    }}>
      <div style={{
        width: '100%', aspectRatio: '4 / 3',
        borderRadius: 12,
        background: `repeating-linear-gradient(135deg, #f4f4f6 0 14px, #ededf0 14px 15px)`,
        position: 'relative',
        overflow: 'hidden',
        marginBottom: 14,
      }}>
        <div style={{
          position: 'absolute', inset: 0,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontFamily: E.mono, fontSize: 11,
          color: 'rgba(0,0,0,0.42)', letterSpacing: 0.8,
        }}>PRODUCT · {sku}</div>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <div style={{ fontSize: 15, fontWeight: 600, color: E.ink, letterSpacing: -0.2 }}>
          {name}
        </div>
        <div style={{ fontFamily: E.mono, fontSize: 13, color: E.ink }}>
          {price}
        </div>
      </div>
      <div style={{ marginTop: 10, fontSize: 11, fontFamily: E.mono, color: E.ink3, letterSpacing: 0.4 }}>
        VISUAL DETAILS GROUNDED
      </div>
      <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
        {(details || ['matte ceramic', 'walnut handle', 'spiral ridges', '60mm cone']).map(d => (
          <span key={d} style={{
            padding: '4px 8px',
            border: `1px solid ${E.hair}`,
            borderRadius: 999,
            fontSize: 11, color: E.ink,
          }}>{d}</span>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Section header — uppercase tiny label + optional right-side slot
// ─────────────────────────────────────────────────────────────
function Eyebrow({ children, right, style = {} }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      marginBottom: 10, ...style,
    }}>
      <div style={{
        fontFamily: E.mono, fontSize: 10, letterSpacing: 1.2,
        textTransform: 'uppercase', color: E.ink3, fontWeight: 600,
      }}>{children}</div>
      {right}
    </div>
  );
}

// Global keyframes — injected once
function EmpireGlobalStyles() {
  return (
    <style>{`
      @keyframes empirePulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.45; transform: scale(0.82); }
      }
      @keyframes empireAura {
        0%, 100% { opacity: 0.16; transform: scale(1); }
        50% { opacity: 0.32; transform: scale(1.05); }
      }
      @keyframes empireFadeUp {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
      }
      @keyframes empireShimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
      }
      * { box-sizing: border-box; }
      body { margin: 0; font-family: ${E.font}; color: ${E.ink}; background: ${E.bg};
        -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
      button { font-family: inherit; cursor: pointer; }
      .empire-btn {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 10px 18px; border-radius: 999px;
        font-size: 13px; font-weight: 500; letter-spacing: -0.1px;
        border: 1px solid ${E.hair}; background: ${E.card}; color: ${E.ink};
        transition: all 180ms cubic-bezier(.2,.8,.2,1);
      }
      .empire-btn:hover { background: ${E.ink}; color: #fff; border-color: ${E.ink}; }
      .empire-btn-primary { background: ${E.ink}; color: #fff; border-color: ${E.ink}; }
      .empire-btn-primary:hover { background: #000; }
      .empire-fade-in { animation: empireFadeUp 420ms cubic-bezier(.2,.8,.2,1) both; }
    `}</style>
  );
}

Object.assign(window, {
  E, AvatarPortrait, StageBar, LatencyBadge, CommentChip,
  AgentLine, AGENT_COLORS, ProductCard, Eyebrow, EmpireGlobalStyles,
});
