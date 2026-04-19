// empire-variation-studio.jsx — Variation 1: Cinema Studio
// Big avatar hero, stage bar up top, comments float right, agent log bottom

function V1Studio({ scale = 1 }) {
  const show = useEmpireShow();

  return (
    <div style={{
      width: 1440, height: 900,
      background: E.bg,
      display: 'grid',
      gridTemplateColumns: '260px 1fr 340px',
      gridTemplateRows: '64px 1fr 200px',
      fontFamily: E.font,
    }}>
      {/* ── Top bar ── */}
      <div style={{
        gridColumn: '1 / -1',
        display: 'flex', alignItems: 'center',
        padding: '0 28px',
        borderBottom: `1px solid ${E.hair}`,
        background: 'rgba(255,255,255,0.8)',
        backdropFilter: 'blur(20px)',
        gap: 24,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 22, height: 22, borderRadius: 6,
            background: E.ink,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: '#fff', fontSize: 11, fontWeight: 700, letterSpacing: -0.4,
          }}>E</div>
          <div style={{ fontSize: 15, fontWeight: 600, letterSpacing: -0.3 }}>
            Empire
          </div>
          <div style={{
            padding: '2px 8px', borderRadius: 6,
            background: E.hair2, fontSize: 10, color: E.ink2,
            fontFamily: E.mono, letterSpacing: 0.4, fontWeight: 600,
          }}>STUDIO</div>
        </div>
        <div style={{ flex: 1 }} />
        <StageBar phase={show.phase} compact />
        <div style={{ flex: 1 }} />
        <LatencyBadge ms={5600} label="wav2lip p50" />
        <LatencyBadge ms={284} label="gemma" />
        <div style={{
          width: 30, height: 30, borderRadius: '50%',
          background: `repeating-linear-gradient(45deg, #eee 0 4px, #e4e4e7 4px 5px)`,
          border: `1px solid ${E.hair}`,
        }} />
      </div>

      {/* ── Left sidebar ── */}
      <aside style={{
        borderRight: `1px solid ${E.hair}`,
        padding: '22px 18px',
        display: 'flex', flexDirection: 'column', gap: 4,
      }}>
        {[
          { icon: '●', label: 'Live Mode', active: true },
          { icon: '○', label: 'Catalog' },
          { icon: '○', label: 'Intake Queue' },
          { icon: '○', label: 'Avatars' },
          { icon: '○', label: 'Global Launch' },
          { icon: '○', label: 'UGC Mode' },
          { icon: '○', label: 'Analytics' },
        ].map(item => (
          <div key={item.label} style={{
            display: 'flex', alignItems: 'center', gap: 10,
            padding: '8px 10px',
            borderRadius: 8,
            background: item.active ? 'rgba(0,0,0,0.04)' : 'transparent',
            color: item.active ? E.ink : E.ink2,
            fontSize: 13, fontWeight: item.active ? 600 : 400,
            letterSpacing: -0.1,
          }}>
            <span style={{
              fontSize: 8,
              color: item.active ? E.live : E.ink3,
            }}>●</span>
            {item.label}
          </div>
        ))}

        <div style={{
          marginTop: 'auto', padding: '14px 10px',
          borderTop: `1px solid ${E.hair}`,
        }}>
          <div style={{
            fontFamily: E.mono, fontSize: 10, color: E.ink3,
            letterSpacing: 0.8, marginBottom: 8,
          }}>POD · 5090</div>
          <div style={{
            display: 'flex', justifyContent: 'space-between',
            fontSize: 12, color: E.ink,
          }}>
            <span>149.36.0.168</span>
            <span style={{ color: E.live, fontFamily: E.mono, fontSize: 11 }}>● warm</span>
          </div>
          <div style={{ marginTop: 4, fontSize: 11, color: E.ink3 }}>
            Wav2Lip :8010 · LatentSync :8766
          </div>
        </div>
      </aside>

      {/* ── Stage (main) ── */}
      <main style={{ padding: 28, display: 'flex', flexDirection: 'column', gap: 20, minHeight: 0 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between' }}>
          <div>
            <Eyebrow>Stream · New York · 21:47</Eyebrow>
            <div style={{ fontSize: 32, fontWeight: 600, letterSpacing: -0.8, color: E.ink }}>
              The Pour-Over Drop
            </div>
          </div>
          <div style={{
            display: 'flex', alignItems: 'center', gap: 12,
          }}>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontFamily: E.mono, fontSize: 10, color: E.ink3, letterSpacing: 0.6 }}>VIEWERS</div>
              <div style={{ fontSize: 24, fontWeight: 600, letterSpacing: -0.6, color: E.ink }}>
                12,408
              </div>
            </div>
            <div style={{ width: 1, height: 32, background: E.hair }} />
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontFamily: E.mono, fontSize: 10, color: E.ink3, letterSpacing: 0.6 }}>GMV</div>
              <div style={{ fontSize: 24, fontWeight: 600, letterSpacing: -0.6, color: E.ink }}>
                $4,812
              </div>
            </div>
          </div>
        </div>

        <div style={{
          flex: 1, display: 'grid', gridTemplateColumns: '1fr 280px', gap: 18, minHeight: 0,
        }}>
          <AvatarPortrait
            speaking
            caption={show.caption}
            label={`AVATAR · MAYA · ${show.avatarState.toUpperCase()}`}
          />
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <ProductCard />
            <div style={{
              padding: 14,
              background: E.card,
              border: `1px solid ${E.hair}`,
              borderRadius: 14,
            }}>
              <Eyebrow>Now playing</Eyebrow>
              <div style={{ fontFamily: E.mono, fontSize: 11, color: E.ink, lineHeight: 1.7 }}>
                pitch_001.mp4<br/>
                <span style={{ color: E.ink3 }}>state:</span> {show.avatarState}<br/>
                <span style={{ color: E.ink3 }}>phase:</span> {show.phase}<br/>
                <span style={{ color: E.ink3 }}>elapsed:</span> {fmtTime(show.t)}
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* ── Right: comments rail ── */}
      <aside style={{
        borderLeft: `1px solid ${E.hair}`,
        padding: '28px 20px',
        display: 'flex', flexDirection: 'column', gap: 12,
        overflow: 'hidden',
      }}>
        <Eyebrow right={
          <span style={{ fontFamily: E.mono, fontSize: 10, color: E.ink3 }}>
            {show.comments.length} active
          </span>
        }>Viewer comments</Eyebrow>

        <div style={{
          display: 'flex', flexDirection: 'column', gap: 10,
          overflow: 'hidden', flex: 1,
        }}>
          {show.comments.map((c, i) => (
            <div key={c.at} className="empire-fade-in" style={{ animationDelay: `${i * 40}ms` }}>
              <CommentChip {...c} replying={c.reply} />
            </div>
          ))}
        </div>
      </aside>

      {/* ── Bottom: agent log ── */}
      <section style={{
        gridColumn: '1 / -1',
        borderTop: `1px solid ${E.hair}`,
        background: '#fafafc',
        padding: '14px 28px',
        display: 'flex', flexDirection: 'column',
        minHeight: 0,
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
          <Eyebrow style={{ margin: 0 }}>Agent bus · ws://empire.local</Eyebrow>
          <div style={{ display: 'flex', gap: 14 }}>
            {Object.entries(AGENT_COLORS).map(([k, v]) => (
              <div key={k} style={{
                display: 'flex', alignItems: 'center', gap: 6,
                fontFamily: E.mono, fontSize: 10, color: E.ink2, letterSpacing: 0.6,
              }}>
                <span style={{ width: 7, height: 7, borderRadius: '50%', background: v }} />
                {k.toUpperCase()}
              </div>
            ))}
          </div>
        </div>
        <div style={{
          flex: 1, overflow: 'hidden', position: 'relative',
        }}>
          <div style={{
            position: 'absolute', inset: 0,
            maskImage: 'linear-gradient(to bottom, transparent, black 20%, black 100%)',
            WebkitMaskImage: 'linear-gradient(to bottom, transparent, black 20%, black 100%)',
            display: 'flex', flexDirection: 'column-reverse',
          }}>
            {[...show.logs].reverse().map((l, i) => (
              <AgentLine key={l.at} agent={l.agent} message={l.msg} t={fmtTime(l.at)} />
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}

Object.assign(window, { V1Studio });
