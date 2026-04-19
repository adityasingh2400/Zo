// empire-variation-console.jsx — Variation 2: Agent Console
// Denser, ops/terminal feel. Agent swarm visual front-and-center.

function V2Console({ scale = 1 }) {
  const show = useEmpireShow();

  return (
    <div style={{
      width: 1440, height: 900,
      background: E.bg,
      fontFamily: E.font,
      display: 'grid',
      gridTemplateColumns: '1fr 1fr',
      gridTemplateRows: '56px 1fr',
    }}>
      {/* ── Top bar ── */}
      <div style={{
        gridColumn: '1 / -1',
        display: 'flex', alignItems: 'center',
        padding: '0 24px', gap: 20,
        borderBottom: `1px solid ${E.hair}`,
        background: E.card,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 20, height: 20, borderRadius: 5,
            background: E.ink, color: '#fff',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 11, fontWeight: 700,
          }}>E</div>
          <div style={{ fontSize: 14, fontWeight: 600, letterSpacing: -0.2 }}>
            Empire · <span style={{ color: E.ink3, fontWeight: 500 }}>Console</span>
          </div>
        </div>
        <div style={{ width: 1, height: 20, background: E.hair }} />
        <div style={{ display: 'flex', gap: 22, fontSize: 13 }}>
          <span style={{ color: E.ink, fontWeight: 500 }}>Live</span>
          <span style={{ color: E.ink3 }}>Catalog</span>
          <span style={{ color: E.ink3 }}>Pipeline</span>
          <span style={{ color: E.ink3 }}>Agents</span>
          <span style={{ color: E.ink3 }}>Shop</span>
        </div>
        <div style={{ flex: 1 }} />
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          fontFamily: E.mono, fontSize: 11, color: E.ink2,
        }}>
          <span style={{
            width: 8, height: 8, borderRadius: '50%',
            background: E.live,
            animation: 'empirePulse 1.4s ease-in-out infinite',
          }} />
          stream_01 · {fmtTime(show.t)}
        </div>
        <StageBar phase={show.phase} compact />
      </div>

      {/* ── LEFT: live stage ── */}
      <div style={{
        borderRight: `1px solid ${E.hair}`,
        padding: 24,
        display: 'flex', flexDirection: 'column', gap: 16,
        minHeight: 0,
      }}>
        <div>
          <Eyebrow>What the audience sees</Eyebrow>
          <div style={{ fontSize: 22, fontWeight: 600, letterSpacing: -0.5 }}>
            TikTok Shop · @maya.makes
          </div>
        </div>

        {/* Phone mockup */}
        <div style={{
          alignSelf: 'center',
          width: 300, aspectRatio: '9 / 19.5',
          background: '#000',
          borderRadius: 38,
          padding: 8,
          boxShadow: E.shadowHi,
          position: 'relative',
        }}>
          <div style={{
            width: '100%', height: '100%',
            borderRadius: 32,
            overflow: 'hidden',
            background: '#f5f5f7',
            position: 'relative',
          }}>
            {/* notch */}
            <div style={{
              position: 'absolute', top: 8, left: '50%', transform: 'translateX(-50%)',
              width: 90, height: 24, borderRadius: 12, background: '#000', zIndex: 10,
            }} />
            <AvatarPortrait
              speaking
              caption={show.caption}
              label={`MAYA · ${show.avatarState.toUpperCase()}`}
            />
            {/* right-rail tiktok-ish controls */}
            <div style={{
              position: 'absolute', right: 10, bottom: 100,
              display: 'flex', flexDirection: 'column', gap: 14,
              alignItems: 'center',
            }}>
              {['♡', '○', '↗', '◈'].map((g, i) => (
                <div key={i} style={{
                  width: 36, height: 36, borderRadius: '50%',
                  background: 'rgba(255,255,255,0.2)',
                  backdropFilter: 'blur(10px)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  color: '#fff', fontSize: 16,
                }}>{g}</div>
              ))}
            </div>
            {/* product sticker */}
            <div style={{
              position: 'absolute', left: 10, bottom: 60,
              display: 'flex', alignItems: 'center', gap: 8,
              padding: '6px 10px 6px 6px',
              background: 'rgba(255,255,255,0.95)',
              borderRadius: 22,
              fontSize: 11, fontWeight: 600,
            }}>
              <div style={{
                width: 30, height: 30, borderRadius: 6,
                background: `repeating-linear-gradient(135deg,#f4f4f6 0 6px,#ededf0 6px 7px)`,
              }} />
              <div style={{ lineHeight: 1.1 }}>
                Matte Pour-Over<br/>
                <span style={{ color: E.ink2, fontWeight: 500 }}>$48</span>
              </div>
            </div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 8 }}>
          {[
            { k: 'Viewers', v: '12,408' },
            { k: 'Baskets', v: '+47' },
            { k: 'GMV', v: '$4,812' },
          ].map(s => (
            <div key={s.k} style={{
              padding: 12,
              border: `1px solid ${E.hair}`,
              borderRadius: 10,
              background: E.card,
            }}>
              <div style={{
                fontFamily: E.mono, fontSize: 10, color: E.ink3, letterSpacing: 0.6,
              }}>{s.k.toUpperCase()}</div>
              <div style={{ fontSize: 20, fontWeight: 600, letterSpacing: -0.5, marginTop: 2 }}>
                {s.v}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── RIGHT: ops console ── */}
      <div style={{
        padding: 24, display: 'flex', flexDirection: 'column', gap: 16,
        minHeight: 0,
      }}>
        {/* Agent swarm */}
        <div>
          <Eyebrow right={
            <span style={{ fontFamily: E.mono, fontSize: 10, color: E.ink3 }}>
              ws bus · 4 agents
            </span>
          }>Agent swarm</Eyebrow>
          <div style={{
            padding: 16,
            background: E.card,
            border: `1px solid ${E.hair}`,
            borderRadius: 14,
            display: 'grid',
            gridTemplateColumns: 'repeat(4, 1fr)',
            gap: 12,
          }}>
            {[
              { name: 'Eyes',     role: 'perception',     stack: 'deepgram · claude · gemma4', key: 'eyes' },
              { name: 'Seller',   role: 'spoken brain',   stack: 'haiku · eleven · wav2lip',   key: 'seller' },
              { name: 'Director', role: 'cinematographer',stack: 'state machine · tier0/1',    key: 'director' },
              { name: 'Hands',    role: 'actuation',      stack: 'shop · ugc · ship',          key: 'hands' },
            ].map(a => {
              const active = show.logs.some(l => l.agent === a.key && (show.t - l.at) < 1500);
              return (
                <div key={a.key} style={{
                  padding: 12,
                  background: active ? 'rgba(0,0,0,0.03)' : 'transparent',
                  border: `1px solid ${active ? AGENT_COLORS[a.key] : E.hair}`,
                  borderRadius: 10,
                  transition: 'all 240ms',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                    <span style={{
                      width: 8, height: 8, borderRadius: '50%',
                      background: AGENT_COLORS[a.key],
                      opacity: active ? 1 : 0.35,
                      animation: active ? 'empirePulse 1.2s ease-in-out infinite' : 'none',
                    }} />
                    <div style={{ fontSize: 13, fontWeight: 600, letterSpacing: -0.2 }}>
                      {a.name}
                    </div>
                  </div>
                  <div style={{ fontSize: 11, color: E.ink2 }}>{a.role}</div>
                  <div style={{
                    marginTop: 8, fontFamily: E.mono, fontSize: 10,
                    color: E.ink3, letterSpacing: 0.3,
                  }}>{a.stack}</div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Pipeline */}
        <div>
          <Eyebrow>Intake pipeline</Eyebrow>
          <div style={{
            padding: 14,
            background: E.card,
            border: `1px solid ${E.hair}`,
            borderRadius: 14,
            display: 'grid',
            gridTemplateColumns: 'repeat(5, 1fr)',
            gap: 8,
            alignItems: 'center',
          }}>
            {['Clip', 'Deepgram', 'Claude', 'Eleven', 'Wav2Lip'].map((s, i) => (
              <div key={s} style={{ textAlign: 'center' }}>
                <div style={{
                  width: 36, height: 36, margin: '0 auto 6px',
                  borderRadius: 10,
                  background: i < 4 ? E.ink : 'rgba(0,0,0,0.04)',
                  border: i === 4 ? `1px dashed ${E.ink2}` : 'none',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  color: i < 4 ? '#fff' : E.ink2,
                  fontSize: 12, fontWeight: 600,
                }}>
                  {i < 4 ? '✓' : (
                    <span style={{
                      width: 10, height: 10, borderRadius: '50%',
                      background: E.aura,
                      animation: 'empirePulse 1s ease-in-out infinite',
                    }} />
                  )}
                </div>
                <div style={{ fontSize: 11, fontWeight: 600, color: E.ink }}>{s}</div>
                <div style={{ fontFamily: E.mono, fontSize: 10, color: E.ink3 }}>
                  {['0.4s','0.5s','1.2s','1.1s','5.6s'][i]}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Agent log */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          <Eyebrow right={
            <div style={{ display: 'flex', gap: 10 }}>
              {Object.entries(AGENT_COLORS).map(([k, v]) => (
                <span key={k} style={{
                  display: 'inline-flex', alignItems: 'center', gap: 4,
                  fontFamily: E.mono, fontSize: 9, color: E.ink3, letterSpacing: 0.5,
                }}>
                  <span style={{ width: 6, height: 6, borderRadius: '50%', background: v }} />
                  {k}
                </span>
              ))}
            </div>
          }>WebSocket bus</Eyebrow>
          <div style={{
            flex: 1,
            background: E.card,
            border: `1px solid ${E.hair}`,
            borderRadius: 14,
            padding: '4px 16px',
            overflow: 'hidden',
            position: 'relative',
          }}>
            <div style={{
              position: 'absolute', inset: '4px 16px',
              maskImage: 'linear-gradient(to bottom, transparent, black 15%, black 100%)',
              WebkitMaskImage: 'linear-gradient(to bottom, transparent, black 15%, black 100%)',
              display: 'flex', flexDirection: 'column-reverse',
            }}>
              {[...show.logs].reverse().map((l) => (
                <AgentLine key={l.at} agent={l.agent} message={l.msg} t={fmtTime(l.at)} />
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { V2Console });
