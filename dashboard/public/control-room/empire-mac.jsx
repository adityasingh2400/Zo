// empire-mac.jsx — Mac control-room. Full desktop window, way more controls.
// Uses: MacWindow, MacSidebarItem, MacSidebarHeader from macos-window.jsx

function EmpireMac() {
  const show = useEmpireShow();
  const [selectedAvatar, setSelectedAvatar] = React.useState('maya');
  const [streamEnabled, setStreamEnabled] = React.useState({
    tiktok: true, shopify: true, etsy: false, instagram: false,
  });
  const [voice, setVoice] = React.useState('warm');
  const [lang, setLang] = React.useState('EN');
  const [quality, setQuality] = React.useState('wav2lip');

  const sidebar = (
    <>
      <MacSidebarHeader title="STUDIO" />
      <MacSidebarItem label="Live Control Room" selected />
      <MacSidebarItem label="Intake Queue" />
      <MacSidebarItem label="Catalog · 247" />
      <MacSidebarItem label="Avatars · 6" />

      <MacSidebarHeader title="DISTRIBUTION" />
      <MacSidebarItem label="TikTok Shop" />
      <MacSidebarItem label="Shopify Live" />
      <MacSidebarItem label="Etsy" />
      <MacSidebarItem label="Global Launch" />
      <MacSidebarItem label="UGC Mode" />

      <MacSidebarHeader title="AGENTS" />
      <MacSidebarItem label="Eyes · perception" />
      <MacSidebarItem label="Seller · spoken" />
      <MacSidebarItem label="Director · stage" />
      <MacSidebarItem label="Hands · actuation" />

      <MacSidebarHeader title="INFRA" />
      <MacSidebarItem label="Pod · 5090" />
      <MacSidebarItem label="Voice · Eleven" />
      <MacSidebarItem label="On-device · Gemma 4" />
    </>
  );

  return (
    <MacWindow width={1440} height={900} title="Live Control Room · The Pour-Over Drop" sidebar={sidebar}>
      <div style={{
        padding: '14px 20px 20px',
        display: 'grid',
        gridTemplateColumns: '1.35fr 1fr',
        gridTemplateRows: 'auto auto 1fr',
        gap: 16,
        height: 'calc(100% - 8px)',
        fontFamily: E.font,
        color: E.ink,
      }}>
        {/* ── Hero strip: metrics across top ── */}
        <div style={{
          gridColumn: '1 / -1',
          display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)',
          gap: 1, background: E.hair,
          border: `1px solid ${E.hair}`, borderRadius: 12,
          overflow: 'hidden',
        }}>
          {[
            { k: 'PHASE',     v: show.phase,       s: 'stage machine' },
            { k: 'VIEWERS',   v: '12,408',         s: '+248/min' },
            { k: 'GMV',       v: '$4,812',         s: 'this stream' },
            { k: 'BASKETS',   v: '+47',            s: 'last 5min' },
            { k: 'WAV2LIP',   v: '5.6s',           s: 'warm p50' },
            { k: 'GEMMA 4',   v: '284ms',          s: 'on-device' },
          ].map(m => (
            <div key={m.k} style={{ background: '#fff', padding: '12px 16px' }}>
              <div style={{
                fontFamily: E.mono, fontSize: 9, letterSpacing: 0.8,
                color: E.ink3, fontWeight: 600,
              }}>{m.k}</div>
              <div style={{ fontSize: 22, fontWeight: 600, letterSpacing: -0.6, marginTop: 2 }}>
                {m.v}
              </div>
              <div style={{ fontSize: 10, color: E.ink3, marginTop: 1 }}>{m.s}</div>
            </div>
          ))}
        </div>

        {/* ── Stage bar + transport ── */}
        <div style={{
          gridColumn: '1 / -1',
          display: 'flex', alignItems: 'center', gap: 14,
          padding: '10px 14px',
          background: '#fff',
          border: `1px solid ${E.hair}`,
          borderRadius: 12,
        }}>
          <StageBar phase={show.phase} compact />
          <div style={{ flex: 1 }} />
          {/* transport cluster */}
          <div style={{
            display: 'inline-flex', alignItems: 'center', gap: 4,
            padding: 4, background: '#f5f5f7', borderRadius: 10,
          }}>
            {['⏮', '⏸', '⏭'].map((g, i) => (
              <button key={i} style={{
                width: 30, height: 26, borderRadius: 7, border: 'none',
                background: i === 1 ? '#fff' : 'transparent',
                boxShadow: i === 1 ? '0 1px 2px rgba(0,0,0,0.08)' : 'none',
                fontSize: 11, color: E.ink, cursor: 'pointer',
              }}>{g}</button>
            ))}
          </div>
          <div style={{
            fontFamily: E.mono, fontSize: 11, color: E.ink2,
          }}>{fmtTime(show.t)} / 00:20</div>
          <div style={{ width: 1, height: 18, background: E.hair }} />
          {/* force-phase cluster */}
          <div style={{ fontFamily: E.mono, fontSize: 10, color: E.ink3, letterSpacing: 0.6 }}>
            FORCE
          </div>
          {['INTRO', 'BRIDGE', 'PITCH', 'LIVE'].map(p => (
            <button key={p} style={{
              padding: '5px 9px',
              border: `1px solid ${E.hair}`,
              borderRadius: 7, background: '#fff',
              fontFamily: E.mono, fontSize: 10, letterSpacing: 0.6,
              color: E.ink, cursor: 'pointer', fontWeight: 600,
            }}>{p}</button>
          ))}
          <div style={{ width: 1, height: 18, background: E.hair }} />
          <button className="empire-btn empire-btn-primary" style={{ padding: '7px 14px', fontSize: 12 }}>
            <span style={{
              width: 7, height: 7, borderRadius: '50%', background: E.live,
              animation: 'empirePulse 1.4s ease-in-out infinite',
            }} />
            On Air
          </button>
        </div>

        {/* ── LEFT: avatar library rail + square live stage ── */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '128px 1fr',
          gap: 12, minHeight: 0,
        }}>
          {/* Avatar library — vertical rail */}
          <div style={{
            padding: 10,
            background: '#fff', border: `1px solid ${E.hair}`,
            borderRadius: 12,
            display: 'flex', flexDirection: 'column', gap: 8,
            minHeight: 0, overflow: 'hidden',
          }}>
            <div style={{
              fontFamily: E.mono, fontSize: 9, letterSpacing: 1.0,
              textTransform: 'uppercase', color: E.ink3, fontWeight: 600,
              padding: '2px 2px 4px',
            }}>Avatars</div>
            <div style={{
              display: 'flex', flexDirection: 'column', gap: 7, flex: 1,
              overflow: 'hidden',
            }}>
              {[
                { k: 'maya',    name: 'Maya',    tag: 'en·es·tl' },
                { k: 'leo',     name: 'Leo',     tag: 'en·es' },
                { k: 'mei',     name: 'Mei',     tag: 'zh·en' },
                { k: 'jasper',  name: 'Jasper',  tag: 'en' },
              ].map(a => {
                const sel = selectedAvatar === a.k;
                return (
                  <button key={a.k} onClick={() => setSelectedAvatar(a.k)} style={{
                    padding: 6,
                    border: `1px solid ${sel ? E.ink : E.hair}`,
                    borderRadius: 10,
                    background: sel ? 'rgba(0,0,0,0.03)' : '#fff',
                    cursor: 'pointer', textAlign: 'left',
                    display: 'flex', flexDirection: 'column', gap: 4,
                  }}>
                    <div style={{
                      width: '100%', aspectRatio: '1', borderRadius: 7,
                      background: `repeating-linear-gradient(135deg,#f4f4f6 0 8px,#ededf0 8px 9px)`,
                      position: 'relative',
                    }}>
                      {sel && (
                        <div style={{
                          position: 'absolute', top: 4, right: 4,
                          width: 14, height: 14, borderRadius: '50%',
                          background: E.ink, color: '#fff',
                          display: 'flex', alignItems: 'center', justifyContent: 'center',
                          fontSize: 8,
                        }}>●</div>
                      )}
                    </div>
                    <div style={{
                      display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
                      padding: '0 2px',
                    }}>
                      <div style={{ fontSize: 11, fontWeight: 600, color: E.ink }}>{a.name}</div>
                      <div style={{ fontSize: 9, color: E.ink3, fontFamily: E.mono }}>{a.tag}</div>
                    </div>
                  </button>
                );
              })}
            </div>
            <button style={{
              padding: '7px 8px',
              border: `1px dashed ${E.hair}`,
              borderRadius: 8, background: 'transparent',
              fontSize: 11, color: E.ink2, cursor: 'pointer',
              fontFamily: E.font,
            }}>+ New avatar</button>
          </div>

          {/* Square live stage */}
          <div style={{
            minHeight: 0, display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <div style={{
              aspectRatio: '1 / 1',
              height: '100%',
              maxWidth: '100%',
            }}>
              <AvatarPortrait
                speaking
                caption={show.caption}
                label={`AVATAR · MAYA · ${show.avatarState.toUpperCase()}`}
              />
            </div>
          </div>
        </div>

        {/* ── RIGHT: ops stack ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14, minHeight: 0 }}>
          {/* Now pitching */}
          <div style={{
            padding: 14,
            background: '#fff', border: `1px solid ${E.hair}`, borderRadius: 12,
            display: 'grid', gridTemplateColumns: '88px 1fr', gap: 14, alignItems: 'center',
          }}>
            <div style={{
              width: 88, height: 88, borderRadius: 10,
              background: `repeating-linear-gradient(135deg,#f4f4f6 0 14px,#ededf0 14px 15px)`,
            }} />
            <div>
              <Eyebrow>Now pitching</Eyebrow>
              <div style={{ fontSize: 17, fontWeight: 600, letterSpacing: -0.3 }}>
                Matte Black Pour-Over
              </div>
              <div style={{ display: 'flex', gap: 6, marginTop: 6, flexWrap: 'wrap' }}>
                {['matte', 'walnut handle', 'spiral ridges', '60mm'].map(d => (
                  <span key={d} style={{
                    padding: '3px 7px', border: `1px solid ${E.hair}`,
                    borderRadius: 999, fontSize: 10, color: E.ink,
                  }}>{d}</span>
                ))}
              </div>
              <div style={{
                marginTop: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
              }}>
                <div style={{ fontFamily: E.mono, fontSize: 11, color: E.ink3 }}>
                  ANAGAMA-01 · seller: anya.mnl
                </div>
                <div style={{ fontSize: 14, fontWeight: 600 }}>$48</div>
              </div>
            </div>
          </div>

          {/* Pipeline w/ controls */}
          <div style={{
            padding: 14,
            background: '#fff', border: `1px solid ${E.hair}`, borderRadius: 12,
          }}>
            <Eyebrow right={<button style={{
              fontFamily: E.mono, fontSize: 10, letterSpacing: 0.6,
              border: `1px solid ${E.hair}`, background: '#fff',
              padding: '3px 7px', borderRadius: 6, cursor: 'pointer',
            }}>RE-INTAKE</button>}>Intake pipeline</Eyebrow>
            <div style={{
              display: 'grid', gridTemplateColumns: 'repeat(5,1fr)',
              gap: 6, alignItems: 'start',
            }}>
              {[
                { s: 'Clip',     t: '0.4s', d: 'phone' },
                { s: 'Deepgram', t: '0.5s', d: 'asr' },
                { s: 'Claude',   t: '1.2s', d: 'pitch' },
                { s: 'Eleven',   t: '1.1s', d: 'tts' },
                { s: 'Wav2Lip',  t: '5.6s', d: 'lipsync' },
              ].map((st, i) => (
                <div key={st.s} style={{
                  padding: 8, borderRadius: 8,
                  background: i < 4 ? 'rgba(0,0,0,0.03)' : 'transparent',
                  border: i === 4 ? `1px dashed ${E.ink2}` : `1px solid ${E.hair}`,
                  textAlign: 'center',
                }}>
                  <div style={{ fontSize: 11, fontWeight: 600 }}>{st.s}</div>
                  <div style={{ fontFamily: E.mono, fontSize: 10, color: E.ink3 }}>{st.t}</div>
                  <div style={{ fontSize: 9, color: E.ink3, marginTop: 2 }}>{st.d}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Distribution toggles */}
          <div style={{
            padding: 14,
            background: '#fff', border: `1px solid ${E.hair}`, borderRadius: 12,
          }}>
            <Eyebrow>Distribution</Eyebrow>
            {[
              { k: 'tiktok',    label: 'TikTok Shop',   sub: '@maya.makes · 12,408 watching' },
              { k: 'shopify',   label: 'Shopify Live',  sub: 'anagamastudio.myshopify.com' },
              { k: 'etsy',      label: 'Etsy',          sub: 'mirror to UGC only (no live api)' },
              { k: 'instagram', label: 'Instagram Live', sub: 'UGC fanout · 9:16' },
            ].map((d, i) => (
              <div key={d.k} style={{
                display: 'flex', alignItems: 'center', gap: 12,
                padding: '8px 0',
                borderTop: i === 0 ? 'none' : `1px solid ${E.hair2}`,
              }}>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 13, fontWeight: 500, color: E.ink }}>{d.label}</div>
                  <div style={{ fontSize: 11, color: E.ink3, fontFamily: E.mono }}>{d.sub}</div>
                </div>
                <button
                  onClick={() => setStreamEnabled(s => ({ ...s, [d.k]: !s[d.k] }))}
                  style={{
                    width: 36, height: 20, borderRadius: 999, border: 'none', padding: 2,
                    background: streamEnabled[d.k] ? E.ink : '#d4d4d7',
                    cursor: 'pointer', display: 'flex',
                    justifyContent: streamEnabled[d.k] ? 'flex-end' : 'flex-start',
                    transition: 'all 220ms',
                  }}>
                  <span style={{
                    width: 16, height: 16, borderRadius: '50%',
                    background: '#fff', boxShadow: '0 1px 2px rgba(0,0,0,0.2)',
                  }} />
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* ── Bottom row: comments + agent log spanning both cols ── */}
        <div style={{
          gridColumn: '1 / -1',
          display: 'grid', gridTemplateColumns: '1fr 1.5fr', gap: 16,
          minHeight: 0, height: 240,
        }}>
          {/* Comments */}
          <div style={{
            background: '#fff', border: `1px solid ${E.hair}`, borderRadius: 12,
            padding: 14, display: 'flex', flexDirection: 'column', minHeight: 0,
          }}>
            <Eyebrow right={
              <span style={{ fontFamily: E.mono, fontSize: 10, color: E.ink3 }}>
                {show.comments.length} · gemma 4 routing
              </span>
            }>Viewer comments</Eyebrow>
            <div style={{
              flex: 1, overflow: 'hidden',
              maskImage: 'linear-gradient(to bottom,black 85%,transparent)',
              WebkitMaskImage: 'linear-gradient(to bottom,black 85%,transparent)',
              display: 'flex', flexDirection: 'column', gap: 8,
            }}>
              {show.comments.slice(0, 3).map(c => (
                <CommentChip key={c.at} {...c} replying={c.reply} />
              ))}
            </div>
          </div>

          {/* Agent log */}
          <div style={{
            background: '#fafafc', border: `1px solid ${E.hair}`, borderRadius: 12,
            padding: '12px 14px', display: 'flex', flexDirection: 'column', minHeight: 0,
          }}>
            <Eyebrow right={
              <div style={{ display: 'flex', gap: 10 }}>
                {Object.entries(AGENT_COLORS).map(([k, v]) => (
                  <span key={k} style={{
                    display: 'inline-flex', alignItems: 'center', gap: 4,
                    fontFamily: E.mono, fontSize: 9, color: E.ink2, letterSpacing: 0.5,
                  }}>
                    <span style={{ width: 6, height: 6, borderRadius: '50%', background: v }} />
                    {k}
                  </span>
                ))}
              </div>
            }>WebSocket bus · ws://empire.local</Eyebrow>
            <div style={{
              flex: 1, overflow: 'hidden', position: 'relative',
            }}>
              <div style={{
                position: 'absolute', inset: 0,
                maskImage: 'linear-gradient(to bottom, transparent, black 20%, black 100%)',
                WebkitMaskImage: 'linear-gradient(to bottom, transparent, black 20%, black 100%)',
                display: 'flex', flexDirection: 'column-reverse',
              }}>
                {[...show.logs].reverse().map(l => (
                  <AgentLine key={l.at} agent={l.agent} message={l.msg} t={fmtTime(l.at)} />
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </MacWindow>
  );
}

Object.assign(window, { EmpireMac });
