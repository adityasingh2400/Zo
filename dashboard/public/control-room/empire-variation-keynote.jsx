// empire-variation-keynote.jsx — Variation 3: Keynote
// Extreme minimalism, huge type, avatar to the side, one hero metric

function V3Keynote({ scale = 1 }) {
  const show = useEmpireShow();

  // Compute an apparent latency number that animates as time passes
  const latencyMs = 5600 + Math.round(Math.sin(show.t / 1000) * 300);

  return (
    <div style={{
      width: 1440, height: 900,
      background: '#ffffff',
      fontFamily: E.font,
      display: 'flex', flexDirection: 'column',
      position: 'relative',
    }}>
      {/* minimal nav */}
      <div style={{
        display: 'flex', alignItems: 'center',
        padding: '22px 40px',
        gap: 24,
      }}>
        <div style={{
          fontSize: 15, fontWeight: 600, letterSpacing: -0.3,
        }}>Empire</div>
        <div style={{ flex: 1 }} />
        <div style={{ display: 'flex', gap: 28, fontSize: 13, color: E.ink2 }}>
          <span>Live</span>
          <span>Catalog</span>
          <span>Avatars</span>
          <span>Launch</span>
        </div>
        <div style={{ flex: 1 }} />
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: 8,
          padding: '6px 12px',
          background: 'rgba(0,0,0,0.04)',
          borderRadius: 999,
          fontFamily: E.mono, fontSize: 11, color: E.ink,
        }}>
          <span style={{
            width: 7, height: 7, borderRadius: '50%',
            background: E.live,
            animation: 'empirePulse 1.4s ease-in-out infinite',
          }} />
          streaming · {fmtTime(show.t)}
        </div>
      </div>

      {/* hero split */}
      <div style={{
        flex: 1,
        display: 'grid',
        gridTemplateColumns: '1.2fr 1fr',
        gap: 64,
        padding: '32px 72px 60px',
        alignItems: 'center',
        minHeight: 0,
      }}>
        {/* Left: massive type */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 28 }}>
          <div>
            <div style={{
              fontFamily: E.mono, fontSize: 11,
              color: E.ink3, letterSpacing: 1.4,
              textTransform: 'uppercase', fontWeight: 600,
              marginBottom: 18,
            }}>
              Live · The Pour-Over Drop
            </div>
            <StageBar phase={show.phase} compact />
          </div>

          <div style={{
            fontSize: 72, fontWeight: 600,
            lineHeight: 1.02, letterSpacing: -2.4,
            color: E.ink,
          }}>
            A seller. <br/>
            An avatar. <br/>
            <span style={{ color: E.ink3 }}>Thirty seconds apart.</span>
          </div>

          <div style={{
            fontSize: 19, lineHeight: 1.5, color: E.ink2,
            letterSpacing: -0.2, maxWidth: 520,
          }}>
            Maya is pitching a matte black pour-over that Anya filmed on her phone
            thirty seconds ago, in Manila, in a language Anya barely speaks on camera.
          </div>

          {/* live caption as quote */}
          <div style={{
            padding: '18px 22px',
            borderLeft: `2px solid ${E.ink}`,
            background: 'transparent',
          }}>
            <div style={{
              fontFamily: E.mono, fontSize: 10, color: E.ink3,
              letterSpacing: 1.0, marginBottom: 6,
            }}>
              MAYA · {show.avatarState.toUpperCase()}
            </div>
            <div style={{
              fontSize: 18, lineHeight: 1.45, color: E.ink,
              letterSpacing: -0.2, fontWeight: 500,
              minHeight: 52,
            }} key={show.caption} className="empire-fade-in">
              "{show.caption}"
            </div>
          </div>

          {/* metric strip */}
          <div style={{
            display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)',
            gap: 1,
            background: E.hair,
            border: `1px solid ${E.hair}`,
            borderRadius: 2,
          }}>
            {[
              { k: 'Viewers',   v: '12,408',     sub: '+248/min' },
              { k: 'GMV',       v: '$4,812',     sub: 'this stream' },
              { k: 'Wav2Lip',   v: `${(latencyMs/1000).toFixed(1)}s`, sub: 'p50 render' },
              { k: 'Gemma 4',   v: '284ms',      sub: 'on-device' },
            ].map(s => (
              <div key={s.k} style={{
                background: '#fff', padding: '16px 18px',
              }}>
                <div style={{
                  fontFamily: E.mono, fontSize: 9, color: E.ink3,
                  letterSpacing: 0.8, fontWeight: 600,
                }}>{s.k.toUpperCase()}</div>
                <div style={{
                  fontSize: 28, fontWeight: 600, letterSpacing: -0.8,
                  marginTop: 4, color: E.ink,
                }}>{s.v}</div>
                <div style={{ fontSize: 11, color: E.ink3, marginTop: 2 }}>
                  {s.sub}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right: avatar + comment stack */}
        <div style={{
          display: 'flex', flexDirection: 'column', gap: 14,
          height: '100%', minHeight: 0,
        }}>
          <div style={{ flex: 1, minHeight: 0 }}>
            <AvatarPortrait
              speaking
              caption={null}
              label={`AVATAR · MAYA · ${show.avatarState.toUpperCase()}`}
            />
          </div>

          <div style={{
            display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10,
          }}>
            {show.comments.slice(0, 2).map((c, i) => (
              <div key={c.at} className="empire-fade-in" style={{ animationDelay: `${i * 60}ms` }}>
                <CommentChip {...c} replying={c.reply} />
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* foot — single line agent trace */}
      <div style={{
        padding: '14px 40px',
        borderTop: `1px solid ${E.hair}`,
        display: 'flex', alignItems: 'center', gap: 18,
        fontFamily: E.mono, fontSize: 11,
        overflow: 'hidden', whiteSpace: 'nowrap',
      }}>
        <span style={{ color: E.ink3, letterSpacing: 0.8 }}>BUS</span>
        {show.logs.slice(-3).map((l, i) => (
          <React.Fragment key={l.at}>
            <span style={{ color: AGENT_COLORS[l.agent], fontWeight: 600 }}>
              {l.agent.toUpperCase()}
            </span>
            <span style={{ color: E.ink }}>{l.msg}</span>
            {i < 2 && <span style={{ color: E.ink3 }}>·</span>}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

Object.assign(window, { V3Keynote });
