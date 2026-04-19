// empire-mobile.jsx — iOS companion app for the seller.
// Two-screen story: Home (live stream monitor) + Capture (film product).

function EmpireMobileHome() {
  const show = useEmpireShow();
  return (
    <div style={{
      padding: '0 0 16px',
      background: '#F2F2F7',
      fontFamily: '-apple-system, system-ui',
      color: '#000',
    }}>
      {/* Top greeting */}
      <div style={{ padding: '0 20px 12px' }}>
        <div style={{
          fontSize: 13, color: 'rgba(60,60,67,0.6)', letterSpacing: -0.08,
        }}>Good evening, Anya</div>
        <div style={{
          fontSize: 28, fontWeight: 700, letterSpacing: 0.35, marginTop: 2,
          lineHeight: 1.1,
        }}>You're on air.</div>
      </div>

      {/* Live card — hero */}
      <div style={{ padding: '0 16px 14px' }}>
        <div style={{
          borderRadius: 22, overflow: 'hidden',
          background: '#fff',
          boxShadow: '0 1px 2px rgba(0,0,0,0.04), 0 8px 24px rgba(0,0,0,0.04)',
        }}>
          <div style={{ aspectRatio: '1 / 1.05', position: 'relative' }}>
            <AvatarPortrait
              speaking
              caption={show.caption}
              label={`MAYA · ${show.avatarState.toUpperCase()}`}
            />
          </div>
          {/* live metrics strip */}
          <div style={{
            display: 'grid', gridTemplateColumns: 'repeat(3,1fr)',
            padding: '14px 18px', gap: 12,
            borderTop: `1px solid ${E.hair2}`,
          }}>
            {[
              { k: 'Viewers', v: '12.4k' },
              { k: 'Baskets', v: '+47' },
              { k: 'GMV',     v: '$4,812' },
            ].map((m, i) => (
              <div key={m.k} style={{
                borderLeft: i === 0 ? 'none' : `1px solid ${E.hair2}`,
                paddingLeft: i === 0 ? 0 : 12,
              }}>
                <div style={{
                  fontFamily: E.mono, fontSize: 9,
                  color: 'rgba(60,60,67,0.6)', letterSpacing: 0.6,
                }}>{m.k.toUpperCase()}</div>
                <div style={{
                  fontSize: 20, fontWeight: 600, letterSpacing: -0.4, marginTop: 1,
                }}>{m.v}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Stage pill */}
      <div style={{
        margin: '0 16px 14px',
        padding: '12px 14px',
        background: '#fff',
        borderRadius: 16,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <div>
          <div style={{
            fontFamily: E.mono, fontSize: 9,
            color: 'rgba(60,60,67,0.6)', letterSpacing: 0.8,
          }}>STAGE</div>
          <div style={{ fontSize: 15, fontWeight: 600, marginTop: 2 }}>
            {show.phase} · {show.avatarState}
          </div>
        </div>
        <StageBar phase={show.phase} compact />
      </div>

      {/* Now pitching */}
      <div style={{ padding: '0 20px 6px' }}>
        <div style={{
          fontSize: 13, textTransform: 'uppercase', letterSpacing: -0.08,
          color: 'rgba(60,60,67,0.6)', marginBottom: 6, paddingLeft: 16,
        }}>Now pitching</div>
      </div>
      <div style={{
        background: '#fff', borderRadius: 22, margin: '0 16px 14px',
        padding: 14,
        display: 'flex', alignItems: 'center', gap: 12,
      }}>
        <div style={{
          width: 56, height: 56, borderRadius: 10,
          background: `repeating-linear-gradient(135deg,#f4f4f6 0 10px,#ededf0 10px 11px)`,
          flexShrink: 0,
        }} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 15, fontWeight: 600, letterSpacing: -0.2 }}>
            Matte Black Pour-Over
          </div>
          <div style={{ fontSize: 12, color: 'rgba(60,60,67,0.6)', marginTop: 2 }}>
            4 visual details · grounded
          </div>
        </div>
        <div style={{ fontSize: 17, fontWeight: 600 }}>$48</div>
      </div>

      {/* Recent comments */}
      <div style={{ padding: '0 20px 6px' }}>
        <div style={{
          fontSize: 13, textTransform: 'uppercase', letterSpacing: -0.08,
          color: 'rgba(60,60,67,0.6)', marginBottom: 6, paddingLeft: 16,
          display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
        }}>
          <span>Comments</span>
          <span style={{
            fontFamily: E.mono, fontSize: 10, color: 'rgba(60,60,67,0.4)',
            textTransform: 'none', letterSpacing: 0.4,
          }}>Maya is replying →</span>
        </div>
      </div>
      <div style={{ background: '#fff', borderRadius: 22, margin: '0 16px 14px', overflow: 'hidden' }}>
        {show.comments.slice(0, 3).map((c, i, arr) => (
          <div key={c.at} style={{
            display: 'flex', gap: 12, padding: '12px 14px',
            borderBottom: i === arr.length - 1 ? 'none' : `1px solid ${E.hair2}`,
            alignItems: 'flex-start',
          }}>
            <div style={{
              width: 32, height: 32, borderRadius: '50%',
              background: `repeating-linear-gradient(45deg,#eee 0 4px,#e4e4e7 4px 5px)`,
              flexShrink: 0,
            }} />
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                <div style={{ fontSize: 13, fontWeight: 600 }}>{c.handle}</div>
                <div style={{ fontFamily: E.mono, fontSize: 10, color: 'rgba(60,60,67,0.5)' }}>{c.when}</div>
              </div>
              <div style={{ fontSize: 14, marginTop: 1, lineHeight: 1.35 }}>{c.text}</div>
              {c.reply && (
                <div style={{
                  marginTop: 6, fontFamily: E.mono, fontSize: 10,
                  color: E.aura, fontWeight: 600, letterSpacing: 0.6,
                }}>
                  <span style={{
                    display: 'inline-block', width: 6, height: 6, borderRadius: '50%',
                    background: E.aura, marginRight: 6, verticalAlign: 'middle',
                    animation: 'empirePulse 1.2s ease-in-out infinite',
                  }} />
                  REPLYING · WAV2LIP
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Actions */}
      <div style={{ padding: '0 16px 24px', display: 'flex', gap: 10 }}>
        <button style={{
          flex: 1, padding: '14px',
          background: '#000', color: '#fff', border: 'none', borderRadius: 14,
          fontSize: 15, fontWeight: 600, letterSpacing: -0.2, cursor: 'pointer',
        }}>
          + Film new product
        </button>
        <button style={{
          padding: '14px 16px',
          background: '#fff', color: '#000', border: `1px solid ${E.hair}`, borderRadius: 14,
          fontSize: 15, fontWeight: 500, letterSpacing: -0.2, cursor: 'pointer',
        }}>
          End stream
        </button>
      </div>
    </div>
  );
}

function EmpireMobileCapture() {
  const show = useEmpireShow();
  // Show pipeline progress based on t
  const steps = [
    { k: 'Uploaded',         t: 400 },
    { k: 'Deepgram ASR',     t: 1100 },
    { k: 'Claude identifies object', t: 1800 },
    { k: 'ElevenLabs voice', t: 2600 },
    { k: 'Wav2Lip · 5090',   t: 3400 },
    { k: 'Going live',       t: 4200 },
  ];

  const elapsed = show.t;

  return (
    <div style={{
      background: '#000',
      color: '#fff',
      fontFamily: '-apple-system, system-ui',
      minHeight: '100%',
      position: 'relative',
    }}>
      {/* Camera viewfinder */}
      <div style={{
        position: 'absolute', inset: 0,
        background: `repeating-linear-gradient(135deg, #1a1a1c 0 14px, #222224 14px 15px)`,
      }} />
      {/* Product silhouette */}
      <div style={{
        position: 'absolute', left: '50%', top: '50%',
        transform: 'translate(-50%,-50%)',
        width: '55%', aspectRatio: '4/5',
      }}>
        <svg viewBox="0 0 200 250" style={{ width: '100%', height: '100%' }}>
          <defs>
            <radialGradient id="prod" cx="50%" cy="45%" r="45%">
              <stop offset="0%" stopColor="rgba(255,255,255,0.4)" />
              <stop offset="100%" stopColor="rgba(255,255,255,0)" />
            </radialGradient>
          </defs>
          <ellipse cx="100" cy="110" rx="75" ry="85" fill="url(#prod)" />
          <circle cx="100" cy="110" r="55" fill="rgba(255,255,255,0.06)" stroke="rgba(255,255,255,0.2)" />
        </svg>
      </div>

      {/* Capture UI overlay */}
      <div style={{
        position: 'relative', zIndex: 2,
        padding: '90px 20px 0',
        display: 'flex', flexDirection: 'column', height: '100%',
        minHeight: 760,
      }}>
        {/* Top chrome */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{
            padding: '6px 12px 6px 10px',
            borderRadius: 999,
            background: 'rgba(255,255,255,0.12)',
            backdropFilter: 'blur(20px)',
            WebkitBackdropFilter: 'blur(20px)',
            fontFamily: E.mono, fontSize: 10, letterSpacing: 0.8,
            fontWeight: 600, color: '#fff',
            display: 'inline-flex', alignItems: 'center', gap: 6,
          }}>
            <span style={{
              width: 7, height: 7, borderRadius: '50%', background: 'oklch(0.75 0.2 25)',
              animation: 'empirePulse 1.4s ease-in-out infinite',
            }} />
            REC · 00:0{Math.min(9, Math.floor(elapsed/1000))}
          </div>
          <div style={{
            width: 36, height: 36, borderRadius: '50%',
            background: 'rgba(255,255,255,0.12)',
            backdropFilter: 'blur(20px)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 18, color: '#fff',
          }}>✕</div>
        </div>

        {/* Live narration transcription */}
        <div style={{
          margin: '180px 0 auto',
          padding: '14px 16px',
          background: 'rgba(0,0,0,0.5)',
          backdropFilter: 'blur(20px)',
          WebkitBackdropFilter: 'blur(20px)',
          borderRadius: 16,
          fontSize: 15, lineHeight: 1.45, fontWeight: 500,
        }}>
          <div style={{
            fontFamily: E.mono, fontSize: 10, letterSpacing: 0.8, fontWeight: 600,
            color: 'rgba(255,255,255,0.5)', marginBottom: 6,
          }}>DEEPGRAM · LIVE</div>
          "matte black pour-over, walnut handle, spiral ridges inside, about forty-eight dollars…"
        </div>

        {/* Pipeline rail */}
        <div style={{
          marginTop: 16, padding: '14px 16px',
          background: 'rgba(255,255,255,0.08)',
          backdropFilter: 'blur(20px)',
          WebkitBackdropFilter: 'blur(20px)',
          borderRadius: 18,
        }}>
          <div style={{
            fontFamily: E.mono, fontSize: 9, letterSpacing: 0.8,
            color: 'rgba(255,255,255,0.5)', fontWeight: 600, marginBottom: 10,
          }}>BUILDING YOUR AVATAR · {fmtTime(show.t)}</div>
          {steps.map((s, i) => {
            const done = elapsed > s.t;
            const active = !done && (i === 0 || elapsed > steps[i-1].t);
            return (
              <div key={s.k} style={{
                display: 'flex', alignItems: 'center', gap: 10,
                padding: '6px 0', fontSize: 13,
              }}>
                <div style={{
                  width: 18, height: 18, borderRadius: '50%',
                  background: done ? E.live : (active ? 'transparent' : 'rgba(255,255,255,0.08)'),
                  border: active ? `1.5px solid ${E.aura}` : 'none',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 10, color: '#fff',
                }}>
                  {done ? '✓' : active ? (
                    <span style={{
                      width: 8, height: 8, borderRadius: '50%', background: E.aura,
                      animation: 'empirePulse 1s ease-in-out infinite',
                    }} />
                  ) : ''}
                </div>
                <div style={{
                  flex: 1,
                  color: done ? 'rgba(255,255,255,0.6)' : '#fff',
                  fontWeight: active ? 600 : 400,
                }}>{s.k}</div>
                {done && <span style={{ fontFamily: E.mono, fontSize: 10, color: 'rgba(255,255,255,0.4)' }}>done</span>}
              </div>
            );
          })}
        </div>

        {/* Shutter */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          gap: 30, padding: '24px 0',
        }}>
          <div style={{
            width: 44, height: 44, borderRadius: 10,
            background: `repeating-linear-gradient(135deg,#1a1a1c 0 6px,#222224 6px 7px)`,
            border: '1px solid rgba(255,255,255,0.2)',
          }} />
          <div style={{
            width: 72, height: 72, borderRadius: '50%',
            border: '4px solid #fff', padding: 4,
          }}>
            <div style={{
              width: '100%', height: '100%', borderRadius: '50%',
              background: 'oklch(0.75 0.2 25)',
            }} />
          </div>
          <div style={{
            width: 44, height: 44, borderRadius: '50%',
            background: 'rgba(255,255,255,0.12)',
            backdropFilter: 'blur(20px)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 18, color: '#fff',
          }}>↻</div>
        </div>
      </div>
    </div>
  );
}

function EmpireMobile({ screen = 'home' }) {
  return (
    <IOSDevice width={390} height={844} dark={screen === 'capture'}>
      {screen === 'home' && (
        <>
          {/* custom top greeting slot under status bar */}
          <div style={{ height: 54 }} />
          <EmpireMobileHome />
        </>
      )}
      {screen === 'capture' && <EmpireMobileCapture />}
    </IOSDevice>
  );
}

Object.assign(window, { EmpireMobile, EmpireMobileHome, EmpireMobileCapture });
