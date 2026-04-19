// empire-state.jsx — stage machine, scripts, hooks

// The live show script — auto-plays on load, cycles naturally
const LIVE_SCRIPT = [
  { phase: 'INTRO', state: 'idle', ms: 3200,
    caption: "Hey everyone — welcome back to the drop. New pieces just landed." },
  { phase: 'BRIDGE', state: 'explaining', ms: 2600,
    caption: "Give me one second — I'm pulling up something I'm genuinely excited about." },
  { phase: 'PITCH', state: 'pitching', ms: 4200,
    caption: "This is the matte black pour-over. Spiral ridges inside, walnut handle — the ceramic's hand-thrown in Gifu." },
  { phase: 'PITCH', state: 'excited', ms: 3400,
    caption: "60-millimeter cone, fits a standard V60 filter. Forty-eight dollars. Tap the basket." },
  { phase: 'LIVE', state: 'idle', ms: 3000,
    caption: "I see a bunch of you asking about shipping — let me check that real quick." },
  { phase: 'LIVE', state: 'reaching', ms: 3400,
    caption: "@milktea.lvr — yes, we ship to the Philippines. Three to five days, tracked." },
];

const LIVE_COMMENTS = [
  { at: 6200,  handle: '@milktea.lvr',   text: 'does it ship to PH?', reply: true,  when: 'now' },
  { at: 9800,  handle: '@bean.obsessed', text: 'is the ceramic hand-thrown?? 😍',  when: '2s' },
  { at: 13400, handle: '@ritual.cafe',   text: 'bundle w/ filters?',   when: '4s' },
  { at: 17000, handle: '@anya.mnl',      text: 'love the walnut handle', when: '5s' },
  { at: 20600, handle: '@morning.pour',  text: 'size vs. V60-02?',      when: '8s' },
];

const AGENT_LOG = [
  { at: 200,   agent: 'eyes',     msg: 'phone_clip.mp4 received · 11.2s · 1080p' },
  { at: 520,   agent: 'eyes',     msg: 'deepgram.transcribe → "matte black pour over… walnut handle… spiral…"' },
  { at: 1100,  agent: 'eyes',     msg: 'claude.vision → object=pour_over_dripper · color=matte_black · material=ceramic' },
  { at: 1880,  agent: 'seller',   msg: 'claude.haiku → pitch drafted · grounded in 4 visual details' },
  { at: 2700,  agent: 'seller',   msg: 'elevenlabs.flash_v2_5 → 9.4s audio · 1.1s render' },
  { at: 3400,  agent: 'director', msg: 'stage → INTRO (generic idle)' },
  { at: 5800,  agent: 'director', msg: 'stage → BRIDGE (generic filler · hides render)' },
  { at: 6400,  agent: 'seller',   msg: 'wav2lip.5090 → pitch_001.mp4 · warm p50 5.6s' },
  { at: 8200,  agent: 'director', msg: 'stage → PITCH · crossfade 320ms · no gap' },
  { at: 9900,  agent: 'eyes',     msg: 'gemma4.e4b → @milktea.lvr · intent=shipping · 284ms' },
  { at: 10400, agent: 'seller',   msg: 'reactive_reply queued · tier 1' },
  { at: 12100, agent: 'director', msg: 'bridge.tier0 → "let me check that real quick"' },
  { at: 13200, agent: 'eyes',     msg: 'gemma4.e4b → @bean.obsessed · intent=product_detail · 312ms' },
  { at: 15600, agent: 'seller',   msg: 'wav2lip.5090 → reply_001.mp4 · 5.4s' },
  { at: 17200, agent: 'director', msg: 'stage → LIVE · reactive crossfade' },
  { at: 19800, agent: 'hands',    msg: 'tiktok_shop.mock → basket_impression +47' },
];

// Hook: auto-advancing show. Returns { phase, state, caption, t, comments, logs, playing, toggle, reset }
function useEmpireShow() {
  const [t, setT] = React.useState(0); // ms
  const [playing, setPlaying] = React.useState(true);
  const rafRef = React.useRef(null);
  const lastRef = React.useRef(null);

  const totalDuration = LIVE_SCRIPT.reduce((s, x) => s + x.ms, 0);

  React.useEffect(() => {
    if (!playing) return;
    const tick = (now) => {
      if (lastRef.current == null) lastRef.current = now;
      const dt = now - lastRef.current;
      lastRef.current = now;
      setT(prev => (prev + dt) % totalDuration);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(rafRef.current);
      lastRef.current = null;
    };
  }, [playing, totalDuration]);

  // Find current script step
  let acc = 0;
  let step = LIVE_SCRIPT[0];
  for (const s of LIVE_SCRIPT) {
    if (t < acc + s.ms) { step = s; break; }
    acc += s.ms;
  }

  const comments = LIVE_COMMENTS.filter(c => c.at <= t).slice(-5).reverse();
  const logs = AGENT_LOG.filter(l => l.at <= t).slice(-12);

  return {
    phase: step.phase,
    avatarState: step.state,
    caption: step.caption,
    t,
    totalDuration,
    comments,
    logs,
    playing,
    toggle: () => setPlaying(p => !p),
    reset: () => { setT(0); lastRef.current = null; },
    seek: (ms) => setT(ms),
  };
}

// Format ms as mm:ss
function fmtTime(ms) {
  const s = Math.floor(ms / 1000);
  return `${String(Math.floor(s / 60)).padStart(2,'0')}:${String(s % 60).padStart(2,'0')}`;
}

Object.assign(window, { LIVE_SCRIPT, LIVE_COMMENTS, AGENT_LOG, useEmpireShow, fmtTime });
