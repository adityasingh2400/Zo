import React, { useEffect, useMemo, useRef, useState } from 'react';

const API_BASE = `http://${window.location.hostname}:8000`;

/**
 * Spin3D — turns a stack of pre-rendered angle PNGs into something that
 * reads like a real 3D-rendered object on stage. The whole thing is
 * on-device — WebGL2 fragment shader doing studio relighting, rim light,
 * floor shadow, motion blur, color grade, vignette. Zero server roundtrip.
 *
 * Why not just play the PNGs as a flipbook (the previous design): a viewer
 * can always catch the seam between frames, the lighting is whatever the
 * source video had, and there's no sense of an actual light source. The
 * shader pipeline below fakes a stage light that orbits relative to the
 * camera, gives the object an alpha-edge rim, and drops a contact shadow.
 *
 * Falls back to the legacy CSS opacity crossfade if WebGL2 isn't available.
 *
 * Props (all but `view` optional):
 *   view    : { kind: 'frame_carousel', frames: [...], ms } | { kind: 'glb', url }
 *   height  : px height of the canvas / GLB viewer
 *   label   : big readable product name shown on the active item
 *   state   : 'idle' | 'listening' | 'thinking' | 'responding'
 *             reactive style — rim brightens on listening, slows on thinking,
 *             accent flashes on responding. Drives the voice-flow narrative.
 *   accent  : CSS hex for the rim/accent (default: brand purple)
 *   theme   : 'studio_light' (white seamless paper, soft gray shadow, floor
 *             reflection — the high-end e-commerce look) | 'studio_dark'
 *             (deep stage with rim light — the dramatic look). Default: light.
 *   secondsPerRev : full rotation period at idle speed (default 4.2s)
 *   autoSpin: false to lock to drag-only mode (default true)
 */
export function Spin3D({
  view,
  height = 220,
  label,
  state = 'idle',
  accent = '#7c3aed',
  theme = 'studio_light',
  secondsPerRev = 4.2,
  autoSpin = true,
}) {
  if (!view) return null;
  if (view.kind === 'glb' && view.url) {
    return <GLBView url={view.url} height={height} label={label} state={state} accent={accent} />;
  }
  if (view.kind === 'frame_carousel' && view.frames?.length) {
    return (
      <CarouselSpin
        frames={view.frames}
        height={height}
        label={label}
        state={state}
        accent={accent}
        theme={theme}
        secondsPerRev={secondsPerRev}
        autoSpin={autoSpin}
      />
    );
  }
  return null;
}

// ── Theme presets — the visual character of the stage ───────────────────────
// Each value here maps directly to a shader uniform. Adjusting these means
// no GLSL changes — pure JS knobs to tune the look.
//
// studio_light: high-end e-commerce look. Pure-white seamless paper, subtle
//   floor curve, soft cool-gray shadow, faint cool-tint rim, gentle floor
//   reflection. The Apple product-page aesthetic.
// studio_dark: dramatic stage look. Deep radial backdrop, accent-colored rim
//   that orbits with the spin, cinematic vignette. Better for OLED / dark UIs.
const THEMES = {
  studio_light: {
    bgTop:        [1.00, 1.00, 1.01],   // crisp white at top
    bgBot:        [0.86, 0.87, 0.91],   // light cool-gray at floor
    bgFloorY:     0.55,                 // where wall transitions into floor
    rimColor:     [0.55, 0.62, 0.85],   // cool blue rim — reads as photo light
    rimGain:      0.35,
    bloomGain:    0.0,                  // bloom on white = blowout, kill it
    shadowColor:  [0.18, 0.20, 0.30],   // cool gray shadow
    shadowGain:   0.55,
    reflectionGain: 0.32,                // floor reflection strength
    vignetteGain: 0.06,                 // barely-there
    backdropAlpha: 1.0,                 // backdrop fully opaque
    motionBlurGain: 0.0,                // OFF on white — alpha halos look like ghosts
  },
  studio_dark: {
    bgTop:        [0.105, 0.108, 0.130],
    bgBot:        [0.018, 0.020, 0.028],
    bgFloorY:     0.42,
    rimColor:     null,                 // null = use accent prop
    rimGain:      1.0,
    bloomGain:    1.0,
    shadowColor:  [0.0, 0.0, 0.0],
    shadowGain:   0.55,
    reflectionGain: 0.0,
    vignetteGain: 0.22,
    backdropAlpha: 1.0,
    motionBlurGain: 0.45,               // moderate on dark — too much smears
  },
};

// ── Reactive style derived from `state` prop ─────────────────────────────────
// All shader-side knobs live here so the visual character stays consistent
// across every <Spin3D> on the page.
function styleForState(state) {
  switch (state) {
    case 'listening':
      return { speedMul: 0.85, rimGain: 1.6, bloomGain: 1.25, label: 'LISTENING' };
    case 'thinking':
      return { speedMul: 0.45, rimGain: 0.6, bloomGain: 0.7, label: 'THINKING' };
    case 'responding':
      return { speedMul: 1.15, rimGain: 2.1, bloomGain: 1.6, label: 'RESPONDING' };
    case 'idle':
    default:
      return { speedMul: 1.0, rimGain: 1.0, bloomGain: 1.0, label: '' };
  }
}

// ── Spring physics for the rotation speed ────────────────────────────────────
// We never set the speed directly — we set a target and let a critically-damped
// spring chase it. Result: state changes feel physical instead of teleporting.
function stepSpring(curr, target, vel, k = 18, d = 7, dt = 1 / 60) {
  const f = -k * (curr - target) - d * vel;
  const newVel = vel + f * dt;
  const newCurr = curr + newVel * dt;
  return [newCurr, newVel];
}

// ── Image loader. Returns a promise that resolves to an HTMLImageElement[]. ──
function loadImages(urls) {
  return Promise.all(
    urls.map(
      (url) =>
        new Promise((resolve) => {
          const img = new Image();
          img.crossOrigin = 'anonymous';
          img.onload = () => resolve(img);
          img.onerror = () => resolve(null);
          img.src = `${API_BASE}${url}`;
        }),
    ),
  );
}

// ── WebGL helpers ────────────────────────────────────────────────────────────
function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error(`shader compile failed: ${log}`);
  }
  return sh;
}

function link(gl, vs, fs) {
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(prog);
    gl.deleteProgram(prog);
    throw new Error(`program link failed: ${log}`);
  }
  return prog;
}

// Hex → vec3 in 0..1
function hexToVec3(hex) {
  const m = /^#?([\da-f]{2})([\da-f]{2})([\da-f]{2})$/i.exec(hex);
  if (!m) return [0.49, 0.23, 0.93];
  return [parseInt(m[1], 16) / 255, parseInt(m[2], 16) / 255, parseInt(m[3], 16) / 255];
}

// ── Vertex: full-screen triangle (single tri is faster than two tris) ────────
const VERT = /* glsl */ `#version 300 es
precision highp float;
out vec2 vUv;
const vec2 verts[3] = vec2[3](
  vec2(-1.0, -1.0),
  vec2( 3.0, -1.0),
  vec2(-1.0,  3.0)
);
void main() {
  vec2 p = verts[gl_VertexID];
  vUv = (p + 1.0) * 0.5;
  // Flip Y so texture upload matches HTML image orientation.
  vUv.y = 1.0 - vUv.y;
  gl_Position = vec4(p, 0.0, 1.0);
}
`;

// ── Fragment: studio relight pipeline ────────────────────────────────────────
// Inputs: a 2D texture array of bg-removed PNG frames, a floating phase
// (0..frameCount), time, and reactive state knobs.
//
// Pipeline (in order):
//   1. Sample lower / upper / next frames; blend by frac with optional
//      temporal blur (sells fast-rotation as proper motion blur).
//   2. Composite over a soft radial studio backdrop (warm top-light, cool floor).
//   3. Soft elliptical contact shadow under the object (driven by alpha density
//      sampled along the bottom band of the frame).
//   4. Alpha-edge rim light using fwidth() as the edge mask, tinted by accent,
//      modulated by rotation phase + a slow breathing wave.
//   5. Cheap bloom: square-blur the bright pixels of the comp and add back.
//   6. S-curve color grade + saturation lift.
//   7. Vignette.
const FRAG = /* glsl */ `#version 300 es
precision highp float;
precision highp sampler2DArray;

in vec2 vUv;
out vec4 fragColor;

uniform sampler2DArray uFrames;
uniform int   uFrameCount;
uniform float uPhase;          // 0..uFrameCount, fractional
uniform float uTime;           // seconds
uniform vec2  uResolution;
uniform vec3  uAccent;
uniform vec3  uRimColor;       // theme-controlled rim tint
uniform float uRimGain;
uniform float uBloomGain;
uniform float uMotionBlur;     // 0..1, scaled with speed
uniform float uHighlight;      // 0..1, set on state change

// Theme-controlled stage uniforms
uniform vec3  uBgTop;
uniform vec3  uBgBot;
uniform float uBgFloorY;       // y in 0..1 where wall meets floor
uniform vec3  uShadowColor;
uniform float uShadowGain;
uniform float uReflectionGain; // 0 = off, ~0.3 = subtle floor reflection
uniform float uVignetteGain;

// Square-fit UV: takes the canvas-relative uv (0..1 in both axes) and remaps
// it so the square texture is centered, never stretched. Areas outside the
// square fit return uv outside [0,1], which we use as a transparency mask.
vec2 squareFitUv(vec2 uv, vec2 res) {
  float canvasAspect = res.x / res.y;
  vec2 fit = uv;
  if (canvasAspect > 1.0) {
    // Canvas is wider than tall — letterbox horizontally.
    fit.x = (uv.x - 0.5) * canvasAspect + 0.5;
  } else {
    // Canvas is taller than wide — letterbox vertically.
    fit.y = (uv.y - 0.5) / canvasAspect + 0.5;
  }
  return fit;
}

// Sample object at a given fractional phase (linear blend of two adjacent layers).
// Returns transparent for samples outside the square fit area.
vec4 sampleObj(vec2 uv, float phase) {
  vec2 fit = squareFitUv(uv, uResolution);
  if (fit.x < 0.0 || fit.x > 1.0 || fit.y < 0.0 || fit.y > 1.0) {
    return vec4(0.0);
  }
  float wrapped = mod(phase, float(uFrameCount));
  float lo = floor(wrapped);
  float hi = mod(lo + 1.0, float(uFrameCount));
  float t = wrapped - lo;
  vec4 ca = texture(uFrames, vec3(fit, lo));
  vec4 cb = texture(uFrames, vec3(fit, hi));
  return mix(ca, cb, t);
}

// Cyclorama backdrop: a vertical gradient that simulates seamless studio
// paper. The "floor curve" at uBgFloorY is a soft transition from wall to
// floor. On light theme this reads as the iconic Apple-page white-on-light-
// gray look; on dark theme it's the dramatic radial. Same shader, different
// uniforms.
vec3 backdrop(vec2 uv) {
  // Vertical wall-to-floor sweep with a soft curve at the floor line.
  float floorT = smoothstep(uBgFloorY - 0.10, uBgFloorY + 0.20, uv.y);
  vec3 base = mix(uBgTop, uBgBot, floorT);
  // Subtle horizontal vignette so the corners have a touch of depth even on
  // a flat-white setup. Skipped entirely if uVignetteGain is 0.
  if (uVignetteGain > 0.001) {
    float r = length(uv - vec2(0.5, 0.5));
    base *= mix(1.0 - uVignetteGain, 1.0, smoothstep(1.05, 0.42, r));
  }
  return base;
}

// "Contact line": where the bottom of the actual product silhouette sits
// in canvas space. We compute it dynamically by scanning down from the
// top and finding the lowest non-transparent pixel of the object — that
// way the shadow + reflection track the product as it rotates instead of
// being pinned to a fixed canvas y.
//
// Returns y in 0..1, or 1.0 if no opaque pixel found in the column.
float productContactY(vec2 uv, float phase) {
  // 8-tap vertical sample sweeping down. Cheap and gives sub-pixel-ish
  // accuracy for the bottom edge across the spin.
  float bestY = 1.0;
  for (int i = 0; i < 16; ++i) {
    float t = float(i) / 15.0;
    float y = mix(0.20, 0.92, t);
    float a = sampleObj(vec2(uv.x, y), phase).a;
    if (a > 0.18) bestY = y;
  }
  return bestY;
}

void main() {
  vec2 uv = vUv;

  // 1. Object sample with optional temporal blur
  vec4 obj;
  if (uMotionBlur > 0.001) {
    float dp = uMotionBlur * 0.55;
    vec4 a = sampleObj(uv, uPhase - dp);
    vec4 b = sampleObj(uv, uPhase);
    vec4 c = sampleObj(uv, uPhase + dp);
    obj = a * 0.27 + b * 0.46 + c * 0.27;
  } else {
    obj = sampleObj(uv, uPhase);
  }

  // 2. Backdrop (cyclorama: white-to-light-gray sweep on light theme,
  //    dark radial on dark theme — driven by uBg* uniforms).
  vec3 col = backdrop(uv);

  // 3. Floor reflection — sample the object MIRRORED across the contact
  //    line and fade with distance. Adds the "polished surface" look that
  //    high-end product photos have. Light theme on, dark theme off.
  if (uReflectionGain > 0.001) {
    float floorY = 0.78;
    float reflectY = 2.0 * floorY - uv.y;
    if (uv.y > floorY && reflectY > 0.0 && reflectY < floorY) {
      vec4 mirror = sampleObj(vec2(uv.x, reflectY), uPhase);
      // Distance-based fade: full strength at the contact line, gone by
      // the bottom of the canvas. Squared falloff = realistic.
      float dist = (uv.y - floorY) / (1.0 - floorY);
      float fade = pow(max(0.0, 1.0 - dist), 1.6) * uReflectionGain;
      // Reflection is darker + slightly desaturated (mirrors absorb energy).
      vec3 refColor = mirror.rgb * 0.72;
      float lum = dot(refColor, vec3(0.299, 0.587, 0.114));
      refColor = mix(vec3(lum), refColor, 0.85);
      col = mix(col, refColor, mirror.a * fade);
    }
  }

  // 4. Contact shadow — sample object alpha at a band just above the floor
  //    and project as a soft ellipse below. Driven by uShadow* so light
  //    theme gets a cool gray, dark theme gets pure black.
  {
    float floorY = 0.78;
    float shadowFalloff = smoothstep(floorY - 0.04, floorY + 0.22, uv.y);
    float shadowDensity = 0.0;
    // 7-tap horizontal sample → softer shadow than the old 5-tap
    for (int i = -3; i <= 3; ++i) {
      float dx = float(i) * 0.035;
      vec2 sUv = vec2(uv.x + dx, floorY - 0.05);
      shadowDensity += sampleObj(sUv, uPhase).a;
    }
    shadowDensity /= 7.0;
    // Ellipse — slightly wider & flatter than the old version so the
    // shadow reads as soft diffuse light from above.
    float ex = (uv.x - 0.5) * 1.25;
    float ey = (uv.y - 0.93) * 4.0;
    float ellipse = 1.0 - smoothstep(0.0, 1.0, sqrt(ex * ex + ey * ey));
    float shadowMask = shadowDensity * ellipse * shadowFalloff * uShadowGain;
    // Tinted shadow: blend toward uShadowColor instead of pure subtract.
    // Realistic on white, identical to the old look on dark (shadow=black).
    col = mix(col, uShadowColor, shadowMask);
  }

  // 5. Composite the object over the lit backdrop
  col = mix(col, obj.rgb, obj.a);

  // 6. Rim light. fwidth(alpha) gives us an edge band naturally; we widen
  //    it with smoothstep, then modulate by an azimuth wave so the rim
  //    "rotates around" the object as it spins, faking a fixed light.
  if (uRimGain > 0.001) {
    float edge = fwidth(obj.a);
    float rimMask = smoothstep(0.0, 0.6, edge) * (1.0 - smoothstep(0.55, 0.95, obj.a));
    float ang = uPhase * 0.40 + uTime * 0.18;
    float azimuth = 0.55 + 0.45 * cos((uv.x - 0.5) * 6.2832 + ang);
    float breath = 0.85 + 0.15 * sin(uTime * 1.4);
    vec3 rim = uRimColor * rimMask * azimuth * breath * uRimGain * 1.4;
    col += rim;
  }

  // 7. Cheap bloom — sample 4 corners, blur, threshold-add. Disabled on
  //    light themes (would just blow out the white background).
  if (uBloomGain > 0.001) {
    vec2 px = 1.0 / uResolution;
    vec3 b1 = sampleObj(uv + vec2( 1.5,  1.5) * px, uPhase).rgb;
    vec3 b2 = sampleObj(uv + vec2(-1.5,  1.5) * px, uPhase).rgb;
    vec3 b3 = sampleObj(uv + vec2( 1.5, -1.5) * px, uPhase).rgb;
    vec3 b4 = sampleObj(uv + vec2(-1.5, -1.5) * px, uPhase).rgb;
    vec3 blur = (b1 + b2 + b3 + b4) * 0.25;
    vec3 bright = max(blur - 0.62, vec3(0.0));
    col += bright * 0.55 * uBloomGain;
  }

  // 8. Color grade — gentle S-curve + saturation lift
  col = pow(col, vec3(0.96));
  float lum = dot(col, vec3(0.299, 0.587, 0.114));
  col = mix(vec3(lum), col, 1.08);

  // 9. Highlight pulse — used for "responding" state. Brief brightness
  //    lift that the JS side ramps up then decays; doesn't affect alpha.
  col += uAccent * uHighlight * 0.18;

  fragColor = vec4(col, 1.0);
}
`;

// ── The main carousel ────────────────────────────────────────────────────────
function CarouselSpin({
  frames,
  height,
  label,
  state,
  accent,
  theme,
  secondsPerRev,
  autoSpin,
}) {
  const stateStyle = useMemo(() => styleForState(state), [state]);
  const accentVec = useMemo(() => hexToVec3(accent), [accent]);
  const themeCfg = useMemo(() => THEMES[theme] || THEMES.studio_light, [theme]);
  // rim color: theme override, or fallback to accent
  const rimVec = useMemo(
    () => themeCfg.rimColor || accentVec,
    [themeCfg, accentVec],
  );

  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const fallbackRef = useRef(null); // for non-WebGL2 path
  const glStateRef = useRef(null);
  const rafRef = useRef(0);
  const dragStartRef = useRef(null);
  const inertiaVelRef = useRef(0); // frames-per-second from a release
  const speedSpringRef = useRef({ curr: 0, vel: 0 });
  const highlightRef = useRef(0);
  // Phase truth lives in a ref (advanced every rAF tick). The state mirror
  // is used only to drive the scrub bar UI and the CSS fallback's two-frame
  // crossfade. Scheduling setPhase every tick would re-render at 60Hz, so
  // we throttle to whole-frame transitions.
  const phaseRef = useRef(0);
  // Ping-pong state. The carousel is NOT a true 360° capture — the seller
  // films a partial sweep, so wrapping from frame N back to frame 0 produces
  // a visible jump cut. Instead we bounce at the ends: forward → 1s dwell at
  // the last frame → reverse at the same speed → 1s dwell at frame 0 → repeat.
  // direction = +1 (forward) or -1 (reverse). dwellUntil = perf.now() ts;
  // while now < dwellUntil the spin holds still on the boundary frame.
  const directionRef = useRef(1);
  const dwellUntilRef = useRef(0);
  // Dwell time at each end before reversing. 1000ms reads as a deliberate
  // "look at this side" beat without feeling like the demo froze.
  const PINGPONG_DWELL_MS = 1000;

  const [phase, setPhase] = useState(0);
  const [paused, setPaused] = useState(false);
  const [loaded, setLoaded] = useState(0);
  const [revealing, setRevealing] = useState(true); // initial fade-in
  const [glReady, setGlReady] = useState(false);
  const [glFailed, setGlFailed] = useState(false);

  // ── Set up WebGL2 once ─────────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const gl = canvas.getContext('webgl2', { premultipliedAlpha: true, antialias: true });
    if (!gl) {
      setGlFailed(true);
      return;
    }
    let prog, vao, tex;
    try {
      const vs = compile(gl, gl.VERTEX_SHADER, VERT);
      const fs = compile(gl, gl.FRAGMENT_SHADER, FRAG);
      prog = link(gl, vs, fs);
      vao = gl.createVertexArray();
      tex = gl.createTexture();
    } catch (err) {
      console.warn('[Spin3D] webgl init failed', err);
      setGlFailed(true);
      return;
    }
    glStateRef.current = {
      gl, prog, vao, tex,
      uniforms: {
        uFrames: gl.getUniformLocation(prog, 'uFrames'),
        uFrameCount: gl.getUniformLocation(prog, 'uFrameCount'),
        uPhase: gl.getUniformLocation(prog, 'uPhase'),
        uTime: gl.getUniformLocation(prog, 'uTime'),
        uResolution: gl.getUniformLocation(prog, 'uResolution'),
        uAccent: gl.getUniformLocation(prog, 'uAccent'),
        uRimColor: gl.getUniformLocation(prog, 'uRimColor'),
        uRimGain: gl.getUniformLocation(prog, 'uRimGain'),
        uBloomGain: gl.getUniformLocation(prog, 'uBloomGain'),
        uMotionBlur: gl.getUniformLocation(prog, 'uMotionBlur'),
        uHighlight: gl.getUniformLocation(prog, 'uHighlight'),
        uBgTop: gl.getUniformLocation(prog, 'uBgTop'),
        uBgBot: gl.getUniformLocation(prog, 'uBgBot'),
        uBgFloorY: gl.getUniformLocation(prog, 'uBgFloorY'),
        uShadowColor: gl.getUniformLocation(prog, 'uShadowColor'),
        uShadowGain: gl.getUniformLocation(prog, 'uShadowGain'),
        uReflectionGain: gl.getUniformLocation(prog, 'uReflectionGain'),
        uVignetteGain: gl.getUniformLocation(prog, 'uVignetteGain'),
      },
      uploaded: false,
      texSize: 0,
    };
    return () => {
      const s = glStateRef.current;
      if (!s) return;
      try { s.gl.deleteTexture(s.tex); } catch {}
      try { s.gl.deleteProgram(s.prog); } catch {}
      try { s.gl.deleteVertexArray(s.vao); } catch {}
      glStateRef.current = null;
    };
  }, []);

  // ── Load frames + upload to GPU ────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    setLoaded(0);
    setGlReady(false);
    setRevealing(true);

    const tally = (n) => { if (!cancelled) setLoaded(n); };

    loadImages(frames).then((imgs) => {
      if (cancelled) return;
      tally(imgs.filter(Boolean).length);

      const s = glStateRef.current;
      if (!s) return; // WebGL not ready (or failed); fallback path will paint
      const { gl } = s;
      // texStorage3D is immutable per texture, so we recreate the texture on
      // every frame-set change (e.g. new product). Delete old, allocate new.
      try { gl.deleteTexture(s.tex); } catch {}
      const tex = gl.createTexture();
      s.tex = tex;
      // Pick the largest power-of-two square that fits all images, capped.
      const srcSize = imgs[0]?.naturalWidth || 512;
      const target = Math.min(1024, Math.max(256, nextPow2(srcSize)));
      s.texSize = target;

      gl.bindTexture(gl.TEXTURE_2D_ARRAY, tex);
      // Mipmapping gives clean sampling at any zoom + better trilinear quality.
      // Anisotropic filtering on top would help even more if available.
      const mipLevels = Math.floor(Math.log2(target)) + 1;
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texStorage3D(gl.TEXTURE_2D_ARRAY, mipLevels, gl.RGBA8, target, target, imgs.length);
      // Anisotropic filtering — huge quality lift for textures viewed at angles
      // or partial scales. ~16x is the cap on most GPUs and basically free.
      const anisoExt = gl.getExtension('EXT_texture_filter_anisotropic');
      if (anisoExt) {
        const maxAniso = gl.getParameter(anisoExt.MAX_TEXTURE_MAX_ANISOTROPY_EXT);
        gl.texParameterf(gl.TEXTURE_2D_ARRAY, anisoExt.TEXTURE_MAX_ANISOTROPY_EXT, Math.min(16, maxAniso));
      }

      // Resize each image to a square via 2D canvas (cheap; fires once per load).
      const scratch = document.createElement('canvas');
      scratch.width = scratch.height = target;
      const sctx = scratch.getContext('2d', { willReadFrequently: false });
      imgs.forEach((img, i) => {
        if (!img) return;
        sctx.clearRect(0, 0, target, target);
        // Fit-to-square (preserve aspect). PNGs are already squared by threed.py
        // but be defensive — never want to stretch the product.
        const ratio = Math.min(target / img.naturalWidth, target / img.naturalHeight);
        const w = img.naturalWidth * ratio;
        const h = img.naturalHeight * ratio;
        sctx.drawImage(img, (target - w) / 2, (target - h) / 2, w, h);
        gl.texSubImage3D(
          gl.TEXTURE_2D_ARRAY, 0,
          0, 0, i, target, target, 1,
          gl.RGBA, gl.UNSIGNED_BYTE, scratch,
        );
      });

      // Build the mip chain — required for LINEAR_MIPMAP_LINEAR sampling.
      gl.generateMipmap(gl.TEXTURE_2D_ARRAY);

      s.uploaded = true;
      setGlReady(true);
      // Cinematic reveal — let it sit for a beat then fade overlay out.
      requestAnimationFrame(() => requestAnimationFrame(() => setRevealing(false)));
    });

    return () => { cancelled = true; };
  }, [frames]);

  // ── Resize observer for crisp DPR rendering ────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
      const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
    });
    ro.observe(canvas);
    return () => ro.disconnect();
  }, []);

  // ── Render / animation loop ───────────────────────────────────────────────
  // Runs whenever frames are loaded, regardless of WebGL availability. Phase
  // always advances; GL draw is conditional. This keeps the CSS fallback
  // animating in browsers without WebGL2.
  useEffect(() => {
    const framesReady = glReady || (glFailed && loaded >= frames.length);
    if (!framesReady) return;
    const start = performance.now();
    let last = start;

    const RENDER_FPS = 60;
    const minDt = 1 / RENDER_FPS;

    function tick(now) {
      const rawDt = Math.min(0.05, (now - last) / 1000);
      const dt = rawDt < minDt ? minDt : rawDt;
      last = now;

      const s = glStateRef.current;
      const canDrawGL = !glFailed && s && s.tex;

      // Compute target speed in frames-per-second based on state + drag/inertia.
      // Direction: signed via directionRef (+1 forward, -1 reverse) so the
      // ping-pong reversal at each end happens for free in the speed signal.
      const baseFps = (frames.length / Math.max(0.5, secondsPerRev)) * stateStyle.speedMul;
      const inDwell = now < dwellUntilRef.current;
      // While dwelling at an end, the target is 0 — the spring decelerates
      // smoothly into the boundary frame instead of snapping.
      let targetSpeed = (autoSpin && !paused && !inDwell)
        ? baseFps * directionRef.current
        : 0;
      // Inertia: when user releases a fast drag, ride that velocity for a beat.
      // Drag inertia is signed by the drag itself, so it can carry the spin
      // either direction independent of directionRef.
      if (Math.abs(inertiaVelRef.current) > 0.01) {
        targetSpeed = inertiaVelRef.current;
        // Decay
        inertiaVelRef.current *= Math.max(0, 1 - dt * 2.4);
        if (Math.abs(inertiaVelRef.current) < 0.05) inertiaVelRef.current = 0;
      }

      // Spring the actual speed toward the target — feels physical.
      const sp = speedSpringRef.current;
      const [c1, v1] = stepSpring(sp.curr, targetSpeed, sp.vel, 14, 6, dt);
      sp.curr = c1; sp.vel = v1;

      // Direct speed from the spring. The legacy "seam wobble" was a hack to
      // disguise the wrap from frame N → 0 — irrelevant in ping-pong mode
      // because there's no wrap, so we drop the ±5% sinusoidal modulation.
      const speed = sp.curr;

      // Advance the ref-truth. We never read React `phase` here.
      const prevWhole = Math.floor(phaseRef.current);
      const lastIdx = frames.length - 1;
      const nextPhase = phaseRef.current + speed * dt;

      // Ping-pong boundary handling. When the spin would cross either end,
      // clamp to the boundary frame and either:
      //   (a) under inertia/drag → kill the velocity (no bounce; release
      //       inertia hitting a wall should feel like hitting a wall, not
      //       like a magic ricochet)
      //   (b) under auto-spin → enter a 1s dwell on this boundary, then
      //       flip direction so the next non-dwell tick spins back.
      let resolved = nextPhase;
      if (nextPhase >= lastIdx) {
        resolved = lastIdx;
        if (Math.abs(inertiaVelRef.current) > 0.01) {
          inertiaVelRef.current = 0;
        } else if (autoSpin && !paused && !inDwell) {
          dwellUntilRef.current = now + PINGPONG_DWELL_MS;
          directionRef.current = -1;
        }
      } else if (nextPhase <= 0) {
        resolved = 0;
        if (Math.abs(inertiaVelRef.current) > 0.01) {
          inertiaVelRef.current = 0;
        } else if (autoSpin && !paused && !inDwell) {
          dwellUntilRef.current = now + PINGPONG_DWELL_MS;
          directionRef.current = 1;
        }
      }

      phaseRef.current = resolved;
      // Re-render UI affordances only on whole-frame transitions to avoid 60Hz reflow.
      if (Math.floor(resolved) !== prevWhole) {
        setPhase(resolved);
      }
      const renderPhase = resolved;

      // Highlight ramp — ease toward 1 then back to 0.
      const targetHi = state === 'responding' ? 1.0 : 0.0;
      highlightRef.current += (targetHi - highlightRef.current) * Math.min(1, dt * 6);

      // ── GL draw (only on the WebGL2 path) ───────────────────────────────
      if (canDrawGL) {
        const { gl, prog, uniforms, vao, tex } = s;
        const w = canvasRef.current?.width || 1;
        const h = canvasRef.current?.height || 1;
        gl.viewport(0, 0, w, h);
        gl.clearColor(0, 0, 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.useProgram(prog);
        gl.bindVertexArray(vao);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D_ARRAY, tex);
        gl.uniform1i(uniforms.uFrames, 0);
        gl.uniform1i(uniforms.uFrameCount, frames.length);
        gl.uniform1f(uniforms.uPhase, renderPhase);
        gl.uniform1f(uniforms.uTime, (now - start) / 1000);
        gl.uniform2f(uniforms.uResolution, w, h);
        gl.uniform3fv(uniforms.uAccent, accentVec);
        gl.uniform3fv(uniforms.uRimColor, rimVec);
        // Per-state gains stack ON TOP of theme defaults so listening still
        // brightens the rim and thinking still slows the spin even on light.
        gl.uniform1f(uniforms.uRimGain, themeCfg.rimGain * stateStyle.rimGain);
        gl.uniform1f(uniforms.uBloomGain, themeCfg.bloomGain * stateStyle.bloomGain);
        // Motion blur scales with apparent angular velocity, multiplied by
        // the theme's motionBlurGain (0 = disabled — looks bad on white).
        const mbRaw = Math.min(1, Math.abs(speed) / (frames.length / 2.2));
        gl.uniform1f(uniforms.uMotionBlur, mbRaw * (themeCfg.motionBlurGain ?? 1.0));
        gl.uniform1f(uniforms.uHighlight, highlightRef.current);
        // Theme-controlled stage uniforms
        gl.uniform3fv(uniforms.uBgTop, themeCfg.bgTop);
        gl.uniform3fv(uniforms.uBgBot, themeCfg.bgBot);
        gl.uniform1f(uniforms.uBgFloorY, themeCfg.bgFloorY);
        gl.uniform3fv(uniforms.uShadowColor, themeCfg.shadowColor);
        gl.uniform1f(uniforms.uShadowGain, themeCfg.shadowGain);
        gl.uniform1f(uniforms.uReflectionGain, themeCfg.reflectionGain);
        gl.uniform1f(uniforms.uVignetteGain, themeCfg.vignetteGain);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      }

      rafRef.current = requestAnimationFrame(tick);
    }
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
    // We deliberately omit `phase` from deps so the rAF loop keeps running
    // without restarting on each setPhase tick — phase advances via phaseRef.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [glReady, glFailed, loaded, frames.length, paused, autoSpin, secondsPerRev,
      accentVec, rimVec, themeCfg, stateStyle, state]);

  // ── Pointer drag → scrub + inertia ────────────────────────────────────────
  function setPhaseBoth(v) {
    // Ping-pong mode: the carousel isn't a true 360° capture, so dragging
    // (or arrow-key scrubbing) past either end shouldn't wrap to the other
    // side — that creates the same jump-cut we removed from auto-spin.
    // Clamp to [0, lastIdx] so the user's manual scrub respects the same
    // physical bounds as the autospin.
    const clamped = Math.max(0, Math.min(frames.length - 1, v));
    phaseRef.current = clamped;
    setPhase(clamped);
  }
  function onPointerDown(e) {
    setPaused(true);
    inertiaVelRef.current = 0;
    dragStartRef.current = {
      x: e.clientX,
      lastX: e.clientX,
      lastT: performance.now(),
      phase: phaseRef.current,
      vel: 0, // frames-per-second momentum estimate
    };
    e.currentTarget.setPointerCapture?.(e.pointerId);
  }
  function onPointerMove(e) {
    const drag = dragStartRef.current;
    if (!drag) return;
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const dx = e.clientX - drag.x;
    setPhaseBoth(drag.phase + (dx / rect.width) * frames.length);
    // Estimate velocity from the last sample for a clean release-throw.
    const now = performance.now();
    const ddx = e.clientX - drag.lastX;
    const ddt = Math.max(1, now - drag.lastT) / 1000;
    drag.vel = (ddx / rect.width) * frames.length / ddt;
    drag.lastX = e.clientX;
    drag.lastT = now;
  }
  function onPointerUp(e) {
    const drag = dragStartRef.current;
    if (drag) {
      // Apply momentum if the release was fast enough to be intentional.
      if (Math.abs(drag.vel) > frames.length * 0.25) {
        inertiaVelRef.current = drag.vel;
      }
    }
    dragStartRef.current = null;
    e.currentTarget.releasePointerCapture?.(e.pointerId);
    // Resume after a beat unless still hovered.
    setTimeout(() => setPaused(false), 900);
  }

  // ── Keyboard a11y: ←/→ scrub, space pause ─────────────────────────────────
  function onKeyDown(e) {
    if (e.key === 'ArrowLeft') {
      e.preventDefault();
      setPhaseBoth(phaseRef.current - 1);
      setPaused(true);
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      setPhaseBoth(phaseRef.current + 1);
      setPaused(true);
    } else if (e.key === ' ' || e.key === 'Spacebar') {
      e.preventDefault();
      setPaused((p) => !p);
    }
  }

  const ready = glReady || (glFailed && loaded >= frames.length);
  const lower = Math.floor(phase) % frames.length;
  const upper = (lower + 1) % frames.length;
  const frac = phase - Math.floor(phase);

  // Theme-aware UI overlays — labels and badges need to flip light/dark so
  // they stay readable on whichever backdrop the shader is painting.
  const isLight = theme === 'studio_light';
  const labelBarStyle = {
    ...styles.labelBar,
    background: isLight
      ? 'linear-gradient(to top, rgba(255,255,255,0.92), rgba(255,255,255,0))'
      : 'linear-gradient(to top, rgba(0,0,0,0.85), rgba(0,0,0,0))',
  };
  const labelTextStyle = {
    ...styles.labelText,
    color: isLight ? '#0a0a0a' : '#fafafa',
    textShadow: isLight ? 'none' : '0 2px 8px rgba(0,0,0,0.7)',
  };
  const badgeStyle = {
    ...styles.badge,
    background: glFailed
      ? 'rgba(82,82,91,0.85)'
      : isLight ? 'rgba(15,15,18,0.78)' : 'rgba(124,58,237,0.92)',
  };

  return (
    <div
      ref={containerRef}
      style={{ ...styles.box, height,
        background: isLight ? '#fff' : '#000',
        border: isLight ? '1px solid #e4e4e7' : '1px solid #27272a',
      }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => { dragStartRef.current = null; setPaused(false); }}
      onKeyDown={onKeyDown}
      tabIndex={0}
      role="img"
      aria-label={label ? `3D view of ${label}` : '3D product view'}
      title="Drag to rotate · ← → to scrub · space to pause"
    >
      {/* WebGL canvas — primary path */}
      {!glFailed && (
        <canvas
          ref={canvasRef}
          style={{
            ...styles.canvas,
            opacity: glReady && !revealing ? 1 : 0,
            transition: 'opacity 600ms cubic-bezier(0.4, 0.0, 0.2, 1)',
          }}
        />
      )}

      {/* CSS fallback — old opacity crossfade. Only shown if WebGL2 unavailable. */}
      {glFailed && (
        <div ref={fallbackRef} style={styles.fallbackHost}>
          {frames.map((f, i) => {
            let opacity = 0;
            if (i === lower) opacity = 1 - frac;
            else if (i === upper) opacity = frac;
            return (
              <img
                key={f}
                src={`${API_BASE}${f}`}
                alt=""
                draggable={false}
                style={{ ...styles.fallbackFrame, opacity }}
              />
            );
          })}
        </div>
      )}

      {/* Loading shimmer — shown until upload completes */}
      {!ready && (
        <div style={styles.loadingScrim}>
          <div style={styles.loadingShimmer} />
          <div style={styles.loadingText}>
            <span style={styles.loadingDot} />
            <span>composing studio render · {loaded}/{frames.length}</span>
          </div>
        </div>
      )}

      {/* Top-left badge: 3D / state */}
      <div style={styles.badgeStack}>
        <div style={badgeStyle}>
          {glFailed ? '3D · CSS' : 'ON-DEVICE 3D'}
        </div>
        {stateStyle.label && (
          <div style={{ ...styles.statePill, borderColor: accent, color: accent,
                         background: isLight ? 'rgba(255,255,255,0.85)' : 'rgba(9,9,11,0.85)' }}>
            <span style={{ ...styles.statePulse, background: accent }} />
            {stateStyle.label}
          </div>
        )}
      </div>

      {/* Big readable label — bottom of frame, hides during overlay */}
      {label && (
        <div style={labelBarStyle}>
          <div style={labelTextStyle}>{label}</div>
        </div>
      )}

      {/* Scrub indicator — only shown during pause/drag */}
      <div style={{ ...styles.scrubBar, opacity: paused ? 1 : 0 }}>
        {frames.map((_, i) => {
          const active = i === lower;
          const next = i === upper;
          return (
            <div
              key={i}
              style={{
                ...styles.scrubTick,
                background: active
                  ? accent
                  : next
                  ? `${accent}${Math.round((0.3 + 0.7 * frac) * 255).toString(16).padStart(2, '0')}`
                  : '#3f3f46',
              }}
            />
          );
        })}
      </div>
    </div>
  );
}

function nextPow2(n) {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

function GLBView({ url, height, label }) {
  // <model-viewer> is loaded once via index.html script tag.
  return (
    <div style={{ ...styles.box, height }}>
      {/* @ts-ignore */}
      <model-viewer
        src={`${API_BASE}${url}`}
        camera-controls
        auto-rotate
        auto-rotate-delay="0"
        rotation-per-second="30deg"
        interaction-prompt="none"
        style={{ width: '100%', height: '100%', background: '#0a0a0a' }}
      />
      <div style={styles.badgeStack}>
        <div style={styles.badge}>3D · GLB</div>
      </div>
      {label && (
        <div style={styles.labelBar}>
          <div style={styles.labelText}>{label}</div>
        </div>
      )}
    </div>
  );
}

const styles = {
  box: {
    position: 'relative', width: '100%',
    background: '#000',
    borderRadius: 12, overflow: 'hidden',
    cursor: 'grab', userSelect: 'none',
    border: '1px solid #27272a',
    outline: 'none',
  },
  canvas: {
    position: 'absolute', inset: 0,
    width: '100%', height: '100%',
    display: 'block',
  },
  fallbackHost: {
    position: 'absolute', inset: 0,
    background: 'radial-gradient(ellipse at center, #1f1f23 0%, #09090b 100%)',
  },
  fallbackFrame: {
    position: 'absolute', inset: 0,
    width: '100%', height: '100%', objectFit: 'contain',
    pointerEvents: 'none',
  },
  loadingScrim: {
    position: 'absolute', inset: 0,
    background: 'linear-gradient(180deg, #0a0a0a 0%, #09090b 100%)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    overflow: 'hidden',
    zIndex: 4,
  },
  loadingShimmer: {
    position: 'absolute', inset: 0,
    background: 'linear-gradient(110deg, transparent 30%, rgba(124,58,237,0.10) 50%, transparent 70%)',
    backgroundSize: '200% 100%',
    animation: 'spin3dShimmer 1.6s linear infinite',
  },
  loadingText: {
    position: 'relative',
    color: '#a1a1aa', fontSize: 12,
    fontFamily: 'ui-monospace, SFMono-Regular, monospace',
    letterSpacing: 0.5,
    display: 'flex', alignItems: 'center', gap: 8,
  },
  loadingDot: {
    width: 8, height: 8, borderRadius: 4, background: '#7c3aed',
    animation: 'spin3dPulse 1s ease-in-out infinite',
  },
  badgeStack: {
    position: 'absolute', top: 10, left: 10, zIndex: 5,
    display: 'flex', flexDirection: 'column', gap: 6, alignItems: 'flex-start',
  },
  badge: {
    padding: '3px 9px', borderRadius: 999,
    background: 'rgba(124,58,237,0.92)', color: '#fff',
    fontSize: 10, fontWeight: 800, letterSpacing: 1.2,
    backdropFilter: 'blur(4px)',
  },
  statePill: {
    padding: '3px 9px 3px 8px', borderRadius: 999,
    background: 'rgba(9,9,11,0.85)', border: '1px solid',
    fontSize: 10, fontWeight: 800, letterSpacing: 1.2,
    display: 'flex', alignItems: 'center', gap: 6,
    backdropFilter: 'blur(4px)',
  },
  statePulse: {
    width: 6, height: 6, borderRadius: 3,
    animation: 'spin3dPulse 1.1s ease-in-out infinite',
  },
  labelBar: {
    position: 'absolute', left: 0, right: 0, bottom: 0, zIndex: 4,
    padding: '14px 14px 12px',
    background: 'linear-gradient(to top, rgba(0,0,0,0.85), rgba(0,0,0,0))',
    pointerEvents: 'none',
  },
  labelText: {
    color: '#fafafa', fontSize: 18, fontWeight: 800,
    letterSpacing: 0.3, lineHeight: 1.15,
    textShadow: '0 2px 8px rgba(0,0,0,0.7)',
  },
  scrubBar: {
    position: 'absolute', bottom: 8, left: 8, right: 8,
    display: 'flex', gap: 2, height: 3, zIndex: 6,
    transition: 'opacity 200ms ease',
  },
  scrubTick: { flex: 1, borderRadius: 1 },
};

// ── One-shot keyframe injection for the loading shimmer/pulse ────────────────
if (typeof document !== 'undefined' && !document.getElementById('spin3d-keyframes')) {
  const s = document.createElement('style');
  s.id = 'spin3d-keyframes';
  s.innerHTML = `
    @keyframes spin3dShimmer { 0% { background-position: 200% 0 } 100% { background-position: -200% 0 } }
    @keyframes spin3dPulse  { 0%,100% { opacity: 1 } 50% { opacity: 0.45 } }
  `;
  document.head.appendChild(s);
}
