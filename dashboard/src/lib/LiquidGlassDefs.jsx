import React from 'react';

/**
 * LiquidGlassDefs — global SVG filter for the Apple Liquid Glass effect.
 *
 * Renders ONCE at the app root (see main.jsx). The filter is referenced by
 * the .lg-glass-filter CSS class via `filter: url(#lg-distortion)`. It uses
 * fractalNoise + gaussian blur + displacementMap to bend pixels of whatever
 * sits BEHIND a glass layer, the same way light refracts through curved
 * glass. This is the mechanic Apple introduced at WWDC 2025 ("Liquid Glass").
 *
 * Why a separate component?
 *   - The <filter> needs to live inside ONE <svg> in the DOM, accessible by
 *     its id from anywhere. Mounting once at the root keeps the def DRY and
 *     guarantees there's exactly one filter instance the GPU caches.
 *   - The svg itself is 0×0 / display:none so it doesn't take layout space.
 *   - Reference cost is near-zero on modern Chromium/WebKit; the filter
 *     only runs for elements that actually opt in via the CSS class.
 *
 * Tuning (intentionally subtle):
 *   - baseFrequency 0.008 → wavelength ~125px → big, slow ripples (not
 *     tight noise that would read as "static"). Matches the calm Zo demo
 *     aesthetic instead of the playful CodePen demos at scale=70.
 *   - scale=24 → moderate displacement. Visible behind the avatar video,
 *     never wobbly enough to make text or borders look broken. Apple's own
 *     iOS controls land in the 18-32 range.
 *   - seed=92 → arbitrary but fixed. Determinism keeps the noise stable
 *     across reloads so the look doesn't shift between rehearsal runs.
 *
 * Caveats:
 *   - The displacement is invisible against a flat solid color. Surfaces
 *     that sit on the bezel `#000` won't show ripple — they still get the
 *     dark-glass aesthetic from the rest of the .lg-glass recipe (blur,
 *     specular highlight, rim). Reserve heavy-displacement glass for
 *     overlays that sit over the avatar video.
 *   - Compositing cost: each .lg-glass-filter element triggers a separate
 *     filter pass. Don't apply to dozens of moving elements (e.g., the 30+
 *     scrolling chat bubbles inside the phone). Persistent chrome only.
 */
export function LiquidGlassDefs() {
  return (
    <svg
      aria-hidden="true"
      width="0"
      height="0"
      style={{ position: 'absolute', width: 0, height: 0, pointerEvents: 'none' }}
    >
      <defs>
        <filter
          id="lg-distortion"
          x="-10%"
          y="-10%"
          width="120%"
          height="120%"
          colorInterpolationFilters="sRGB"
        >
          <feTurbulence
            type="fractalNoise"
            baseFrequency="0.008 0.008"
            numOctaves="2"
            seed="92"
            result="noise"
          />
          <feGaussianBlur in="noise" stdDeviation="2" result="blurred" />
          <feDisplacementMap
            in="SourceGraphic"
            in2="blurred"
            scale="24"
            xChannelSelector="R"
            yChannelSelector="G"
          />
        </filter>
      </defs>
    </svg>
  );
}
