// FeatureFlags.swift — compile-time flags for stage-risk-controlled features.
//
// The stage demo is whisper-driven: hold-to-speak → router → local MP4 +
// Mac Gemma card. Anything beyond that path is gated here so we can ship
// experiments without putting the stage-ready flow at risk.
//
// Flip flags, rebuild (⌘R), test. Default: stage-safe (everything off).

import Foundation

enum FeatureFlags {
    /// Phase 0.2 / 1.1: iPhone as the seller's camera — the "film product"
    /// button + upload to /api/sell-video. Off by default so the stage
    /// demo stays whisper-only. Flip to true post-pitch to expose the
    /// Capture flow.
    static let sellerMode: Bool = false
}
