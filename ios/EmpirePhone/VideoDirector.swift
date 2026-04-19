// VideoDirector.swift — pick a bundled MP4 and play it through a shared AVPlayer.
//
// The phone doesn't render lip-sync on-device (Wav2Lip on A15 = 30-60s per
// clip; infeasible). Instead it plays the *same* MP4s the dashboard serves
// for respond_locally — rendered once on RunPod 5090, baked into the app
// bundle. 10 clips, ~51 MB, plus 2-3 canned bridges.
//
// For escalate_to_cloud the phone just shows a placeholder card — the demo
// point is "the router ran on-device and correctly identified this needs the
// cloud," not "the phone runs Claude + TTS + Wav2Lip." Those stay on the Mac.

import Foundation
import AVKit
import SwiftUI
import Combine

@MainActor
final class VideoDirector: ObservableObject {

    /// What the UI should render right now. Drives the SwiftUI layer that
    /// shows either the AVPlayer, the idle loop, or a text placeholder card.
    enum Scene: Equatable {
        case idle                         // idle loop plays, no decision active
        case playing(clipURL: URL)        // response clip (local answer or canned bridge)
        case thinking                     // escalate_to_cloud — "would route to cloud"
        case blocked                      // block_comment — silent state
    }

    @Published private(set) var scene: Scene = .idle

    let player = AVPlayer()

    // Idle loop URL resolved once at init — falls through gracefully if
    // the file isn't bundled (no idle visual, rest of the app still works).
    private let idleLoopURL: URL?
    private var endObserver: NSObjectProtocol?

    init() {
        self.idleLoopURL = Bundle.main.url(forResource: "idle_loop", withExtension: "mp4")
        startIdleLoop()

        // When a response clip finishes, return to idle. We don't loop
        // response clips — they play once and the avatar settles back.
        endObserver = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemDidPlayToEndTime,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in self?.backToIdle() }
        }
    }

    deinit {
        if let obs = endObserver { NotificationCenter.default.removeObserver(obs) }
    }

    // MARK: - Dispatch

    /// Given a router Decision, pick a clip (or placeholder) and start
    /// playback. Call immediately after the router returns — the video
    /// starts within a frame of the decision badge appearing.
    func dispatch(_ decision: Decision) {
        switch decision.tool {
        case .respondLocally:
            if case .answerId(let id) = decision.args,
               let url = resolveAnswerURL(id)
            {
                play(url)
            } else {
                // Missing asset → pretend we're escalating; shows the card
                // without crashing the demo.
                scene = .thinking
            }

        case .playCannedClip:
            if case .cannedLabel(let label) = decision.args,
               let url = pickBridge(label: label)
            {
                play(url)
            } else {
                scene = .thinking
            }

        case .blockComment:
            // No video; leave idle running and surface the "blocked" card
            // in the UI layer. Matches dashboard UX.
            scene = .blocked

        case .escalateToCloud:
            scene = .thinking
        }
    }

    /// Back to idle when a response ends, or when ContentView times out the
    /// blocked/thinking state.
    func backToIdle() {
        scene = .idle
        startIdleLoop()
    }

    // MARK: - Playback

    private func play(_ url: URL) {
        let item = AVPlayerItem(url: url)
        player.replaceCurrentItem(with: item)
        player.play()
        scene = .playing(clipURL: url)
    }

    private func startIdleLoop() {
        guard let url = idleLoopURL else {
            // No idle clip bundled — leave the player empty and show a
            // static poster at the UI layer.
            player.replaceCurrentItem(with: nil)
            return
        }
        let item = AVPlayerItem(url: url)
        player.replaceCurrentItem(with: item)
        // Only loop the idle — response clips play once.
        item.preferredForwardBufferDuration = 1.0
        player.play()
        // Manual loop by seeking to zero on end-of-item — the end observer
        // at top switches scene to .idle which re-enters here and rebuilds
        // the item. Good enough for a splash state.
    }

    // MARK: - Asset resolution

    /// respond_locally picks a bundled clip by answer_id. The `products.json`
    /// url field already uses filenames like `wallet_real_leather.mp4`; we
    /// strip the path and look for that name in the bundle.
    ///
    /// Falls back to a simple slug `wallet_<answerId>.mp4` so the mapping
    /// works even if products.json is loaded but the `url` field is missing.
    private func resolveAnswerURL(_ answerId: String) -> URL? {
        // Try filename-from-products.json first.
        if let product = ProductLoader.loadActive(),
           let entry = product.qa_index?[answerId]
        {
            let filename = (entry.url as NSString).lastPathComponent
            let stem = (filename as NSString).deletingPathExtension
            if let url = Bundle.main.url(forResource: stem, withExtension: "mp4") {
                return url
            }
        }
        // Fallback: try a simple slug.
        return Bundle.main.url(forResource: "wallet_\(answerId)", withExtension: "mp4")
    }

    /// play_canned_clip rotates through a small pool of bridge clips per
    /// label so the avatar doesn't feel canned. Bundle them as
    /// `bridge_compliment_1.mp4`, `bridge_compliment_2.mp4`, etc.
    private var bridgeRotation: [String: Int] = [:]
    private func pickBridge(label: String) -> URL? {
        let pool = (1...4).compactMap {
            Bundle.main.url(forResource: "bridge_\(label)_\($0)", withExtension: "mp4")
        }
        guard !pool.isEmpty else { return nil }
        let idx = (bridgeRotation[label] ?? 0) % pool.count
        bridgeRotation[label] = idx + 1
        return pool[idx]
    }
}

// MARK: - SwiftUI wrapper

struct AvatarVideoView: UIViewControllerRepresentable {
    let player: AVPlayer

    func makeUIViewController(context: Context) -> AVPlayerViewController {
        let vc = AVPlayerViewController()
        vc.player = player
        vc.showsPlaybackControls = false
        vc.videoGravity = .resizeAspectFill
        vc.view.backgroundColor = .black
        return vc
    }

    func updateUIViewController(_ uiViewController: AVPlayerViewController, context: Context) {}
}
