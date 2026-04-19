// ContentView.swift — app shell. Camera-first flow per Cody's directive:
//   "user opens the app then takes a video of an item then it pushes it
//    to the mac and it uses gemma locally for the rest of the work,
//    it takes what the user says (what the product is exactly)"
//
// Launch logic:
//   1. Fetch /api/state from the Mac backend to see if a product is loaded.
//   2. If no product_data yet: open SellerCaptureView full-screen (film flow).
//   3. If product_data exists: show StreamView directly (pitched state).
//   4. After SellerCaptureView.onComplete: flip to StreamView and refresh.
//
// The seller's narration becomes voice_text on the /api/sell-video upload;
// the Mac's run_sell_pipeline routes through analyze_and_script_gemma first
// (PRODUCT_ANALYSIS_MODEL=auto in .env, defaults to Gemma+fallback).

import SwiftUI
import AVKit

struct ContentView: View {
    @StateObject private var director = VideoDirector()
    @State private var socket = EmpireSocket()

    // Launch gate. While `.checking`, show a splash. Once state resolves we
    // either open the camera (no product yet) or land on StreamView.
    @State private var phase: Phase = .checking
    @State private var showingCapture = false

    enum Phase: Equatable {
        case checking
        case needsProduct      // land on camera
        case hasProduct        // land on StreamView
    }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            switch phase {
            case .checking:
                splash
            case .hasProduct:
                StreamView(
                    director: director,
                    socket: socket,
                    onRefilm: { showingCapture = true }
                )
            case .needsProduct:
                // Render an empty background — SellerCaptureView auto-opens
                // as a fullScreenCover below. If the user closes the camera
                // without uploading, they'll see this void + a "Tap to film"
                // button as an escape hatch.
                restartSurface
            }
        }
        .preferredColorScheme(.dark)
        .fullScreenCover(isPresented: $showingCapture) {
            SellerCaptureView { requestID in
                if requestID != nil {
                    phase = .hasProduct
                }
                showingCapture = false
            }
        }
        .task { await bootstrap() }
    }

    // MARK: - Bootstrap

    private func bootstrap() async {
        director.backToIdle()
        socket.start()
        await resolveLaunchPhase()
    }

    /// Ask the Mac if there's a product loaded. If yes → StreamView.
    /// If no (or if the host isn't reachable) → camera path.
    /// This is a plain HTTP call so it works regardless of WS state.
    private func resolveLaunchPhase() async {
        guard let host = GemmaClient.backendHost,
              let url = URL(string: "http://\(host):8000/api/state") else {
            await MainActor.run {
                phase = .needsProduct
                showingCapture = true
            }
            return
        }
        do {
            var req = URLRequest(url: url)
            req.timeoutInterval = 3
            let (data, _) = try await URLSession.shared.data(for: req)
            let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
            let hasProduct = (obj?["product_data"] as? [String: Any])?["name"] != nil
            await MainActor.run {
                if hasProduct {
                    phase = .hasProduct
                } else {
                    phase = .needsProduct
                    showingCapture = true
                }
            }
        } catch {
            await MainActor.run {
                phase = .needsProduct
                showingCapture = true
            }
        }
    }

    // MARK: - Splash + empty state

    private var splash: some View {
        VStack(spacing: 14) {
            ProgressView().tint(.white)
            Text("Connecting to EMPIRE…")
                .font(.system(size: 13, weight: .medium, design: .monospaced))
                .foregroundColor(.white.opacity(0.7))
        }
    }

    /// Shown when phase=.needsProduct AND the camera sheet has been
    /// dismissed without uploading — lets the user try again.
    private var restartSurface: some View {
        VStack(spacing: 18) {
            Image(systemName: "video.fill")
                .font(.system(size: 40))
                .foregroundColor(.white.opacity(0.4))
            Text("Film your product to begin")
                .font(.system(size: 14, weight: .semibold))
                .foregroundColor(.white.opacity(0.7))
            Button {
                showingCapture = true
            } label: {
                Text("TAP TO FILM")
                    .font(.system(size: 13, weight: .heavy, design: .monospaced))
                    .tracking(1.5)
                    .foregroundColor(.black)
                    .padding(.horizontal, 22)
                    .padding(.vertical, 12)
                    .background(Color.white)
                    .clipShape(Capsule())
            }
        }
    }
}
