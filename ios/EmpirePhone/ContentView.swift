// ContentView.swift — app shell. Camera-first flow per Cody's directive:
//   "user opens the app then takes a video of an item then it pushes it
//    to the mac and it uses gemma locally for the rest of the work,
//    it takes what the user says (what the product is exactly)"
//
// Launch logic:
//   1. Cold launch → always open SellerCaptureView full-screen.
//   2. After a successful sell (SellerCaptureView onComplete with a
//      non-nil requestID), phase flips to .hasProduct → StreamView.
//   3. StreamView's refilm button re-opens the capture sheet.
//   4. Backend state is intentionally NOT probed — products.json
//      auto-seeds a default on Mac boot and we don't want that to
//      override the user's intent to film a new product every open.
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
    @State private var showingHostSheet = false
    @State private var hostInput = ""

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
        .alert("Backend host", isPresented: $showingHostSheet) {
            TextField("e.g. 172.20.10.2", text: $hostInput)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .keyboardType(.URL)
            Button("Save") {
                GemmaClient.setBackendHost(hostInput)
                Task { await resolveLaunchPhase() }
            }
            Button("Reset to default", role: .destructive) {
                GemmaClient.setBackendHost(nil)
                Task { await resolveLaunchPhase() }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            let resolved = GemmaClient.backendHost ?? "(none)"
            let source = GemmaClient.hasUserDefaultsOverride ? "runtime override"
                       : "Info.plist / default"
            Text("Current: \(resolved) · \(source)\nRun `ipconfig getifaddr en0` on the Mac")
        }
        .task { await bootstrap() }
    }

    // MARK: - Bootstrap

    private func bootstrap() async {
        director.backToIdle()
        socket.start()
        await resolveLaunchPhase()
    }

    /// Camera-first launch: always land on the capture flow on cold start
    /// or after a host change. Phase moves to .hasProduct only after a
    /// successful sell in the current session (SellerCaptureView's
    /// onComplete sets it). We deliberately don't probe /api/state —
    /// backend-seeded defaults (e.g. the wallet auto-loaded from
    /// products.json at boot) shouldn't override the user's intent to
    /// film a new product every time they open the app.
    private func resolveLaunchPhase() async {
        await MainActor.run {
            phase = .needsProduct
            showingCapture = true
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
    /// dismissed without uploading — lets the user try again. Also the
    /// escape hatch for fixing a wrong backend host: long-press the
    /// TAP TO FILM button to open the host-override alert.
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
            .onLongPressGesture(minimumDuration: 0.6) {
                hostInput = GemmaClient.backendHost ?? ""
                showingHostSheet = true
            }

            // Always-visible tiny footer showing the resolved backend host
            // so when the Mac's IP changes the user sees exactly what the
            // phone is targeting and can long-press to fix.
            VStack(spacing: 2) {
                Text("backend")
                    .font(.system(size: 9, weight: .heavy, design: .monospaced))
                    .tracking(1)
                    .foregroundColor(.white.opacity(0.3))
                Text(GemmaClient.backendHost ?? "(not configured)")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(.white.opacity(0.5))
                Text("long-press the button above to change")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundColor(.white.opacity(0.25))
            }
            .padding(.top, 14)
        }
    }
}
