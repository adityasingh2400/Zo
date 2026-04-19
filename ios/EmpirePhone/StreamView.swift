// StreamView.swift — viewer-facing EMPIRE surface.
//
// Replaces the voice-driven whisper/router/Gemma debug view as the main
// iPhone screen. Based on Cody's sketch:
//
//   ┌───────────────┬──────┐
//   │               │      │
//   │   HEAD SHOT   │  3D  │   3D revolving video of the product
//   │   (avatar)    │ spin │   on a white backdrop
//   │               │      │
//   ├───────────────┤      │
//   │   COMMENTS    │      │
//   │   ~~~~~       ├──────┤
//   │   ~~~~        │ BUY  │   Link to buy
//   └───────────────┴──────┘
//
// Voice / push-to-talk / whisper / local-router UI removed — Mac owns those
// paths now. iPhone is the SELLER's viewer + Capture (via FeatureFlags).
//
// The static best-frames tile was removed — the rotating 3D spin already
// shows the product from every angle, so a second image tile was redundant.
//
// Data sources (all via EmpireSocket's /ws/dashboard subscription):
//   - Head shot:       VideoDirector plays avatar state MP4s
//   - 3D spin:         view_3d WS event (kind=frames|glb, url|frames)
//   - Comments:        audience_comment + comment_response_video events
//   - Buy URL:         product_data.buy_url (added to products.json)
//
// Long-press the head shot to open the runtime host-override sheet
// (preserves the UserDefaults path shipped in Phase 0.2).

import SwiftUI
import AVKit
import Combine

struct StreamView: View {
    @ObservedObject var director: VideoDirector
    let socket: EmpireSocket
    /// Called when the user taps the film button — ContentView opens the
    /// camera full-screen. StreamView doesn't manage the sheet because
    /// the camera flow is owned by the app shell (re-enters .needsProduct
    /// if the capture errors mid-upload).
    var onRefilm: (() -> Void)? = nil

    @State private var showingHostSheet = false
    @State private var hostInput = ""
    /// Drives the LIVE pill's ~1Hz pulse — Visibility (p202) + Operant
    /// Conditioning (p144). Flipped once in onAppear; the .animation
    /// repeatForever handles the oscillation.
    @State private var livePulse = false

    var body: some View {
        GeometryReader { geo in
            HStack(spacing: 10) {
                // LEFT COLUMN — head shot on top, comments below
                VStack(spacing: 10) {
                    headShot
                        .frame(maxHeight: .infinity)
                    commentsCard
                        // Hierarchy (p104) + 80/20 Rule (p12) — avatar is
                        // the pitch; comments are secondary. Give the hero
                        // more vertical weight.
                        .frame(height: geo.size.height * 0.28)
                }
                .frame(width: geo.size.width * 0.58)

                // RIGHT COLUMN — 3D spin (grows) + buy
                VStack(spacing: 10) {
                    threeDSpinCard
                        .frame(maxHeight: .infinity)
                    buyButton
                }
                .frame(maxWidth: .infinity)
            }
            .padding(10)
        }
        .background(Color.black.ignoresSafeArea())
        .task {
            socket.start()
        }
        .alert("Backend host", isPresented: $showingHostSheet) {
            TextField("e.g. 192.168.1.42", text: $hostInput)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .keyboardType(.URL)
            Button("Save") { GemmaClient.setBackendHost(hostInput) }
            Button("Reset to default", role: .destructive) {
                GemmaClient.setBackendHost(nil)
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            let resolved = GemmaClient.backendHost ?? "(none)"
            let source = GemmaClient.hasUserDefaultsOverride ? "runtime override"
                       : "Info.plist / default"
            Text("Current: \(resolved) · \(source)\nipconfig getifaddr en0 on the Mac")
        }
    }

    // MARK: - Head shot

    private var headShot: some View {
        ZStack(alignment: .topLeading) {
            AvatarVideoView(player: director.player)
                .clipShape(RoundedRectangle(cornerRadius: 22))
                .overlay(
                    RoundedRectangle(cornerRadius: 22)
                        .stroke(Color.white.opacity(0.08), lineWidth: 1)
                )
                // Top-Down Lighting Bias (p196) — shadow-below reads the
                // avatar card as raised, pulling the eye to the hero.
                .shadow(color: .black.opacity(0.45), radius: 16, x: 0, y: 8)
                .onLongPressGesture(minimumDuration: 0.6) {
                    hostInput = GemmaClient.backendHost ?? ""
                    showingHostSheet = true
                }

            // Floating LIVE pill + "film another" entry — the camera IS the
            // primary way into the app now, so the button is always visible
            // (no longer behind FeatureFlags.sellerMode).
            HStack(spacing: 8) {
                liveTag
                Spacer()
                captureButton
            }
            .padding(12)
        }
    }

    private var liveTag: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(Color.green)
                .frame(width: 6, height: 6)
                .scaleEffect(livePulse ? 1.35 : 1.0)
                .opacity(livePulse ? 0.55 : 1.0)
                .animation(.easeInOut(duration: 0.9).repeatForever(autoreverses: true),
                           value: livePulse)
            Text("LIVE")
                .font(.system(size: 9, weight: .heavy, design: .monospaced))
                .tracking(1.2)
                .foregroundColor(.white)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(.ultraThinMaterial.opacity(0.5), in: Capsule())
        .onAppear { livePulse = true }
    }

    private var captureButton: some View {
        Button {
            onRefilm?()
        } label: {
            Image(systemName: "video.fill")
                .font(.system(size: 12, weight: .semibold))
                .foregroundColor(.white)
                .frame(width: 28, height: 28)
                .background(.ultraThinMaterial.opacity(0.5), in: Circle())
        }
    }

    // MARK: - 3D revolving spin card (top-right)

    private var threeDSpinCard: some View {
        ZStack {
            // White backdrop — product spins against the neutral background
            // e-commerce shoppers expect, and pops against the dark shell.
            RoundedRectangle(cornerRadius: 18)
                .fill(Color.white)
                .overlay(
                    RoundedRectangle(cornerRadius: 18)
                        .stroke(Color.black.opacity(0.08), lineWidth: 1)
                )

            if let view3d = socket.view3d, let frames = view3d.frames, !frames.isEmpty {
                // Rotating through the 3D frames manifest (threed agent output)
                SpinningFrames(frameURLs: frames.map { relativeURL($0) })
                    .padding(8)
            } else {
                VStack(spacing: 4) {
                    Image(systemName: "cube.transparent")
                        .font(.system(size: 24))
                        .foregroundColor(.black.opacity(0.25))
                    Text("3D SPIN")
                        .font(.system(size: 9, weight: .heavy, design: .monospaced))
                        .tracking(1.2)
                        .foregroundColor(.black.opacity(0.35))
                    Text("waiting for intake")
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundColor(.black.opacity(0.2))
                }
            }

            // Product name + price — Proximity (p160) keeps product info
            // with the product image instead of next to the BUY CTA.
            // Hidden until there's actually a name, so empty state stays
            // clean.
            if let name = socket.productData?.name, !name.isEmpty {
                VStack(alignment: .leading, spacing: 2) {
                    Text(name)
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundColor(.black.opacity(0.88))
                        .lineLimit(1)
                        .minimumScaleFactor(0.7)
                    if let price = socket.productData?.price, !price.isEmpty {
                        Text(price)
                            .font(.system(size: 12, weight: .medium, design: .monospaced))
                            .foregroundColor(.black.opacity(0.55))
                    }
                }
                .padding(.horizontal, 12)
                .padding(.bottom, 12)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomLeading)
                .allowsHitTesting(false)
            }
        }
        // Top-Down Lighting Bias (p196) — matches the avatar card's lift.
        .shadow(color: .black.opacity(0.35), radius: 14, x: 0, y: 6)
    }

    // MARK: - Comments card (bottom-left)

    private var commentsCard: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("COMMENTS")
                    .font(.system(size: 9, weight: .heavy, design: .monospaced))
                    .tracking(1.2)
                    .foregroundColor(.white.opacity(0.4))
                Spacer()
                Text("\(socket.comments.count) live")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundColor(.white.opacity(0.35))
            }
            .padding(.horizontal, 12)
            .padding(.top, 10)

            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    if socket.comments.isEmpty {
                        Text("No comments yet")
                            .font(.system(size: 12, design: .monospaced))
                            .italic()
                            .foregroundColor(.white.opacity(0.25))
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                    } else {
                        ForEach(socket.comments.reversed(), id: \.id) { c in
                            commentRow(comment: c)
                        }
                    }
                }
                .padding(.horizontal, 12)
                .padding(.bottom, 10)
            }
        }
        .background(Color.white.opacity(0.04))
        .clipShape(RoundedRectangle(cornerRadius: 18))
        .overlay(
            RoundedRectangle(cornerRadius: 18)
                .stroke(Color.white.opacity(0.08), lineWidth: 1)
        )
    }

    private func commentRow(comment: LiveComment) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(comment.handle)
                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                    .foregroundColor(.white.opacity(0.7))
                Spacer()
                Text(comment.badge)
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundColor(comment.replying
                        ? Color(red: 0.49, green: 0.71, blue: 1.0)
                        : .white.opacity(0.25))
            }
            Text(comment.text)
                .font(.system(size: 13))
                .foregroundColor(.white.opacity(0.9))
                .multilineTextAlignment(.leading)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(8)
        .background(Color.white.opacity(0.03))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    // MARK: - Buy button (bottom-right)

    private var buyButton: some View {
        Button {
            if let urlString = socket.productData?.buy_url,
               let url = URL(string: urlString) {
                UIApplication.shared.open(url)
            }
        } label: {
            HStack(spacing: 10) {
                Image(systemName: "cart.fill")
                    .font(.system(size: 16, weight: .bold))
                Text(buyLabel)
                    .font(.system(size: 15, weight: .heavy, design: .monospaced))
                    .tracking(1.3)
                    .lineLimit(1)
                    .minimumScaleFactor(0.75)
            }
            .foregroundColor(.white)
            .frame(maxWidth: .infinity)
            // Fitts' Law (p82) — big primary action, generous vertical
            // reach at the screen-edge terminal area.
            .frame(minHeight: 56)
            .background(
                // Von Restorff (p204) — saturated warm orange, the one
                // visibly-different element. Stands apart from the
                // green-on-dark LIVE pill and white 3D backdrop.
                LinearGradient(
                    colors: [Color(red: 1.00, green: 0.45, blue: 0.20),
                             Color(red: 0.95, green: 0.28, blue: 0.10)],
                    startPoint: .top, endPoint: .bottom
                )
            )
            .clipShape(RoundedRectangle(cornerRadius: 18))
            .shadow(color: Color(red: 1.00, green: 0.38, blue: 0.15).opacity(0.45),
                    radius: 18, x: 0, y: 6)
        }
        .disabled(socket.productData?.buy_url == nil)
        // Keep the button visually present even when disabled — its
        // absence creates more confusion than a slightly dimmed present.
        .opacity(socket.productData?.buy_url == nil ? 0.7 : 1)
    }

    private var buyLabel: String {
        if let price = socket.productData?.price, !price.isEmpty {
            return "BUY · \(price)"
        }
        return "LINK TO BUY"
    }

    // MARK: - Data

    private func relativeURL(_ path: String) -> URL? {
        guard let base = GemmaClient.backendBaseURL else { return nil }
        if path.hasPrefix("http") { return URL(string: path) }
        // path from the backend starts with "/", e.g. "/renders/..."
        let trimmed = path.hasPrefix("/") ? String(path.dropFirst()) : path
        return base.appendingPathComponent(trimmed)
    }
}

// ── Supporting views ─────────────────────────────────────────────────────

/// Loops through an array of remote image URLs at a steady interval so the
/// 3D-spin card feels like a rotating product. Falls back to a static
/// first frame if the list has <2 items.
struct SpinningFrames: View {
    let frameURLs: [URL?]
    @State private var index: Int = 0
    private let tick = Timer.publish(every: 0.15, on: .main, in: .common).autoconnect()

    var body: some View {
        ZStack {
            if let url = frameURLs[safe: index] ?? nil {
                AsyncImage(url: url) { phase in
                    switch phase {
                    case .success(let img):
                        img.resizable().aspectRatio(contentMode: .fit)
                    default:
                        Color.white.opacity(0.05)
                    }
                }
            } else {
                Color.white.opacity(0.05)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .onReceive(tick) { _ in
            guard frameURLs.count > 1 else { return }
            index = (index + 1) % frameURLs.count
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────

extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}

