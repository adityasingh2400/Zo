// StreamView.swift — viewer-facing EMPIRE surface.
//
// Replaces the voice-driven whisper/router/Gemma debug view as the main
// iPhone screen. Based on Cody's sketch:
//
//   ┌───────────────┬──────┐
//   │               │ 3D   │   3D revolving video
//   │   HEAD SHOT   │video │   of the current product
//   │   (avatar)    ├──────┤
//   │               │ best │   3D rotating images
//   │               │ N    │   = best intake frames
//   ├───────────────┤ frms │
//   │               │      │
//   │   COMMENTS    ├──────┤
//   │   ~~~~~       │ BUY  │   Link to buy
//   │   ~~~~        │      │
//   └───────────────┴──────┘
//
// Voice / push-to-talk / whisper / local-router UI removed — Mac owns those
// paths now. iPhone is the SELLER's viewer + Capture (via FeatureFlags).
//
// Data sources (all via EmpireSocket's /ws/dashboard subscription):
//   - Head shot:       VideoDirector plays avatar state MP4s
//   - 3D spin:         view_3d WS event (kind=frames|glb, url|frames)
//   - Best frames:     GET /api/best_frames on mount + product_data events
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
    @State private var bestFrames: [String] = []  // base64-encoded JPEGs
    @State private var bestFramesIndex: Int = 0

    var body: some View {
        GeometryReader { geo in
            HStack(spacing: 10) {
                // LEFT COLUMN — head shot on top, comments below
                VStack(spacing: 10) {
                    headShot
                        .frame(maxHeight: .infinity)
                    commentsCard
                        .frame(height: geo.size.height * 0.38)
                }
                .frame(width: geo.size.width * 0.58)

                // RIGHT COLUMN — 3D spin / best frames / buy
                VStack(spacing: 10) {
                    threeDSpinCard
                        .frame(height: geo.size.height * 0.30)
                    bestFramesCard
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
            await refreshBestFrames()
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
            Text("LIVE")
                .font(.system(size: 9, weight: .heavy, design: .monospaced))
                .tracking(1.2)
                .foregroundColor(.white)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(.ultraThinMaterial.opacity(0.5), in: Capsule())
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
            RoundedRectangle(cornerRadius: 18)
                .fill(Color.white.opacity(0.04))
                .overlay(
                    RoundedRectangle(cornerRadius: 18)
                        .stroke(Color.white.opacity(0.08), lineWidth: 1)
                )

            if let view3d = socket.view3d, let frames = view3d.frames, !frames.isEmpty {
                // Rotating through the 3D frames manifest (threed agent output)
                SpinningFrames(frameURLs: frames.map { relativeURL($0) })
                    .padding(8)
            } else {
                VStack(spacing: 4) {
                    Image(systemName: "cube.transparent")
                        .font(.system(size: 24))
                        .foregroundColor(.white.opacity(0.3))
                    Text("3D SPIN")
                        .font(.system(size: 9, weight: .heavy, design: .monospaced))
                        .tracking(1.2)
                        .foregroundColor(.white.opacity(0.4))
                    Text("waiting for intake")
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundColor(.white.opacity(0.25))
                }
            }
        }
    }

    // MARK: - Best frames card (middle-right)

    private var bestFramesCard: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 18)
                .fill(Color.white.opacity(0.04))
                .overlay(
                    RoundedRectangle(cornerRadius: 18)
                        .stroke(Color.white.opacity(0.08), lineWidth: 1)
                )

            if !bestFrames.isEmpty {
                frameRotator
                    .padding(8)
            } else {
                VStack(spacing: 4) {
                    Image(systemName: "square.stack.3d.up")
                        .font(.system(size: 24))
                        .foregroundColor(.white.opacity(0.3))
                    Text("BEST FRAMES")
                        .font(.system(size: 9, weight: .heavy, design: .monospaced))
                        .tracking(1.2)
                        .foregroundColor(.white.opacity(0.4))
                    Text("film a product to begin")
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundColor(.white.opacity(0.25))
                }
            }
        }
    }

    private var frameRotator: some View {
        VStack(spacing: 6) {
            // Active frame
            Group {
                if let img = UIImage.fromBase64(bestFrames[safe: bestFramesIndex] ?? "") {
                    Image(uiImage: img)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                } else {
                    Color.white.opacity(0.05)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 12))

            // Thumbnail strip
            HStack(spacing: 4) {
                ForEach(Array(bestFrames.enumerated().prefix(6)), id: \.offset) { idx, b64 in
                    let active = idx == bestFramesIndex
                    Group {
                        if let img = UIImage.fromBase64(b64) {
                            Image(uiImage: img)
                                .resizable()
                                .aspectRatio(contentMode: .fill)
                        } else {
                            Color.white.opacity(0.05)
                        }
                    }
                    .frame(height: 28)
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(active ? Color.white : Color.white.opacity(0.15),
                                    lineWidth: active ? 1.5 : 1)
                    )
                    .onTapGesture { bestFramesIndex = idx }
                }
            }
            .frame(maxWidth: .infinity)
        }
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
            HStack(spacing: 8) {
                Image(systemName: "cart.fill")
                    .font(.system(size: 14, weight: .semibold))
                Text(buyLabel)
                    .font(.system(size: 13, weight: .heavy, design: .monospaced))
                    .tracking(1.2)
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
            }
            .foregroundColor(.black)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 14)
            .background(
                LinearGradient(
                    colors: [Color.white, Color(white: 0.92)],
                    startPoint: .top, endPoint: .bottom
                )
            )
            .clipShape(RoundedRectangle(cornerRadius: 16))
            .shadow(color: .white.opacity(0.2), radius: 16, x: 0, y: 0)
        }
        .disabled(socket.productData?.buy_url == nil)
        .opacity(socket.productData?.buy_url == nil ? 0.5 : 1)
    }

    private var buyLabel: String {
        if let price = socket.productData?.price, !price.isEmpty {
            return "BUY · \(price)"
        }
        return "LINK TO BUY"
    }

    // MARK: - Data

    private func refreshBestFrames() async {
        guard let host = GemmaClient.backendHost else { return }
        guard let url = URL(string: "http://\(host):8000/api/best_frames") else { return }
        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let frames = obj["frames"] as? [String] {
                await MainActor.run {
                    bestFrames = frames
                    bestFramesIndex = 0
                }
            }
        } catch {
            // silent — card shows the empty state
        }
    }

    private func relativeURL(_ path: String) -> URL? {
        guard let host = GemmaClient.backendHost else { return nil }
        if path.hasPrefix("http") { return URL(string: path) }
        return URL(string: "http://\(host):8000\(path)")
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

extension UIImage {
    static func fromBase64(_ s: String) -> UIImage? {
        guard !s.isEmpty, let data = Data(base64Encoded: s) else { return nil }
        return UIImage(data: data)
    }
}
