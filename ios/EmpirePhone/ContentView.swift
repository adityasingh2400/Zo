// ContentView.swift — single-screen demo surface.
//
// Flow:
//   press+hold → record → release → Cactus whisper transcribe (on phone) →
//   Swift rule-based Router.decide() (on phone) → VideoDirector plays MP4 →
//   in parallel: GemmaClient classify on Mac → "GEMMA 4 (Mac)" card lights up
//
// On-phone LLM classify: not done. Cactus probes (CactusRunner.swift header)
// found no Gemma family that fits both iPhone RAM (~3 GB app ceiling on A15)
// and demo latency (<1s). The phone's router runs the deterministic Swift
// keyword path; Gemma 4 runs on the Mac and fires AFTER the MP4 dispatch
// so the on-phone demo stays sub-1s. The Gemma card is the visible "the
// laptop's on-device LLM verified this" beat.
//
// Colors match the dashboard's RoutingPanel:
//   respond_locally    → green   (#22c55e)
//   play_canned_clip   → purple  (#7c3aed)
//   block_comment      → gray    (#a1a1aa)
//   escalate_to_cloud  → amber   (#f59e0b)
//   GEMMA 4 (Mac)      → magenta (#d946ef)

import SwiftUI
import AVKit

struct ContentView: View {
    @StateObject private var cactus = CactusRunner()
    @StateObject private var recorder = AudioRecorder()
    @StateObject private var director = VideoDirector()
    @State private var state: UIState = .ready
    @State private var transcript: TranscriptCardData? = nil
    @State private var decision:   DecisionCardData?   = nil
    @State private var gemma:      GemmaCardData?      = nil
    @State private var currentProduct: Product? = nil

    // Runtime backend-host override. Long-press the GEMMA card or the status
    // pill to open the sheet; saves to UserDefaults via GemmaClient. Lets us
    // chase a new WiFi IP without a 90s Swift rebuild at venue.
    @State private var showingHostSheet = false
    @State private var hostInput: String = ""

    // Rolling history of the last few transcripts — so during stage rehearsal
    // the operator can see what whisper has been hearing across reps, not just
    // the active one. Capped at 5. Tap the header to clear.
    @State private var transcriptHistory: [TranscriptCardData] = []

    // Phase 0.2: "+ Film product" flow — camera shim + upload to /api/sell-video.
    // Gated by FeatureFlags.sellerMode (default false — off for stage).
    @State private var showingCapture = false
    @State private var uploadState: SellerCaptureUploadState = .idle

    enum UIState: Equatable {
        case loading
        case ready
        case recording
        case transcribing
        case responding
        case error(String)
    }

    var body: some View {
        ZStack {
            Color(red: 0.035, green: 0.035, blue: 0.047).ignoresSafeArea()

            VStack(spacing: 0) {
                avatarSection
                    .frame(height: UIScreen.main.bounds.height * 0.42)

                Spacer(minLength: 12)

                statusPill
                    .padding(.bottom, 12)

                ScrollView {
                    VStack(spacing: 10) {
                        if !transcriptHistory.isEmpty { transcriptHistorySection }
                        if let t = transcript { transcriptCard(t) }
                        if let g = gemma      { gemmaCard(g) }
                        if let d = decision   { decisionCard(d) }
                    }
                    .padding(.horizontal, 16)
                }
                .frame(maxHeight: .infinity)

                if FeatureFlags.sellerMode {
                    sellerCaptureSection
                        .padding(.horizontal, 16)
                        .padding(.bottom, 8)
                }

                pushToTalkButton
                    .padding(.horizontal, 16)
                    .padding(.bottom, 24)
            }

            if state == .loading { splash }
        }
        .sheet(isPresented: $showingCapture) {
            SellerCaptureShim { url in
                showingCapture = false
                Task { await uploadCapturedVideo(url) }
            }
            .ignoresSafeArea()
        }
        .alert("Backend host", isPresented: $showingHostSheet) {
            TextField("e.g. 192.168.1.42", text: $hostInput)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .keyboardType(.URL)
            Button("Save") {
                GemmaClient.setBackendHost(hostInput)
            }
            Button("Reset to default", role: .destructive) {
                GemmaClient.setBackendHost(nil)
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            let resolved = GemmaClient.backendHost ?? "(none)"
            let source = GemmaClient.hasUserDefaultsOverride ? "runtime override"
                       : "Info.plist / default"
            Text("Current: \(resolved)  ·  \(source)\nFind your Mac IP: ipconfig getifaddr en0")
        }
        .task { await bootstrap() }
    }

    /// Opens the host-override alert. Seeds the text field with the currently
    /// resolved host so edits are a small delta, not a from-scratch type-in.
    private func openHostSheet() {
        hostInput = GemmaClient.backendHost ?? ""
        showingHostSheet = true
    }

    // MARK: - Sections

    private var avatarSection: some View {
        ZStack {
            AvatarVideoView(player: director.player)
                .ignoresSafeArea(edges: .top)
                .opacity(director.scene == .blocked ? 0.25 : 1.0)

            if case .thinking = director.scene {
                VStack(spacing: 8) {
                    ProgressView().controlSize(.large).tint(.white)
                    Text("Router: would escalate to cloud")
                        .font(.system(size: 14, weight: .medium, design: .monospaced))
                        .foregroundColor(.white.opacity(0.9))
                }
                .padding(20)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
            }

            if case .blocked = director.scene {
                Text("Blocked (spam)")
                    .font(.system(size: 16, weight: .bold, design: .monospaced))
                    .foregroundColor(.gray)
                    .padding(16)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
            }

            VStack {
                HStack {
                    Text("AIRPLANE MODE OK")
                        .font(.system(size: 9, weight: .heavy, design: .monospaced))
                        .tracking(1.5)
                        .foregroundColor(.green.opacity(0.85))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.green.opacity(0.15))
                        .overlay(
                            RoundedRectangle(cornerRadius: 999)
                                .stroke(Color.green.opacity(0.35), lineWidth: 1)
                        )
                        .clipShape(RoundedRectangle(cornerRadius: 999))
                    Spacer()
                }
                Spacer()
            }
            .padding(12)
        }
    }

    private var statusPill: some View {
        let (label, color): (String, Color) = {
            switch state {
            case .loading:      return ("LOADING...",   .gray)
            case .ready:        return ("READY",        .green)
            case .recording:    return ("LISTENING",    .red)
            case .transcribing: return ("WHISPER...",   .blue)
            case .responding:   return ("RESPONDING",   .green)
            case .error(let m): return ("ERROR: \(m)",  .red)
            }
        }()
        return HStack(spacing: 6) {
            Circle().fill(color).frame(width: 6, height: 6)
            Text(label)
                .font(.system(size: 11, weight: .heavy, design: .monospaced))
                .tracking(1.5)
                .foregroundColor(color)
        }
        .padding(.horizontal, 12).padding(.vertical, 6)
        .background(color.opacity(0.15))
        .overlay(RoundedRectangle(cornerRadius: 999).stroke(color.opacity(0.4), lineWidth: 1))
        .clipShape(RoundedRectangle(cornerRadius: 999))
        // Long-press the pill to change the Mac backend host at runtime —
        // fallback affordance for before the first transcript, when the GEMMA
        // card isn't visible yet.
        .onLongPressGesture(minimumDuration: 0.6) { openHostSheet() }
    }

    // MARK: - Cards

    struct TranscriptCardData { let text: String; let ms: Int }
    struct DecisionCardData   { let tool: Tool; let reason: String; let ms: Int; let costSaved: Double }
    enum   GemmaCardData {
        case pending                                 // request in flight
        case ok(label: String, ms: Int, source: String)
        case offline(reason: String)                 // host unset, network down, etc.
    }

    private func transcriptCard(_ d: TranscriptCardData) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Text("WHISPER").font(.system(size: 9, weight: .heavy, design: .monospaced))
                .foregroundColor(.white.opacity(0.45)).tracking(1.2).frame(width: 60, alignment: .leading)
            Text("\"\(d.text)\"").font(.system(size: 14)).foregroundColor(.white).frame(maxWidth: .infinity, alignment: .leading)
            Text("\(d.ms)ms").font(.system(size: 10, design: .monospaced)).foregroundColor(.white.opacity(0.6))
        }
        .padding(12).background(Color.white.opacity(0.04)).cornerRadius(8)
    }

    // Rehearsal affordance: shows the last 5 transcripts above the active one
    // with dimmed styling so it's clear which one is current. Tap the header
    // to clear — useful between reps if the history gets cluttered.
    private var transcriptHistorySection: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("HISTORY")
                    .font(.system(size: 9, weight: .heavy, design: .monospaced))
                    .foregroundColor(.white.opacity(0.35))
                    .tracking(1.2)
                Spacer()
                Text("tap to clear")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundColor(.white.opacity(0.25))
            }
            .padding(.horizontal, 4)
            .contentShape(Rectangle())
            .onTapGesture { transcriptHistory.removeAll() }

            ForEach(Array(transcriptHistory.enumerated()), id: \.offset) { _, d in
                HStack(alignment: .top, spacing: 10) {
                    Text("·")
                        .font(.system(size: 11, weight: .heavy, design: .monospaced))
                        .foregroundColor(.white.opacity(0.25))
                        .frame(width: 60, alignment: .leading)
                    Text("\"\(d.text)\"")
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(.white.opacity(0.45))
                        .frame(maxWidth: .infinity, alignment: .leading)
                    Text("\(d.ms)ms")
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundColor(.white.opacity(0.3))
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 4)
            }
        }
        .padding(.vertical, 4)
        .background(Color.white.opacity(0.015))
        .cornerRadius(6)
    }

    private func gemmaCard(_ d: GemmaCardData) -> some View {
        // Magenta to distinguish from WHISPER (gray) and ROUTER (color-coded).
        let accent = Color(red: 0.851, green: 0.275, blue: 0.937)
        return HStack(alignment: .top, spacing: 10) {
            VStack(alignment: .leading, spacing: 1) {
                Text("GEMMA 4").font(.system(size: 9, weight: .heavy, design: .monospaced))
                    .foregroundColor(accent.opacity(0.85)).tracking(1.2)
                Text("(Mac)").font(.system(size: 8, design: .monospaced))
                    .foregroundColor(.white.opacity(0.4))
            }
            .frame(width: 60, alignment: .leading)
            switch d {
            case .pending:
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small).tint(accent)
                    Text("classifying on Mac…").font(.system(size: 13))
                        .foregroundColor(.white.opacity(0.65))
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            case .ok(let label, let ms, let source):
                Text(label).font(.system(size: 14, weight: .bold, design: .monospaced))
                    .foregroundColor(accent)
                    .frame(maxWidth: .infinity, alignment: .leading)
                Text("\(source) · \(ms)ms").font(.system(size: 10, design: .monospaced))
                    .foregroundColor(.white.opacity(0.5))
            case .offline(let reason):
                Text(reason).font(.system(size: 12)).italic()
                    .foregroundColor(.white.opacity(0.4))
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(12)
        .background(accent.opacity(0.10))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(accent.opacity(0.3), lineWidth: 1))
        .cornerRadius(8)
        // Long-press the GEMMA card to change the Mac backend host. Primary
        // affordance per the pitch runbook; the status pill has the same.
        .onLongPressGesture(minimumDuration: 0.6) { openHostSheet() }
    }

    private func decisionCard(_ d: DecisionCardData) -> some View {
        let color = toolColor(d.tool)
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("ROUTER").font(.system(size: 9, weight: .heavy, design: .monospaced))
                    .foregroundColor(color.opacity(0.85)).tracking(1.2)
                Spacer()
                Text("\(d.ms)ms").font(.system(size: 10, design: .monospaced)).foregroundColor(.white.opacity(0.6))
                if d.costSaved > 0 {
                    Text("-$\(String(format: "%.5f", d.costSaved))")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(color.opacity(0.9))
                }
            }
            Text(d.tool.rawValue).font(.system(size: 16, weight: .bold, design: .monospaced)).foregroundColor(color)
            Text(d.reason).font(.system(size: 12)).foregroundColor(.white.opacity(0.65))
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(color.opacity(0.12))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(color.opacity(0.4), lineWidth: 1))
        .cornerRadius(8)
    }

    private func toolColor(_ t: Tool) -> Color {
        switch t {
        case .respondLocally:  return Color(red: 0.133, green: 0.773, blue: 0.369)
        case .playCannedClip:  return Color(red: 0.486, green: 0.227, blue: 0.929)
        case .blockComment:    return Color(red: 0.631, green: 0.631, blue: 0.678)
        case .escalateToCloud: return Color(red: 0.961, green: 0.620, blue: 0.043)
        }
    }

    // MARK: - Seller capture (Phase 0.2)

    private var sellerCaptureSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button {
                showingCapture = true
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: "video.fill")
                        .font(.system(size: 13, weight: .semibold))
                    Text("+ FILM PRODUCT")
                        .font(.system(size: 13, weight: .heavy, design: .monospaced))
                        .tracking(1.2)
                }
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.white.opacity(0.25), lineWidth: 1)
                )
            }
            .disabled(uploadState == .uploading)

            if case .uploading = uploadState {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small).tint(.white)
                    Text("uploading to Mac…")
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(.white.opacity(0.7))
                }
                .padding(.horizontal, 12)
            } else if case .success(let id) = uploadState {
                Text("uploaded · \(id.prefix(8))")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(.green.opacity(0.85))
                    .padding(.horizontal, 12)
            } else if case .failed(let msg) = uploadState {
                Text(msg)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(.red.opacity(0.85))
                    .lineLimit(2)
                    .padding(.horizontal, 12)
            }
        }
    }

    private func uploadCapturedVideo(_ url: URL) async {
        await MainActor.run { uploadState = .uploading }
        do {
            let result = try await SellerCaptureUploader.upload(videoURL: url)
            let requestID = (result["request_id"] as? String)
                ?? (result["id"] as? String)
                ?? "ok"
            await MainActor.run { uploadState = .success(requestID: requestID) }
        } catch {
            await MainActor.run {
                uploadState = .failed(message: error.localizedDescription)
            }
        }
    }

    // MARK: - Push-to-talk

    private var pushToTalkButton: some View {
        let isActive = state == .recording
        let label = isActive ? "RELEASE TO SEND" : "HOLD TO SPEAK"
        return RoundedRectangle(cornerRadius: 20)
            .fill(isActive ? Color.red : Color(red: 0.486, green: 0.227, blue: 0.929))
            .frame(height: 72)
            .overlay(
                Text(label)
                    .font(.system(size: 18, weight: .heavy, design: .monospaced))
                    .tracking(1.5)
                    .foregroundColor(.white)
            )
            .scaleEffect(isActive ? 0.97 : 1.0)
            .animation(.easeInOut(duration: 0.12), value: isActive)
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in
                        if state == .ready { Task { await beginRecording() } }
                    }
                    .onEnded { _ in
                        if state == .recording { Task { await endRecording() } }
                    }
            )
            .disabled(!(state == .ready || state == .recording))
    }

    // MARK: - Flow

    private func bootstrap() async {
        state = .loading
        currentProduct = ProductLoader.loadActive()

        guard let whisperPath = Bundle.main.path(forResource: "whisper-base", ofType: nil, inDirectory: "Models") else {
            state = .error("whisper model not bundled")
            return
        }

        _ = await recorder.requestPermission()
        try? recorder.configureSession()
        await cactus.bootstrap(whisperPath: whisperPath)
        if case .failed(let m) = cactus.status {
            state = .error(m); return
        }
        state = .ready
    }

    private func beginRecording() async {
        director.player.pause()
        try? await Task.sleep(nanoseconds: 50_000_000)
        do {
            try recorder.start()
            // Preserve the just-shown transcript in the rolling history before
            // wiping so the operator can scan what whisper heard across reps.
            if let prev = transcript {
                transcriptHistory = Array(([prev] + transcriptHistory).prefix(5))
            }
            transcript = nil; decision = nil; gemma = nil
            director.backToIdle()
            state = .recording
        } catch {
            state = .error("mic: \(error.localizedDescription)")
        }
    }

    private func endRecording() async {
        let pcm = recorder.stop()
        state = .transcribing
        do {
            let t = try await cactus.transcribe(pcm: pcm)
            if t.text.isEmpty {
                state = .ready
                return
            }
            transcript = TranscriptCardData(text: t.text, ms: t.latencyMs)

            // Route with a default "question" classify — rule-based router
            // handles URL spam, compliment words+emoji, objection words, and
            // product qa_index keyword matching directly from the comment
            // text. No on-device LLM needed.
            let d = Router.decide(
                comment: t.text,
                classify: ClassifyResult(type: "question", draft: nil),
                product: currentProduct
            )
            decision = DecisionCardData(tool: d.tool, reason: d.reason, ms: d.ms, costSaved: d.costSavedUSD)

            director.dispatch(d)
            state = .responding

            // Fire Gemma classify on the Mac IN PARALLEL — does not block the
            // local-first MP4 dispatch above. The Gemma card animates in 2-4s
            // later as a "Mac just verified this" beat; if the Mac is
            // unreachable the demo still works, the card just shows offline.
            let transcriptText = t.text
            gemma = .pending
            Task {
                let result: GemmaCardData
                do {
                    let g = try await GemmaClient.classify(comment: transcriptText)
                    result = .ok(label: g.label, ms: g.latencyMs, source: g.source)
                } catch GemmaClient.Failure.backendNotConfigured {
                    result = .offline(reason: "set EmpireBackendHost in Info.plist")
                } catch {
                    result = .offline(reason: "Mac unreachable")
                }
                await MainActor.run { self.gemma = result }
            }

            Task {
                try? await Task.sleep(nanoseconds: 6_000_000_000)
                if state == .responding { state = .ready }
            }
        } catch {
            state = .error(error.localizedDescription)
        }
    }

    // MARK: - Splash

    private var splash: some View {
        ZStack {
            Color.black.opacity(0.85).ignoresSafeArea()
            VStack(spacing: 16) {
                ProgressView().controlSize(.large).tint(.white)
                Text("Loading whisper on Cactus…")
                    .font(.system(size: 14, weight: .medium, design: .monospaced))
                    .foregroundColor(.white.opacity(0.85))
                Text("~2s cold boot")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(.white.opacity(0.4))
            }
        }
    }
}
