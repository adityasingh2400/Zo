// ContentView.swift — single-screen demo surface.
//
// Flow:
//   press+hold → record → release → Cactus whisper transcribe →
//   Swift rule-based Router.decide() → VideoDirector plays pre-rendered MP4
//
// No on-device LLM classify. The Swift rule-based router in Router.swift
// handles everything from the transcript text (URL spam cues, compliment
// words+emoji, objection words, product qa_index keyword match). Matches
// the 19 parametrized tests in backend/tests/test_router.py bit-for-bit.
//
// Colors match the dashboard's RoutingPanel:
//   respond_locally    → green  (#22c55e)
//   play_canned_clip   → purple (#7c3aed)
//   block_comment      → gray   (#a1a1aa)
//   escalate_to_cloud  → amber  (#f59e0b)

import SwiftUI
import AVKit

struct ContentView: View {
    @StateObject private var cactus = CactusRunner()
    @StateObject private var recorder = AudioRecorder()
    @StateObject private var director = VideoDirector()
    @State private var state: UIState = .ready
    @State private var transcript: TranscriptCardData? = nil
    @State private var decision:   DecisionCardData?   = nil
    @State private var currentProduct: Product? = nil

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
                        if let t = transcript { transcriptCard(t) }
                        if let d = decision   { decisionCard(d) }
                    }
                    .padding(.horizontal, 16)
                }
                .frame(maxHeight: .infinity)

                pushToTalkButton
                    .padding(.horizontal, 16)
                    .padding(.bottom, 24)
            }

            if state == .loading { splash }
        }
        .task { await bootstrap() }
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
    }

    // MARK: - Cards

    struct TranscriptCardData { let text: String; let ms: Int }
    struct DecisionCardData   { let tool: Tool; let reason: String; let ms: Int; let costSaved: Double }

    private func transcriptCard(_ d: TranscriptCardData) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Text("WHISPER").font(.system(size: 9, weight: .heavy, design: .monospaced))
                .foregroundColor(.white.opacity(0.45)).tracking(1.2).frame(width: 60, alignment: .leading)
            Text("\"\(d.text)\"").font(.system(size: 14)).foregroundColor(.white).frame(maxWidth: .infinity, alignment: .leading)
            Text("\(d.ms)ms").font(.system(size: 10, design: .monospaced)).foregroundColor(.white.opacity(0.6))
        }
        .padding(12).background(Color.white.opacity(0.04)).cornerRadius(8)
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
            transcript = nil; decision = nil
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
