// SellerCaptureView.swift — AVFoundation-backed seller capture screen.
//
// Replaces the Phase 0.2 UIImagePickerController shim. Matches the mockup
// (empire-mobile.jsx EmpireMobileCapture): live camera preview + REC pill +
// close X + Deepgram-style transcription card (post-upload) + "Building your
// avatar" pipeline rail + 3-button shutter cluster.
//
// Flow:
//   onAppear: cameraSession.configure() + socket.start()
//   user taps shutter: startRecording → REC pill pulses
//   user taps shutter again: stopRecording → upload → show PipelineProgressView
//   pipeline "going_live" → auto-dismiss ~2s later, callback onComplete

import SwiftUI
import AVFoundation

struct SellerCaptureView: View {
    @Environment(\.dismiss) private var dismiss
    let onComplete: (String?) -> Void  // callback with the request_id (or nil on cancel)

    @State private var cameraSession = CameraSession()
    @State private var socket = EmpireSocket()
    @State private var flowState: FlowState = .recordingReady
    @State private var timerTick: Int = 0  // seconds since recording started
    @State private var activeRequestID: String?
    @State private var timerTask: Task<Void, Never>?
    @State private var showingHostSheet = false
    @State private var hostInput = ""

    enum FlowState: Equatable {
        case recordingReady
        case recording
        case uploading
        case pipelineActive(requestID: String)
        case complete(requestID: String)
        case failed(String)
    }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            // Live preview — fills the screen.
            CameraPreview(session: cameraSession.captureSession)
                .ignoresSafeArea()
                .overlay(
                    // Product silhouette hint (matches mockup) — soft radial
                    // gradient hinting where to frame the product.
                    Circle()
                        .strokeBorder(Color.white.opacity(0.15), lineWidth: 1)
                        .frame(width: 220, height: 220)
                        .opacity(flowState == .recording ? 0 : 1)
                )

            VStack(spacing: 0) {
                topChrome
                    .padding(.horizontal, 20)
                    .padding(.top, 50)

                Spacer()

                if case .pipelineActive(let rid) = flowState {
                    PipelineProgressView(requestID: rid, socket: socket)
                        .padding(.horizontal, 16)
                        .padding(.bottom, 12)
                } else if case .complete = flowState {
                    completeBanner
                        .padding(.horizontal, 16)
                        .padding(.bottom, 12)
                } else if case .failed(let msg) = flowState {
                    failureBanner(message: msg)
                        .padding(.horizontal, 16)
                        .padding(.bottom, 12)
                }

                shutterCluster
                    .padding(.bottom, 28)
            }
        }
        .task {
            _ = await cameraSession.configure()
            socket.start()
        }
        .onDisappear {
            timerTask?.cancel()
            socket.stop()
            cameraSession.teardown()
        }
        .alert("Backend host", isPresented: $showingHostSheet) {
            TextField("IP or https://...trycloudflare.com", text: $hostInput)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .keyboardType(.URL)
            Button("Save + retry") {
                GemmaClient.setBackendHost(hostInput.trimmingCharacters(in: .whitespacesAndNewlines))
                flowState = .recordingReady
            }
            Button("Reset to default", role: .destructive) {
                GemmaClient.setBackendHost(nil)
                flowState = .recordingReady
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            let resolved = GemmaClient.backendHost ?? "(none)"
            let source = GemmaClient.hasUserDefaultsOverride ? "runtime override"
                       : "Info.plist / default"
            Text("Current: \(resolved)\nSource: \(source)")
        }
    }

    // MARK: - Top chrome (REC pill + close)

    private var topChrome: some View {
        HStack {
            // REC pill — long-press to change backend host (shortcut to the
            // host-override sheet without needing to close the camera).
            HStack(spacing: 6) {
                Circle()
                    .fill(flowState == .recording ? Color.red : Color.white.opacity(0.3))
                    .frame(width: 7, height: 7)
                Text(recLabel)
                    .font(.system(size: 10, weight: .heavy, design: .monospaced))
                    .tracking(0.8)
                    .foregroundColor(.white)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(.ultraThinMaterial.opacity(0.4), in: Capsule())
            .overlay(Capsule().stroke(Color.white.opacity(0.2), lineWidth: 0.5))
            .onLongPressGesture(minimumDuration: 0.6) {
                hostInput = GemmaClient.backendHost ?? ""
                showingHostSheet = true
            }

            Spacer()

            // Close X — dismiss without saving
            Button {
                onComplete(nil)
                dismiss()
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundColor(.white)
                    .frame(width: 36, height: 36)
                    .background(.ultraThinMaterial.opacity(0.4), in: Circle())
                    .overlay(Circle().stroke(Color.white.opacity(0.2), lineWidth: 0.5))
            }
        }
    }

    private var recLabel: String {
        switch flowState {
        case .recording:
            let mm = timerTick / 60
            let ss = timerTick % 60
            return String(format: "REC · %02d:%02d", mm, ss)
        case .recordingReady: return "READY"
        case .uploading:      return "UPLOADING"
        case .pipelineActive: return "PROCESSING"
        case .complete:       return "DONE"
        case .failed:         return "FAILED"
        }
    }

    // MARK: - Banners

    private var completeBanner: some View {
        HStack(spacing: 10) {
            Image(systemName: "checkmark.circle.fill")
                .foregroundColor(.green)
            Text("Going live — your avatar is pitching it")
                .font(.system(size: 13, weight: .semibold))
                .foregroundColor(.white)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial.opacity(0.6), in: RoundedRectangle(cornerRadius: 14))
    }

    private func failureBanner(message: String) -> some View {
        Button {
            // Open the host sheet immediately — no more close-camera dance.
            hostInput = GemmaClient.backendHost ?? ""
            showingHostSheet = true
        } label: {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 10) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.red)
                    Text(message)
                        .font(.system(size: 12))
                        .foregroundColor(.white)
                        .lineLimit(3)
                        .multilineTextAlignment(.leading)
                }
                HStack(spacing: 6) {
                    Text("target")
                        .font(.system(size: 9, weight: .heavy, design: .monospaced))
                        .tracking(1)
                        .foregroundColor(.white.opacity(0.4))
                    Text(GemmaClient.backendHost ?? "?")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(.white.opacity(0.65))
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                HStack {
                    Text("TAP TO CHANGE HOST")
                        .font(.system(size: 10, weight: .heavy, design: .monospaced))
                        .tracking(1.2)
                        .foregroundColor(.blue)
                    Spacer()
                    Image(systemName: "chevron.right")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(.blue)
                }
                .padding(.top, 2)
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.ultraThinMaterial.opacity(0.6), in: RoundedRectangle(cornerRadius: 14))
        }
        .buttonStyle(.plain)
    }

    // MARK: - Shutter cluster

    private var shutterCluster: some View {
        HStack(spacing: 32) {
            // Placeholder for library / last capture thumbnail
            RoundedRectangle(cornerRadius: 10)
                .fill(Color.white.opacity(0.08))
                .frame(width: 44, height: 44)
                .overlay(
                    Image(systemName: "photo.stack")
                        .font(.system(size: 16))
                        .foregroundColor(.white.opacity(0.7))
                )

            // Main shutter
            Button(action: toggleRecording) {
                ZStack {
                    Circle()
                        .stroke(Color.white, lineWidth: 4)
                        .frame(width: 72, height: 72)
                    if flowState == .recording {
                        RoundedRectangle(cornerRadius: 6)
                            .fill(Color.red)
                            .frame(width: 32, height: 32)
                    } else {
                        Circle()
                            .fill(Color.red)
                            .frame(width: 56, height: 56)
                    }
                }
            }
            .disabled(!shutterEnabled)
            .opacity(shutterEnabled ? 1 : 0.5)

            // Flip camera
            Button {
                cameraSession.flipCamera()
            } label: {
                Image(systemName: "arrow.triangle.2.circlepath.camera.fill")
                    .font(.system(size: 18))
                    .foregroundColor(.white)
                    .frame(width: 44, height: 44)
                    .background(Color.white.opacity(0.12), in: Circle())
            }
            .disabled(flowState == .recording)
            .opacity(flowState == .recording ? 0.4 : 1)
        }
    }

    private var shutterEnabled: Bool {
        switch flowState {
        case .recordingReady, .recording: return true
        default: return false
        }
    }

    // MARK: - Actions

    private func toggleRecording() {
        switch flowState {
        case .recordingReady:
            cameraSession.startRecording()
            flowState = .recording
            timerTick = 0
            timerTask = Task {
                while !Task.isCancelled {
                    try? await Task.sleep(nanoseconds: 1_000_000_000)
                    await MainActor.run { timerTick += 1 }
                }
            }
        case .recording:
            Task {
                let url = await cameraSession.stopRecording()
                await MainActor.run {
                    timerTask?.cancel()
                    timerTask = nil
                }
                guard let url = url else {
                    await MainActor.run { flowState = .failed("no recording produced") }
                    return
                }
                await upload(url: url)
            }
        default:
            break  // ignore taps in other states
        }
    }

    private func upload(url: URL) async {
        await MainActor.run { flowState = .uploading }
        do {
            let result = try await SellerCaptureUploader.upload(videoURL: url)
            let requestID = (result["request_id"] as? String) ?? UUID().uuidString
            await MainActor.run {
                activeRequestID = requestID
                flowState = .pipelineActive(requestID: requestID)
            }
            // Watch for going_live to auto-dismiss.
            await watchForCompletion(requestID: requestID)
        } catch {
            await MainActor.run { flowState = .failed(error.localizedDescription) }
        }
    }

    /// Poll the socket's buffered steps until "going_live" lands, then
    /// auto-dismiss after 2s. If we never see going_live within 60s, keep
    /// the progress card up — operator can dismiss manually via the X.
    private func watchForCompletion(requestID: String) async {
        let deadline = Date().addingTimeInterval(60)
        while Date() < deadline {
            let hasGoingLive = socket.steps(for: requestID).contains {
                $0.step == "going_live" && ($0.status == "done" || $0.status == "active")
            }
            if hasGoingLive {
                await MainActor.run { flowState = .complete(requestID: requestID) }
                try? await Task.sleep(nanoseconds: 2_000_000_000)
                await MainActor.run {
                    onComplete(requestID)
                    socket.clearPipelineSteps(requestID: requestID)
                    dismiss()
                }
                return
            }
            try? await Task.sleep(nanoseconds: 500_000_000)
        }
    }
}
