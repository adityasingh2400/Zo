// CameraSession.swift — AVFoundation capture session for the seller camera.
//
// Phase 1 Item 1 (per empire-phase1-phase3.md plan). Replaces the
// UIImagePickerController shim with a proper AVCaptureSession pipeline:
// live preview + controllable movie output + recording delegate.
//
// Key AVFoundation gotchas handled here:
//   - Session configuration must happen off the main queue
//     (session.beginConfiguration / commitConfiguration called via
//     a dedicated serial queue, not MainActor).
//   - AVCaptureMovieFileOutput needs a delegate that outlives the record
//     call. We hold one as a stored property so ARC doesn't collect it.
//   - On teardown, call stopRunning() before nil-ing inputs — otherwise
//     the camera-hot indicator lingers until the next app launch.
//
// Recording writes to a temp file at .documentsDirectory/capture_<uuid>.mov,
// handed back to the caller via the async stopRecording() return value.
// Caller is responsible for uploading + deleting when done.

import AVFoundation
import Foundation
import Observation
import UIKit

@MainActor
@Observable
final class CameraSession: NSObject {
    enum State: Equatable {
        case idle
        case configuring
        case ready
        case recording(startedAt: Date)
        case stopping
        case failed(String)
    }

    private(set) var state: State = .idle

    /// Capture session — exposed so CameraPreview can attach its preview
    /// layer. Mutations happen on sessionQueue, never from the main actor.
    let captureSession = AVCaptureSession()

    /// AVFoundation requires configuration off the main queue. Serial so
    /// begin/commit pairs don't interleave with start/stop recordings.
    private let sessionQueue = DispatchQueue(label: "empire.camera.session",
                                             qos: .userInitiated)
    private let movieOutput = AVCaptureMovieFileOutput()
    private var recordingURL: URL?
    private var recordingContinuation: CheckedContinuation<URL?, Never>?

    // ── Public API ────────────────────────────────────────────────────────

    /// Configures the capture session for 1080p video + audio. Safe to call
    /// multiple times; subsequent calls are no-ops once state == .ready.
    /// Caller should check the returned state to confirm readiness.
    func configure() async -> State {
        guard state == .idle || state.isFailed else { return state }
        state = .configuring

        let authorized = await requestPermissions()
        guard authorized else {
            state = .failed("camera / microphone permission denied")
            return state
        }

        await withCheckedContinuation { [weak self] (cont: CheckedContinuation<Void, Never>) in
            guard let self = self else { cont.resume(); return }
            self.sessionQueue.async { [weak self] in
                guard let self = self else { cont.resume(); return }
                self.captureSession.beginConfiguration()
                // Prefer 1080p; AVCaptureSession.Preset.hd1920x1080 matches
                // the Veo/Wav2Lip render target so we don't upscale in the
                // intake pipeline.
                if self.captureSession.canSetSessionPreset(.hd1920x1080) {
                    self.captureSession.sessionPreset = .hd1920x1080
                } else {
                    self.captureSession.sessionPreset = .high
                }
                self.addVideoInput()
                self.addAudioInput()
                self.addMovieOutput()
                self.captureSession.commitConfiguration()
                self.captureSession.startRunning()
                cont.resume()
            }
        }

        state = .ready
        return state
    }

    /// Starts writing a new .mov to the temp directory. No-op if not ready.
    func startRecording() {
        guard case .ready = state else { return }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("capture_\(UUID().uuidString).mov")
        recordingURL = url
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            self.movieOutput.startRecording(to: url, recordingDelegate: self)
        }
        state = .recording(startedAt: Date())
    }

    /// Stops the current recording and returns the file URL once AVFoundation
    /// flushes the container. Returns nil if not recording or on failure.
    func stopRecording() async -> URL? {
        guard case .recording = state else { return nil }
        state = .stopping
        return await withCheckedContinuation { [weak self] (cont: CheckedContinuation<URL?, Never>) in
            guard let self = self else { cont.resume(returning: nil); return }
            self.recordingContinuation = cont
            self.sessionQueue.async {
                self.movieOutput.stopRecording()
            }
        }
    }

    /// Flip between front and back cameras. Only valid when .ready.
    func flipCamera() {
        guard case .ready = state else { return }
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            self.captureSession.beginConfiguration()
            defer { self.captureSession.commitConfiguration() }
            guard let oldInput = self.captureSession.inputs
                .first(where: { ($0 as? AVCaptureDeviceInput)?.device.hasMediaType(.video) == true })
                as? AVCaptureDeviceInput else { return }
            let newPosition: AVCaptureDevice.Position =
                oldInput.device.position == .back ? .front : .back
            self.captureSession.removeInput(oldInput)
            if let newDevice = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                       for: .video,
                                                       position: newPosition),
               let newInput = try? AVCaptureDeviceInput(device: newDevice),
               self.captureSession.canAddInput(newInput) {
                self.captureSession.addInput(newInput)
            } else {
                // Put the old input back if we couldn't swap.
                self.captureSession.addInput(oldInput)
            }
        }
    }

    /// Tears down the session. Call from view's onDisappear to release the
    /// camera cleanly so the green hot-camera indicator doesn't stick.
    func teardown() {
        sessionQueue.async { [weak self] in
            self?.captureSession.stopRunning()
        }
        state = .idle
    }

    // ── Setup helpers ─────────────────────────────────────────────────────

    private func requestPermissions() async -> Bool {
        let cameraOK = await checkAuthorization(for: .video)
        let micOK = await checkAuthorization(for: .audio)
        return cameraOK && micOK
    }

    private func checkAuthorization(for mediaType: AVMediaType) async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: mediaType) {
        case .authorized: return true
        case .notDetermined:
            return await AVCaptureDevice.requestAccess(for: mediaType)
        default: return false
        }
    }

    /// Must run on sessionQueue. Adds the back camera as the video input.
    private func addVideoInput() {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                   for: .video,
                                                   position: .back),
              let input = try? AVCaptureDeviceInput(device: device),
              captureSession.canAddInput(input) else { return }
        captureSession.addInput(input)
    }

    /// Must run on sessionQueue. Adds the default mic as the audio input.
    private func addAudioInput() {
        guard let device = AVCaptureDevice.default(for: .audio),
              let input = try? AVCaptureDeviceInput(device: device),
              captureSession.canAddInput(input) else { return }
        captureSession.addInput(input)
    }

    /// Must run on sessionQueue. Attaches the movie file output.
    private func addMovieOutput() {
        if captureSession.canAddOutput(movieOutput) {
            captureSession.addOutput(movieOutput)
            // 15s ceiling matches SellerCaptureUploader + PDF's "10-second
            // phone clip" spec — anything longer is a recording error.
            movieOutput.maxRecordedDuration = CMTime(seconds: 15,
                                                     preferredTimescale: 600)
        }
    }
}

// ── AVCaptureFileOutputRecordingDelegate ─────────────────────────────────

extension CameraSession: AVCaptureFileOutputRecordingDelegate {
    nonisolated func fileOutput(_ output: AVCaptureFileOutput,
                                didFinishRecordingTo outputFileURL: URL,
                                from connections: [AVCaptureConnection],
                                error: Error?) {
        Task { @MainActor [weak self] in
            guard let self = self else { return }
            let resultURL: URL?
            if let error = error as NSError? {
                // AVErrorMaximumDurationReached isn't really a failure —
                // it's the 15s cap firing. Still return the file URL.
                if error.code == AVError.Code.maximumDurationReached.rawValue {
                    resultURL = outputFileURL
                } else {
                    resultURL = nil
                    self.state = .failed("recording failed: \(error.localizedDescription)")
                }
            } else {
                resultURL = outputFileURL
            }
            self.recordingContinuation?.resume(returning: resultURL)
            self.recordingContinuation = nil
            if case .failed = self.state {
                // Keep the failed state; caller shows it.
            } else {
                self.state = .ready
            }
        }
    }
}

private extension CameraSession.State {
    var isFailed: Bool {
        if case .failed = self { return true }
        return false
    }
}
