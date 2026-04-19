// AudioRecorder.swift — push-to-talk mic → 16 kHz mono PCM16 bytes.
//
// Switched from AVAudioEngine + tap to AVAudioRecorder (file-based).
// The engine pattern was unreliable on iOS 26 — input tap would receive
// only ~0.17s of audio before stopping despite the user holding the button
// for several seconds. AVAudioRecorder writes a WAV file directly and we
// strip the 44-byte header to feed PCM bytes to whisper.

import Foundation
import AVFoundation
import Combine

@MainActor
final class AudioRecorder: NSObject, ObservableObject {
    enum RecorderError: Error {
        case permissionDenied
        case sessionFailed(Error)
        case recorderInitFailed(Error)
        case recorderStartFailed
        case fileReadFailed(Error)
    }

    @Published private(set) var isRecording: Bool = false

    private var recorder: AVAudioRecorder?
    private var fileURL: URL?

    // MARK: - Permission

    func requestPermission() async -> Bool {
        if #available(iOS 17, *) {
            return await AVAudioApplication.requestRecordPermission()
        } else {
            return await withCheckedContinuation { cont in
                AVAudioSession.sharedInstance().requestRecordPermission { granted in
                    cont.resume(returning: granted)
                }
            }
        }
    }

    /// Configure once at app launch after permission granted.
    func configureSession() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord,
                                mode: .default,
                                options: [.defaultToSpeaker, .allowBluetooth])
        try session.setActive(true, options: [])
    }

    // MARK: - Start / stop

    func start() throws {
        guard !isRecording else { return }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("recording-\(UUID().uuidString).wav")
        fileURL = url

        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 16_000,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
        ]

        do {
            let r = try AVAudioRecorder(url: url, settings: settings)
            r.prepareToRecord()
            guard r.record() else {
                throw RecorderError.recorderStartFailed
            }
            recorder = r
            isRecording = true
            print("[AudioRecorder] start() recording to \(url.lastPathComponent)")
        } catch {
            throw RecorderError.recorderInitFailed(error)
        }
    }

    func stop() -> Data {
        guard isRecording, let r = recorder, let url = fileURL else { return Data() }
        r.stop()
        isRecording = false
        recorder = nil

        do {
            let wav = try Data(contentsOf: url)
            // Strip standard 44-byte WAV header → raw PCM16 LE mono.
            let pcm = wav.count > 44 ? wav.subdata(in: 44..<wav.count) : Data()
            try? FileManager.default.removeItem(at: url)
            print("[AudioRecorder] stop() pcm bytes: \(pcm.count) (file was \(wav.count))")
            return pcm
        } catch {
            print("[AudioRecorder] stop() file read failed: \(error)")
            try? FileManager.default.removeItem(at: url)
            return Data()
        }
    }
}
