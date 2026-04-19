// CactusRunner.swift — on-device whisper via Cactus.
//
// SCOPE (honest):
//   The phone runs Cactus whisper-tiny for voice → text. The rule-based
//   router in Router.swift takes it from there — no on-device LLM classify.
//
// Why no Gemma on phone:
//   We probed every Cactus-supported on-device LLM for classify accuracy +
//   mobile feasibility. Findings:
//     - Gemma 4 E2B:   6.4 GB (multimodal, can't strip), Apple NPU — too
//                      big to bundle; 4 GB min even stripped would still
//                      require lazy-load rearchitecture in Cactus.
//     - Gemma 4 E4B:   Same multimodal size class; 3-31 s latency for tool
//                      calling, 1/4 accuracy (Mac probe).
//     - Gemma 3n-E2B:  4.5 GB text-only, CPU-only on phone (no NPU), 5-12 s
//                      per classify on A15.
//     - functiongemma-270m: 267 MB, refuses every prompt (0/4 accuracy).
//
// None of those produce a demo-worthy on-phone LLM loop. Cactus whisper
// IS demo-worthy: 120 MB, Apple NPU, 244 ms observed on Mac.
//
// Serial queue wraps the blocking cactusTranscribe. The Cactus C library
// is not re-entrant on one handle — same constraint as the Mac side.

import Foundation
import Combine

@MainActor
final class CactusRunner: ObservableObject {

    enum RunnerError: Error {
        case modelNotLoaded
        case transcriptionFailed(String)
    }

    @Published private(set) var status: Status = .cold

    enum Status: Equatable {
        case cold
        case loading
        case ready
        case failed(String)
    }

    private var whisperHandle: CactusModelT?
    private let whisperQueue = DispatchQueue(label: "cactus.whisper", qos: .userInitiated)

    // MARK: - Lifecycle

    /// Load whisper-tiny into memory. ~1-2 s cold on A15. Call once on launch.
    func bootstrap(whisperPath: String) async {
        if case .ready = status { return }
        status = .loading
        do {
            whisperHandle = try await runOn(whisperQueue) {
                try cactusInit(whisperPath, nil, false)
            }
            status = .ready
        } catch {
            status = .failed(error.localizedDescription)
        }
    }

    func shutdown() {
        if let h = whisperHandle { cactusDestroy(h); whisperHandle = nil }
        status = .cold
    }

    // MARK: - Transcribe

    struct TranscribeResult {
        let text: String
        let latencyMs: Int
    }

    /// Transcribe 16 kHz mono PCM16 bytes captured by AudioRecorder.
    func transcribe(pcm: Data) async throws -> TranscribeResult {
        guard let handle = whisperHandle else { throw RunnerError.modelNotLoaded }
        let t0 = DispatchTime.now()
        // Streaming API forces language=en. The non-stream cactus_transcribe
        // ignores the language option (only temperature/top_p/top_k/max_tokens
        // are parsed from its options JSON), which made whisper-tiny auto-detect
        // iPhone-mic audio as Norwegian Nynorsk and emit garbled <|nn|> tokens.
        let optionsJson = "{\"language\":\"en\"}"
        let raw = try await runOn(whisperQueue) {
            let stream = try cactusStreamTranscribeStart(handle, optionsJson)
            do {
                _ = try cactusStreamTranscribeProcess(stream, pcm)
                return try cactusStreamTranscribeStop(stream)
            } catch {
                _ = try? cactusStreamTranscribeStop(stream)
                throw error
            }
        }
        let ms = Int(Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1_000_000)

        // Streaming API: {"success": bool, "confirmed": "..."}.
        // Non-stream API: {"text": "...", "segments": [...]} — kept as fallback.
        if let data = raw.data(using: .utf8),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let text = (obj["confirmed"] as? String) ?? (obj["text"] as? String)
        {
            return TranscribeResult(text: text.trimmingCharacters(in: .whitespaces),
                                    latencyMs: ms)
        }
        return TranscribeResult(text: raw.trimmingCharacters(in: .whitespaces), latencyMs: ms)
    }

    // MARK: - Helpers

    private func runOn<T>(_ queue: DispatchQueue, _ work: @escaping () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { cont in
            queue.async {
                do {
                    let value = try work()
                    cont.resume(returning: value)
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }
}
