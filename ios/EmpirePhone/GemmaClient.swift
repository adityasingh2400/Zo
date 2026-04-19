// GemmaClient.swift — call Gemma 4 (running via Cactus on the Mac backend)
// from the iPhone over the LAN.
//
// Why this exists: Gemma 4 doesn't fit on iPhone (E2B / E4B are 4-6+ GB
// multimodal; CactusRunner.swift documents the probes). But the Mac backend
// already loads Gemma 4 via Cactus and exposes /api/classify_comment. So the
// phone POSTs the whisper transcript to the Mac and renders Gemma's classify
// output as a third card under WHISPER + ROUTER. Honest framing: the phone
// is the seller's surface; the laptop is the on-device inference machine;
// both are local — no cloud.
//
// Backend host: read from Info.plist key `EmpireBackendHost` (e.g.
// "192.168.0.117"). Find your Mac's LAN IP with: `ipconfig getifaddr en0`.
// Leave the key empty (or unset) to disable the Gemma card entirely.
//
// Latency: Mac runs Gemma 4 on CPU prefill (no NPU mlpackage today),
// so classify is 2-4s. We fire this AFTER the router has already
// dispatched, so the on-phone demo stays fast (<1s) and the Gemma
// card animates in late as a "Mac just verified this" beat.

import Foundation

struct GemmaClassify {
    let label: String           // "question" | "compliment" | "objection" | "spam"
    let latencyMs: Int          // backend-reported classify_ms
    let source: String          // "cactus" | "ollama" | "fallback" — which engine the Mac used
    let draftResponse: String?  // optional Gemma-suggested response text
}

enum GemmaClient {

    enum Failure: Error {
        case backendNotConfigured
        case http(Int)
        case decode
        case network(Error)
    }

    /// Mac LAN IP fallback. Xcode 14+ projects with GENERATE_INFOPLIST_FILE=YES
    /// (the modern default) ignore the source Info.plist and synthesize one
    /// from build settings, so editing the source file doesn't propagate.
    /// This compile-time constant is the reliable path for a hackathon demo.
    /// Find your Mac's IP via: `ipconfig getifaddr en0`. Set to "" to disable
    /// the Gemma card entirely.
    static let DEFAULT_BACKEND_HOST = "172.20.10.2"

    /// UserDefaults key for a runtime-settable backend host. Long-press the
    /// GEMMA card (or status pill) on the iPhone to set this without rebuilding
    /// — every new venue means a new WiFi IP, and a Swift rebuild takes ~90s.
    static let userDefaultsKey = "EmpireBackendHost"

    /// Resolve the backend host. Priority:
    ///   1. UserDefaults (runtime override, set via the long-press sheet)
    ///   2. Info.plist `EmpireBackendHost` (per-build override at config time)
    ///   3. Compile-time `DEFAULT_BACKEND_HOST` constant
    static var backendHost: String? {
        let fromDefaults = (UserDefaults.standard.string(forKey: userDefaultsKey) ?? "")
            .trimmingCharacters(in: .whitespaces)
        if !fromDefaults.isEmpty { return fromDefaults }
        let fromPlist = ((Bundle.main.object(forInfoDictionaryKey: "EmpireBackendHost") as? String) ?? "")
            .trimmingCharacters(in: .whitespaces)
        if !fromPlist.isEmpty { return fromPlist }
        let trimmedDefault = DEFAULT_BACKEND_HOST.trimmingCharacters(in: .whitespaces)
        return trimmedDefault.isEmpty ? nil : trimmedDefault
    }

    /// Whether a UserDefaults override is currently active. Useful for the UI
    /// to show "using runtime host" vs "using Info.plist/default".
    static var hasUserDefaultsOverride: Bool {
        !((UserDefaults.standard.string(forKey: userDefaultsKey) ?? "")
            .trimmingCharacters(in: .whitespaces).isEmpty)
    }

    /// Set or clear the runtime host override. Pass nil or empty to clear,
    /// which falls back to Info.plist → compile-time default.
    static func setBackendHost(_ host: String?) {
        let trimmed = (host ?? "").trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty {
            UserDefaults.standard.removeObject(forKey: userDefaultsKey)
        } else {
            UserDefaults.standard.set(trimmed, forKey: userDefaultsKey)
        }
    }

    /// POST the transcript to the Mac's /api/classify_comment. Sub-5s typical
    /// (Cactus Gemma 4 on CPU prefill is 2-4s; add ~50ms LAN RTT). Returns
    /// nil if the host isn't configured (caller hides the Gemma card).
    static func classify(comment: String, port: Int = 8000) async throws -> GemmaClassify {
        guard let host = backendHost else { throw Failure.backendNotConfigured }

        var components = URLComponents()
        components.scheme = "http"
        components.host = host
        components.port = port
        components.path = "/api/classify_comment"
        guard let url = components.url else { throw Failure.backendNotConfigured }

        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.timeoutInterval = 8
        req.setValue("application/x-www-form-urlencoded; charset=utf-8",
                     forHTTPHeaderField: "Content-Type")
        // Form-encode `comment=<text>`. Keep it simple — backend accepts Form(comment: str).
        let body = "comment=" + (comment.addingPercentEncoding(
            withAllowedCharacters: .alphanumerics) ?? "")
        req.httpBody = body.data(using: .utf8)

        let (data, response): (Data, URLResponse)
        do {
            (data, response) = try await URLSession.shared.data(for: req)
        } catch {
            throw Failure.network(error)
        }

        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw Failure.http(code)
        }

        guard let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            throw Failure.decode
        }
        let label = (obj["label"] as? String) ?? "question"
        let ms = Int((obj["classify_ms"] as? NSNumber)?.intValue
                     ?? (obj["classify_ms"] as? Int) ?? 0)
        let source = (obj["source"] as? String) ?? "unknown"
        let draft = obj["draft_response"] as? String
        return GemmaClassify(label: label, latencyMs: ms, source: source, draftResponse: draft)
    }
}
