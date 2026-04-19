// SellerCaptureShim.swift — upload helper for SellerCaptureView.
//
// Originally Phase 0.2 wrapped UIImagePickerController; that's been replaced
// by Phase 1.1's SellerCaptureView (real AVFoundation pipeline). This file
// kept the `SellerCaptureUploader` enum + multipart-body builder since both
// are used by the new view.
//
// File name retained for git history continuity — a future cleanup could
// rename to Uploader.swift.

import Foundation

// ── Upload ──────────────────────────────────────────────────────────────

/// Upload state for the capture shim. Drives the inline pill rendered
/// below the "+ Film product" button.
enum SellerCaptureUploadState: Equatable {
    case idle
    case uploading
    case success(requestID: String)
    case failed(message: String)
}

enum SellerCaptureUploader {
    /// POST the recorded clip to /api/sell-video on the configured Mac
    /// backend. Returns the parsed { request_id, ... } JSON on success.
    ///
    /// Requires GemmaClient.backendHost to be set — we piggyback on the
    /// same runtime-overridable host used for the Gemma card, so
    /// long-press → host update covers both paths.
    static func upload(
        videoURL: URL,
        voiceText: String = "",
        port: Int = 8000
    ) async throws -> [String: Any] {
        // Route through GemmaClient.backendBaseURL so LAN hostnames AND full
        // tunnel URLs (cloudflared / ngrok / Tailscale) both work without
        // any caller changes. `port` is honored for LAN mode; tunnels embed
        // their own port via the URL scheme.
        guard let base = GemmaClient.backendBaseURL else {
            throw NSError(
                domain: "SellerCaptureUploader",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "backend host not configured — long-press the status pill"]
            )
        }
        let url = base.appendingPathComponent("api/sell-video")

        let boundary = "Boundary-\(UUID().uuidString)"
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.timeoutInterval = 120  // generous — intake + render can run long
        req.setValue("multipart/form-data; boundary=\(boundary)",
                     forHTTPHeaderField: "Content-Type")

        let body = try buildMultipartBody(
            videoURL: videoURL,
            voiceText: voiceText,
            boundary: boundary
        )
        req.httpBody = body

        let (data, response) = try await URLSession.shared.data(for: req)
        guard let http = response as? HTTPURLResponse,
              (200..<300).contains(http.statusCode) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            let preview = String(data: data, encoding: .utf8)?.prefix(200) ?? ""
            throw NSError(
                domain: "SellerCaptureUploader",
                code: code,
                userInfo: [NSLocalizedDescriptionKey: "HTTP \(code): \(preview)"]
            )
        }

        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NSError(domain: "SellerCaptureUploader", code: -3,
                          userInfo: [NSLocalizedDescriptionKey: "non-JSON response"])
        }
        return obj
    }

    private static func buildMultipartBody(
        videoURL: URL,
        voiceText: String,
        boundary: String
    ) throws -> Data {
        var body = Data()
        let crlf = "\r\n"
        let videoData = try Data(contentsOf: videoURL)
        let fileName = videoURL.lastPathComponent

        // voice_text form field
        body.append("--\(boundary)\(crlf)".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"voice_text\"\(crlf)\(crlf)"
            .data(using: .utf8)!)
        body.append("\(voiceText)\(crlf)".data(using: .utf8)!)

        // file form field
        body.append("--\(boundary)\(crlf)".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"\(fileName)\"\(crlf)"
            .data(using: .utf8)!)
        body.append("Content-Type: video/quicktime\(crlf)\(crlf)".data(using: .utf8)!)
        body.append(videoData)
        body.append(crlf.data(using: .utf8)!)

        body.append("--\(boundary)--\(crlf)".data(using: .utf8)!)
        return body
    }
}
