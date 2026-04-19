// SellerCaptureShim.swift — Phase 0.2 camera shim.
//
// Wraps the native iOS camera (UIImagePickerController) so we can record a
// short product clip and POST it to /api/sell-video WITHOUT managing an
// AVFoundation session. The real AVFoundation pipeline (live preview, REC
// pill, shutter, pipeline progress card) lives in Phase 1.1's
// SellerCaptureView.swift — not shipped yet.
//
// Why a shim: the pitch demo is stage-critical; we want the "phone → film
// → avatar sells it" beat available without rebuilding the iPhone app's
// recording infrastructure from scratch. UIImagePickerController is
// Apple-stock, handles permissions, and hands back a movie file at a
// file:// URL. Good enough for the pitch, cheap to ship, risk-contained.
//
// Gated behind FeatureFlags.sellerMode (default false).

import SwiftUI
import UIKit

struct SellerCaptureShim: UIViewControllerRepresentable {
    /// Called with the recorded video's file URL on successful capture.
    var onVideoPicked: (URL) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(onVideoPicked: onVideoPicked)
    }

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        // Camera source; movie mode; cap at 15s to match the pitch spec
        // ("10-second phone clip"). If the device has no camera (simulator),
        // UIImagePickerController.isSourceTypeAvailable(.camera) returns
        // false and we fall back to photoLibrary so the sheet still opens.
        if UIImagePickerController.isSourceTypeAvailable(.camera) {
            picker.sourceType = .camera
            picker.cameraCaptureMode = .video
        } else {
            picker.sourceType = .photoLibrary
        }
        picker.mediaTypes = ["public.movie"]
        picker.videoMaximumDuration = 15
        picker.allowsEditing = false
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_: UIImagePickerController, context _: Context) {}

    final class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let onVideoPicked: (URL) -> Void

        init(onVideoPicked: @escaping (URL) -> Void) {
            self.onVideoPicked = onVideoPicked
        }

        func imagePickerController(
            _ picker: UIImagePickerController,
            didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]
        ) {
            picker.dismiss(animated: true)
            if let url = info[.mediaURL] as? URL {
                onVideoPicked(url)
            }
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            picker.dismiss(animated: true)
        }
    }
}

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
        guard let host = GemmaClient.backendHost else {
            throw NSError(
                domain: "SellerCaptureUploader",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "backend host not configured — long-press the status pill"]
            )
        }

        var components = URLComponents()
        components.scheme = "http"
        components.host = host
        components.port = port
        components.path = "/api/sell-video"
        guard let url = components.url else {
            throw NSError(domain: "SellerCaptureUploader", code: -2,
                          userInfo: [NSLocalizedDescriptionKey: "bad URL"])
        }

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
