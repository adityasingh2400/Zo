// EmpireSocket.swift — iPhone WebSocket client for the backend dashboard
// bus. Subscribes to /ws/dashboard with optional shared-secret token auth.
//
// Why the iPhone connects to /ws/dashboard (not /ws/phone):
//   /ws/phone is inbound-only (handle_phone_message dispatches sell_command /
//   comment / frame to the backend). broadcast_to_dashboards only broadcasts
//   to dashboard_clients. To receive pipeline_step + routing_decision +
//   comment_response_video events, we need to connect as a dashboard client.
//   Future: if the protocol grows asymmetric, spin out /ws/phone_subscriber.
//
// Auto-reconnect: exponential backoff 2s → 30s (capped). Demos are short;
// longer backoffs would make a stage dropout look broken.
//
// Host resolution piggybacks on GemmaClient.backendHost so the long-press
// runtime-override covers WS too (port is fixed at 8000). WS token pulls
// from UserDefaults["EmpireWSToken"] → Info.plist["EmpireWSToken"] → nil.
//
// Requires iOS 17+ (@Observable macro + Observation framework).

import Foundation
import Observation

// ── Runtime config ───────────────────────────────────────────────────────

enum EmpireWSAuth {
    static let userDefaultsKey = "EmpireWSToken"

    /// Resolved WS token. Empty string = no token (matches backend's
    /// default-off WS_SHARED_SECRET behavior).
    static var token: String {
        let fromDefaults = (UserDefaults.standard.string(forKey: userDefaultsKey) ?? "")
            .trimmingCharacters(in: .whitespaces)
        if !fromDefaults.isEmpty { return fromDefaults }
        let fromPlist = ((Bundle.main.object(forInfoDictionaryKey: userDefaultsKey) as? String) ?? "")
            .trimmingCharacters(in: .whitespaces)
        return fromPlist
    }

    static func setToken(_ token: String?) {
        let trimmed = (token ?? "").trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty {
            UserDefaults.standard.removeObject(forKey: userDefaultsKey)
        } else {
            UserDefaults.standard.set(trimmed, forKey: userDefaultsKey)
        }
    }
}

// ── Live comment model ──────────────────────────────────────────────────

/// One row in StreamView's comments card. Mutable on purpose so the
/// "replying" state can flip to "replied" when comment_response_video
/// lands for this comment's text.
struct LiveComment: Identifiable {
    let id: UUID
    let handle: String
    let text: String
    var badge: String     // "now" | "1s" | "replied"
    var replying: Bool    // drives the pulsing dot in StreamView
}

// ── Socket ───────────────────────────────────────────────────────────────

@MainActor
@Observable
final class EmpireSocket {
    enum ConnectionState: Equatable {
        case idle
        case connecting
        case connected
        case reconnecting(attempt: Int, nextDelay: TimeInterval)
        case failed(String)
    }

    private(set) var state: ConnectionState = .idle
    /// Last N agent_log entries received. Capped at 50 so memory doesn't grow.
    private(set) var agentLog: [AgentLogEntry] = []
    /// Pipeline steps keyed by request_id — PipelineProgressView reads this.
    private(set) var pipelineSteps: [String: [PipelineStepEvent]] = [:]
    private(set) var lastRoutingDecision: RoutingDecisionEvent?
    private(set) var lastResponseVideo: CommentResponseVideoEvent?
    private(set) var productData: ProductDataPayload?
    /// Current 3D spin state — StreamView renders this as a revolving frame
    /// carousel when kind=="frames".
    private(set) var view3d: View3dPayload?
    /// Rolling list of audience + response comments. Capped at 30 entries
    /// so the on-device list stays bounded even for long-running streams.
    /// StreamView reverses this for display (newest at top).
    private(set) var comments: [LiveComment] = []

    private var task: URLSessionWebSocketTask?
    private var session: URLSession = .shared
    private var reconnectAttempt = 0
    private var isStarted = false
    private var port: Int = 8000

    // ── Public API ────────────────────────────────────────────────────────

    /// Starts the connect/receive loop. Idempotent; calling twice is a no-op.
    func start(port: Int = 8000) {
        guard !isStarted else { return }
        isStarted = true
        self.port = port
        Task { await connectLoop() }
    }

    /// Stops the socket and prevents further reconnects.
    func stop() {
        isStarted = false
        task?.cancel(with: .goingAway, reason: nil)
        task = nil
        state = .idle
    }

    /// Clear the buffered pipeline steps for a request_id. Call when the
    /// Capture screen finishes so memory doesn't grow unbounded across reps.
    func clearPipelineSteps(requestID: String) {
        pipelineSteps.removeValue(forKey: requestID)
    }

    /// Steps for a specific request_id, sorted by arrival order.
    func steps(for requestID: String) -> [PipelineStepEvent] {
        pipelineSteps[requestID] ?? []
    }

    // ── Connection loop ───────────────────────────────────────────────────

    private func connectLoop() async {
        while isStarted {
            await connectOnce()
            if !isStarted { return }
            let delay = reconnectDelay()
            state = .reconnecting(attempt: reconnectAttempt, nextDelay: delay)
            try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            reconnectAttempt += 1
        }
    }

    private func connectOnce() async {
        guard let url = buildURL() else {
            state = .failed("backend host not configured — long-press the status pill")
            // Don't tight-loop if unconfigured; back off.
            try? await Task.sleep(nanoseconds: 3_000_000_000)
            return
        }

        state = .connecting
        let newTask = session.webSocketTask(with: url)
        task = newTask
        newTask.resume()

        // First successful receive → connected.
        do {
            let firstMsg = try await newTask.receive()
            reconnectAttempt = 0  // successful connect resets backoff
            state = .connected
            handle(firstMsg)
            // Continue reading until socket dies.
            try await receiveLoop(newTask)
        } catch {
            // Drop through; connectLoop will re-enter.
        }

        task = nil
    }

    private func receiveLoop(_ task: URLSessionWebSocketTask) async throws {
        while isStarted, task.state == .running {
            let msg = try await task.receive()
            handle(msg)
        }
    }

    private func handle(_ msg: URLSessionWebSocketTask.Message) {
        let data: Data
        switch msg {
        case .string(let s): data = s.data(using: .utf8) ?? Data()
        case .data(let d):   data = d
        @unknown default:    return
        }
        guard !data.isEmpty else { return }

        do {
            let event = try WSEvent.decode(from: data)
            apply(event)
        } catch {
            // Unknown / malformed payloads are OK; keep the socket alive.
            // (Logging intentionally light — this runs on every frame.)
        }
    }

    private func apply(_ event: WSEvent) {
        switch event {
        case .pipelineStep(let step):
            var arr = pipelineSteps[step.request_id] ?? []
            arr.append(step)
            pipelineSteps[step.request_id] = arr

        case .routingDecision(let d):
            lastRoutingDecision = d

        case .commentResponseVideo(let v):
            lastResponseVideo = v
            // When the avatar just replied to a comment, mark the matching
            // pending comment as "replied" so StreamView's pulsing indicator
            // settles. Keep the comment in the list — viewers like to see
            // the reply context.
            if let text = v.comment, let idx = comments.lastIndex(where: { $0.text == text && $0.replying }) {
                comments[idx].replying = false
                comments[idx].badge = "replied"
            }

        case .agentLog(let log):
            agentLog.append(log.entry)
            if agentLog.count > 50 {
                agentLog.removeFirst(agentLog.count - 50)
            }

        case .stateSync(let s):
            productData = s.state.product_data

        case .productData(let p):
            productData = p.data

        case .audienceComment(let c):
            guard let text = c.text, !text.isEmpty else { return }
            let comment = LiveComment(
                id: UUID(),
                handle: c.username.map { "@\($0)" } ?? "@guest",
                text: text,
                badge: "now",
                replying: true  // optimistic — will clear when response_video fires
            )
            comments.append(comment)
            if comments.count > 30 {
                comments.removeFirst(comments.count - 30)
            }

        case .view3d(let v):
            view3d = v.data

        case .unknown:
            break  // forward-compat: new event types we haven't modeled yet
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    private func buildURL() -> URL? {
        // Route through GemmaClient.websocketBaseURL so the phone supports:
        //   - Plain hostname/IP   → ws://host:8000
        //   - http:// full URL    → ws://host
        //   - https:// full URL   → wss://host  (cloudflared tunnels require wss)
        // The `port` property is still used for plain-host fallback.
        guard let wsBase = GemmaClient.websocketBaseURL else { return nil }
        let token = EmpireWSAuth.token

        // If wsBase is a full tunnel URL it already has scheme+host (+ maybe
        // port), so just append the path. If it's plain-host-based we need
        // to rebuild with the configured port to support non-defaults.
        let pathURL: URL
        if GemmaClient.backendIsFullURL {
            pathURL = wsBase.appendingPathComponent("ws/dashboard")
        } else {
            // Rebuild with the explicit port — websocketBaseURL uses 8000
            // by default but the caller may have overridden it.
            guard var components = URLComponents(url: wsBase, resolvingAgainstBaseURL: false) else {
                return nil
            }
            components.port = port
            components.path = "/ws/dashboard"
            if !token.isEmpty {
                components.queryItems = [URLQueryItem(name: "token", value: token)]
            }
            return components.url
        }

        // Tunnel path — add token as query if present.
        if token.isEmpty { return pathURL }
        var components = URLComponents(url: pathURL, resolvingAgainstBaseURL: false)
        components?.queryItems = [URLQueryItem(name: "token", value: token)]
        return components?.url ?? pathURL
    }

    /// Exponential backoff: 2, 4, 8, 16, 30, 30, 30… seconds. Capped at 30
    /// because stage demos are short — longer reconnects look broken.
    private func reconnectDelay() -> TimeInterval {
        let exp = min(5, reconnectAttempt)
        let base = pow(2.0, Double(exp))
        return min(30.0, base)
    }
}
