// WSEvents.swift — Codable mirrors of the backend's /ws/dashboard broadcast
// vocabulary. Mirrors the switch statement in
// dashboard/src/hooks/useEmpireSocket.js so the iPhone sees the same events
// the dashboard does.
//
// The backend broadcasts a discriminated union: every message is a JSON
// object with a top-level `type` field. We decode in two passes:
//   1. Decode into WSEnvelope to read `type`
//   2. Re-decode the full payload into the typed struct for that type
//
// Unknown types are preserved as .unknown(type, json) rather than crashing —
// the protocol will keep evolving and we don't want the phone losing its
// WS connection every time a new event lands.

import Foundation

// ── Envelope + event dispatch ────────────────────────────────────────────

/// Just the discriminator. Read first to decide which typed struct to use.
struct WSEnvelope: Decodable {
    let type: String
}

/// The subset of backend events the iPhone actually cares about today.
/// v1 scope: Capture screen's PipelineProgressView + generic connectivity.
/// Expand as Home screen / voice integrations ship.
enum WSEvent {
    case pipelineStep(PipelineStepEvent)
    case routingDecision(RoutingDecisionEvent)
    case commentResponseVideo(CommentResponseVideoEvent)
    case agentLog(AgentLogEvent)
    case stateSync(StateSyncEvent)
    case productData(ProductDataEvent)
    case unknown(type: String, raw: Data)

    static func decode(from data: Data) throws -> WSEvent {
        let envelope = try JSONDecoder().decode(WSEnvelope.self, from: data)
        let decoder = JSONDecoder()
        switch envelope.type {
        case "pipeline_step":
            return .pipelineStep(try decoder.decode(PipelineStepEvent.self, from: data))
        case "routing_decision":
            return .routingDecision(try decoder.decode(RoutingDecisionEvent.self, from: data))
        case "comment_response_video":
            return .commentResponseVideo(try decoder.decode(CommentResponseVideoEvent.self, from: data))
        case "agent_log":
            return .agentLog(try decoder.decode(AgentLogEvent.self, from: data))
        case "state_sync":
            return .stateSync(try decoder.decode(StateSyncEvent.self, from: data))
        case "product_data":
            return .productData(try decoder.decode(ProductDataEvent.self, from: data))
        default:
            return .unknown(type: envelope.type, raw: data)
        }
    }
}

// ── Pipeline step (the reason this exists for Item 1) ────────────────────

/// Emitted by intake.py at each stage transition during /api/sell-video.
/// The iPhone's PipelineProgressView subscribes to these scoped to the
/// request_id it got back from the upload.
///
/// Status vocabulary:
///   "active" — step is currently running
///   "done"   — step completed successfully
///   "failed" — step errored (render side handles degraded)
struct PipelineStepEvent: Decodable {
    let type: String
    let request_id: String
    let step: String       // "uploaded" | "deepgram" | "claude" | "eleven" | "wav2lip" | "going_live"
    let status: String     // "active" | "done" | "failed"
    let ms: Int?           // optional per-step elapsed ms
    let detail: String?    // optional human-readable note (error msg, cache hit, etc.)
}

// ── Routing decisions ────────────────────────────────────────────────────

struct RoutingDecisionEvent: Decodable {
    let type: String
    let comment: String?
    let tool: String           // "respond_locally" | "play_canned_clip" | "block_comment" | "escalate_to_cloud"
    let reason: String?
    let ms: Int?
    let was_local: Bool?
    let cost_saved_usd: Double?
}

// ── Comment response video (the avatar just spoke) ───────────────────────

struct CommentResponseVideoEvent: Decodable {
    let type: String
    let url: String?
    let comment: String?
    let response: String?
    let total_ms: Int?
}

// ── Agent log line ───────────────────────────────────────────────────────

struct AgentLogEvent: Decodable {
    let type: String
    let entry: AgentLogEntry
}

struct AgentLogEntry: Decodable {
    let agent: String?         // "EYES" | "SELLER" | "DIRECTOR" | "BRAIN" | ... (uppercased)
    let message: String
    let timestamp: Double?
}

// ── State sync (sent once on every connect) ──────────────────────────────

struct StateSyncEvent: Decodable {
    let type: String
    let state: StateSyncPayload
}

struct StateSyncPayload: Decodable {
    let status: String?
    let product_data: ProductDataPayload?
    let sales_script: String?
    let pitch_video_url: String?
    let last_response_video_url: String?
}

struct ProductDataEvent: Decodable {
    let type: String
    let data: ProductDataPayload
}

/// Flexible product payload — any field may be missing depending on which
/// stage of intake the product is in. Only name is practically required.
struct ProductDataPayload: Decodable {
    let name: String?
    let price: String?
    let visual_details: [String]?
    let active_avatar_id: String?
}
