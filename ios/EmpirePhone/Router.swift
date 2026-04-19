// Router.swift — direct port of backend/agents/router.py
//
// Four-tool dispatcher, fed by Gemma 4 classify output on-device. The Mac
// side uses a rule-based Python implementation because we probed Gemma 4 E4B
// tool-calling at 3-31s and 1/4 accuracy — same reasoning applies here, even
// more so on a phone. Rule-based is 0ms, deterministic, and matches the
// dashboard's behavior bit-for-bit so judges see "same router, two devices."

import Foundation

// MARK: - Cost model
// Dollar value avoided per non-escalating decision. Matches
// backend/agents/router.py:COST_PER_CLOUD_COMMENT_USD.
enum CostModel {
    static let perCloudCommentUSD: Double = 0.00035
    static let savedPerTool: [Tool: Double] = [
        .respondLocally:  perCloudCommentUSD,
        .playCannedClip:  perCloudCommentUSD,
        // Blocking spam earns the full save because a naive no-router path
        // would have entered the Claude drafting pipeline before realizing
        // the comment was spam.
        .blockComment:    perCloudCommentUSD,
        .escalateToCloud: 0.0,
    ]
}

// MARK: - Tools + Decision

enum Tool: String, Hashable {
    case respondLocally   = "respond_locally"
    case playCannedClip   = "play_canned_clip"
    case blockComment     = "block_comment"
    case escalateToCloud  = "escalate_to_cloud"
}

enum DecisionArgs {
    case answerId(String)           // respond_locally
    case cannedLabel(String)        // play_canned_clip — compliment / objection / neutral
    case reason(String)             // block_comment
    case cloudComment(String)       // escalate_to_cloud
}

struct Decision {
    let tool: Tool
    let args: DecisionArgs
    let reason: String              // human-readable, shown to judges
    let ms: Int                     // router decide latency
    let wasLocal: Bool
    let costSavedUSD: Double
}

// MARK: - Cue lists — mirror backend/agents/router.py:174-189

enum RouterCues {
    static let compliment: Set<String> = [
        "love", "lovely", "beautiful", "cute", "amazing", "awesome",
        "perfect", "great", "nice", "cool", "gorgeous", "stunning",
    ]
    // Emoji checked as substrings on the raw (not tokenized) comment because
    // the word-tokenizer below strips emoji.
    static let complimentEmoji: [String] = ["❤️", "😍", "🔥", "💯", "✨", "😊"]

    static let objection: Set<String> = [
        "expensive", "overpriced", "scam", "fake", "cheap",
        "rip-off", "ripoff", "pricey",
    ]

    // URL-heavy spam cues only. Ambiguous words like "buy", "follow", "visit"
    // are NOT here — real customers ask "where do I buy this?" etc. Gemma 4's
    // classify is the primary spam signal; this list is the safety net.
    static let spam: [String] = [
        "http://", "https://", ".com/", ".net/", "www.",
        "promo code", "subscribe to", "check out my",
    ]
}

// MARK: - Tokenizer

private let tokenRegex = try! NSRegularExpression(pattern: "[a-z0-9']+")

private func tokens(of text: String) -> Set<String> {
    let lower = text.lowercased()
    let range = NSRange(lower.startIndex..., in: lower)
    var out: Set<String> = []
    tokenRegex.enumerateMatches(in: lower, range: range) { m, _, _ in
        guard let m = m, let r = Range(m.range, in: lower) else { return }
        out.insert(String(lower[r]))
    }
    return out
}

// MARK: - Product Q&A index + matcher
//
// Mirrors backend/agents/router.py:_match_product_field. Returns the
// answer_id with the most keyword hits against the comment. Multi-word
// keywords checked as substrings (so "return policy" only fires on a real
// phrase match, not on "return my email"); single-word keywords checked
// against the token set to avoid "ship" matching "relationship".

struct QAEntry: Codable {
    let keywords: [String]
    let text: String
    let url: String
}

struct Product: Codable {
    let name: String
    let qa_index: [String: QAEntry]?
    // Other fields (price, materials, etc.) ignored by the router but loaded
    // so the JSON round-trips cleanly.
    let price: String?
    let category: String?
    let materials: String?
}

// Top-level shape of products.json is a dict of product_id -> Product. We
// load the whole file and pick the first entry (matches backend's
// _load_active_product).
struct ProductCatalog: Codable {
    let products: [String: Product]

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        self.products = try container.decode([String: Product].self)
    }
    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(products)
    }
}

func matchProductField(comment: String, product: Product?) -> String? {
    guard let qa = product?.qa_index, !qa.isEmpty else { return nil }
    let cLower = comment.lowercased()
    let cTokens = tokens(of: comment)

    var bestId: String? = nil
    var bestHits = 0
    for (answerId, entry) in qa {
        var hits = 0
        for kw in entry.keywords {
            let kwL = kw.lowercased().trimmingCharacters(in: .whitespaces)
            if kwL.isEmpty { continue }
            if kwL.contains(" ") {
                if cLower.contains(kwL) { hits += 1 }
            } else {
                if cTokens.contains(kwL) { hits += 1 }
            }
        }
        if hits > bestHits {
            bestHits = hits
            bestId = answerId
        }
    }
    return bestId
}

// MARK: - Classify result (produced by CactusRunner)

struct ClassifyResult {
    let type: String    // "question" | "compliment" | "objection" | "spam"
    let draft: String?
}

// MARK: - Rule-based decider

enum Router {
    /// Pure-function rule-based router. Deterministic, 0ms on-device.
    /// Matches backend/agents/router.py:_rule_based_decide line-for-line.
    static func decide(comment: String, classify: ClassifyResult, product: Product?) -> Decision {
        let t0 = DispatchTime.now()
        let decision = _decide(comment: comment, classify: classify, product: product)
        let ms = Int(Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1_000_000)
        return Decision(
            tool: decision.tool,
            args: decision.args,
            reason: decision.reason,
            ms: ms,
            wasLocal: decision.tool != .escalateToCloud,
            costSavedUSD: CostModel.savedPerTool[decision.tool] ?? 0.0
        )
    }

    private static func _decide(comment: String, classify: ClassifyResult, product: Product?)
        -> (tool: Tool, args: DecisionArgs, reason: String)
    {
        let t = classify.type
        let cLower = comment.lowercased()
        let cTokens = tokens(of: comment)

        // 1. Spam filter — Gemma classified OR URL/promo cue.
        if t == "spam" || RouterCues.spam.contains(where: { cLower.contains($0) }) {
            return (.blockComment, .reason("spam"), "Classified as spam / URL spam")
        }

        // 2. Compliment — includes emoji check (tokenizer drops emoji).
        let hasComplimentCue = !cTokens.isDisjoint(with: RouterCues.compliment)
        let hasComplimentEmoji = RouterCues.complimentEmoji.contains(where: { comment.contains($0) })
        if t == "compliment" || hasComplimentCue || hasComplimentEmoji {
            return (.playCannedClip, .cannedLabel("compliment"),
                    "Acknowledging compliment with pre-rendered bridge clip")
        }

        // 3. Objection.
        let hasObjectionCue = !cTokens.isDisjoint(with: RouterCues.objection)
        if t == "objection" || hasObjectionCue {
            return (.playCannedClip, .cannedLabel("objection"),
                    "Defusing objection with pre-rendered bridge clip")
        }

        // 4. Question we have a local answer for.
        if t == "question", let answerId = matchProductField(comment: comment, product: product) {
            return (.respondLocally, .answerId(answerId),
                    "Matched product.qa_index key: \(answerId)")
        }

        // 5. Default — cloud.
        return (.escalateToCloud, .cloudComment(comment),
                "Needs cloud reasoning / no local match")
    }
}

// MARK: - Products.json loader

enum ProductLoader {
    /// Load the first product from the bundled products.json. Returns nil if
    /// the file is missing or empty — the router handles nil gracefully by
    /// escalating every question to cloud.
    static func loadActive(bundle: Bundle = .main) -> Product? {
        guard let url = bundle.url(forResource: "products", withExtension: "json"),
              let data = try? Data(contentsOf: url)
        else {
            return nil
        }
        do {
            let catalog = try JSONDecoder().decode([String: Product].self, from: data)
            // Stable "first" — sort by key so tests are deterministic.
            if let firstKey = catalog.keys.sorted().first {
                return catalog[firstKey]
            }
            return nil
        } catch {
            return nil
        }
    }
}
