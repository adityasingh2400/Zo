// PipelineProgressView.swift — "Building your avatar" pipeline card for the
// seller capture flow. Mirrors the empire-mobile.jsx mockup's step list:
//
//   ✓ Uploaded
//   ● Deepgram ASR        (active — pulsing aura)
//   ○ Claude identifies
//   ○ ElevenLabs voice
//   ○ Wav2Lip · 5090
//   ○ Going live
//
// Subscribes to an EmpireSocket, scoped to a specific request_id returned
// by the /api/sell-video upload. Backend's intake.py emits pipeline_step
// events (Item 1 backend half) which the socket buffers by request_id.

import SwiftUI

struct PipelineProgressView: View {
    let requestID: String
    let socket: EmpireSocket

    /// Step manifest — order matches the mockup + intake.py emit order.
    private static let stepOrder: [(key: String, label: String)] = [
        ("uploaded",   "Uploaded"),
        ("deepgram",   "Deepgram ASR"),
        ("claude",     "Claude identifies object"),
        ("eleven",     "ElevenLabs voice"),
        ("wav2lip",    "Wav2Lip · 5090"),
        ("going_live", "Going live"),
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("BUILDING YOUR AVATAR")
                    .font(.system(size: 9, weight: .heavy, design: .monospaced))
                    .tracking(0.8)
                    .foregroundColor(.white.opacity(0.5))
                Spacer()
                if let last = elapsedLabel {
                    Text(last)
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundColor(.white.opacity(0.4))
                }
            }

            VStack(alignment: .leading, spacing: 8) {
                ForEach(Array(Self.stepOrder.enumerated()), id: \.offset) { idx, step in
                    row(for: step, isFirst: idx == 0)
                }
            }
        }
        .padding(14)
        .background(Color.white.opacity(0.08))
        .cornerRadius(18)
    }

    // MARK: - Rows

    @ViewBuilder
    private func row(for step: (key: String, label: String), isFirst: Bool) -> some View {
        let status = statusFor(step.key)
        HStack(spacing: 10) {
            indicator(status: status)
            Text(step.label)
                .font(.system(size: 13))
                .foregroundColor(textColor(status: status))
                .fontWeight(status == .active ? .semibold : .regular)
                .frame(maxWidth: .infinity, alignment: .leading)
            if status == .done, let ms = elapsedMsFor(step.key) {
                Text("\(ms)ms")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(.white.opacity(0.4))
            } else if status == .done {
                Text("done")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(.white.opacity(0.4))
            } else if status == .failed {
                Text("failed")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(.red.opacity(0.8))
            }
        }
    }

    @ViewBuilder
    private func indicator(status: StepStatus) -> some View {
        let size: CGFloat = 18
        switch status {
        case .pending:
            Circle()
                .fill(Color.white.opacity(0.08))
                .frame(width: size, height: size)
        case .active:
            ZStack {
                Circle()
                    .stroke(Color.blue.opacity(0.8), lineWidth: 1.5)
                    .frame(width: size, height: size)
                Circle()
                    .fill(Color.blue)
                    .frame(width: 8, height: 8)
            }
        case .done:
            ZStack {
                Circle()
                    .fill(Color.green)
                    .frame(width: size, height: size)
                Image(systemName: "checkmark")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundColor(.white)
            }
        case .failed:
            ZStack {
                Circle()
                    .fill(Color.red.opacity(0.25))
                    .frame(width: size, height: size)
                Image(systemName: "xmark")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundColor(.red)
            }
        }
    }

    // MARK: - State derivation

    private enum StepStatus {
        case pending, active, done, failed
    }

    private func statusFor(_ key: String) -> StepStatus {
        let events = socket.steps(for: requestID).filter { $0.step == key }
        // Last event for this step wins.
        guard let latest = events.last else {
            // Rule: a step is implicitly done if a LATER step has events.
            if hasAnyEventLaterThan(key) { return .done }
            return .pending
        }
        switch latest.status {
        case "done": return .done
        case "failed": return .failed
        case "active": return .active
        default: return .pending
        }
    }

    /// True if any step that comes AFTER `key` in the manifest has received
    /// events. Used to mark earlier steps as implicitly done when the
    /// backend didn't emit an explicit "done" (it jumped ahead).
    private func hasAnyEventLaterThan(_ key: String) -> Bool {
        guard let myIdx = Self.stepOrder.firstIndex(where: { $0.key == key }) else {
            return false
        }
        let later = Self.stepOrder.suffix(from: myIdx + 1).map(\.key)
        return socket.steps(for: requestID).contains { later.contains($0.step) }
    }

    private func elapsedMsFor(_ key: String) -> Int? {
        socket.steps(for: requestID)
            .last(where: { $0.step == key && $0.status == "done" })?.ms
    }

    private func textColor(status: StepStatus) -> Color {
        switch status {
        case .pending: return .white.opacity(0.4)
        case .active:  return .white
        case .done:    return .white.opacity(0.6)
        case .failed:  return .red.opacity(0.9)
        }
    }

    // MARK: - Elapsed timer

    /// Cumulative elapsed ms, from the sum of "done" step latencies.
    /// Fallback to "in progress" if nothing has completed yet.
    private var elapsedLabel: String? {
        let doneSteps = socket.steps(for: requestID)
            .filter { $0.status == "done" }
        let totalMs = doneSteps.compactMap(\.ms).reduce(0, +)
        if totalMs > 0 {
            let s = Double(totalMs) / 1000.0
            return String(format: "%.1fs", s)
        }
        let anyEvents = !socket.steps(for: requestID).isEmpty
        return anyEvents ? "in progress" : nil
    }
}
