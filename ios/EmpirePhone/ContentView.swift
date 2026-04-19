// ContentView.swift — app shell. Mounts StreamView as the viewer-facing
// surface per Cody's sketch:
//
//   [HEAD SHOT   ] [3D SPIN   ]
//   [(avatar)    ] [BEST FRMS ]
//   [COMMENTS    ] [LINK BUY  ]
//
// The previous voice-driven demo (push-to-talk + whisper + Swift router
// dispatch + Gemma card) has been retired — Mac owns those paths via the
// /ws/dashboard bus. If you need the legacy flow back, `git log ContentView.swift`
// has it (before this commit).
//
// Kept files that are no longer mounted but still compile:
//   CactusRunner.swift / AudioRecorder.swift / Router.swift
// Safe to delete in a cleanup commit; left in place so the xcframework
// linkage + Cactus import path stays stable.

import SwiftUI
import AVKit

struct ContentView: View {
    @StateObject private var director = VideoDirector()
    @State private var socket = EmpireSocket()

    var body: some View {
        StreamView(director: director, socket: socket)
            .preferredColorScheme(.dark)
            .task { await bootstrap() }
    }

    /// Minimal bootstrap: kick off the Director's idle loop so the head shot
    /// has something to play on first render, then open the WS subscription.
    /// No whisper / mic / router init — those paths are Mac-side now.
    private func bootstrap() async {
        director.backToIdle()
        socket.start()
    }
}
