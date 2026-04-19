// EmpirePhoneApp.swift — @main entry.
//
// Single-scene SwiftUI app. Cold launch: ContentView calls .task to boot
// the Cactus models off the main thread and flips the splash after ~10s.
// No scene delegate, no AppDelegate — we don't need them.

import SwiftUI

@main
struct EmpirePhoneApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .statusBarHidden(false)
                .preferredColorScheme(.dark)
        }
    }
}
