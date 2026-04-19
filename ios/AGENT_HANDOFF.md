# AGENT_HANDOFF — finish the EmpirePhone iOS app

The `bootstrap_phone_app.sh` script already staged all the code, models, and
clips. Everything below requires the Xcode GUI — a coding agent with access to
Xcode or a human driver can complete it in ~10 minutes.

## 1. Create the Xcode project

1. Open **Xcode.app**.
2. **File → New → Project → iOS → App → Next**.
3. Fill in:
   - Product Name: `EmpirePhone`
   - Team: *your free Apple ID* (sign in via `Xcode → Settings → Accounts`)
   - Organization Identifier: `com.codyk` (or any reverse-DNS you own)
   - Bundle Identifier will auto-fill to `com.codyk.empire`
   - Interface: **SwiftUI**
   - Language: **Swift**
   - Tests / Core Data / CloudKit: **all off**
4. Save location: the repo root. **Important:** when Xcode asks where to save,
   point it at `ios/` in this repo. If Xcode refuses because the folder isn't
   empty, pick a temp location and manually move the generated
   `EmpirePhone.xcodeproj` into `ios/` after.
5. Close the default `ContentView.swift` and `EmpirePhoneApp.swift` that
   Xcode auto-generated — **delete them from disk** (Move to Trash). The
   ones that shipped from this repo in `ios/EmpirePhone/` replace them.

## 2. Drag the source files into the project

From Finder, drag these into the Xcode sidebar on the `EmpirePhone` group:

- `ios/Cactus.swift`  (top level, NOT inside the EmpirePhone folder — it's a
  vendored copy of the Cactus Swift wrapper)
- Everything already inside `ios/EmpirePhone/` that isn't `.plist` or `.json`:
  - `AudioRecorder.swift`
  - `CactusRunner.swift`
  - `ContentView.swift`
  - `EmpirePhoneApp.swift`
  - `Router.swift`
  - `VideoDirector.swift`

In the drag-import dialog:
- **Copy items if needed: OFF** (they already live in this repo — no need to duplicate).
- **Create groups**, not folder references.
- Targets: **EmpirePhone** (checked).

## 3. Drag in resources

These must be **folder references** (blue folders) so `Bundle.main.url(forResource:inDirectory:)` works:

- `ios/EmpirePhone/Models/` → drag into Xcode as a **folder reference**
- `ios/EmpirePhone/Clips/` → drag into Xcode as a **folder reference**

Also drag in:
- `ios/EmpirePhone/products.json` (normal group file, Copy items OFF)

## 4. Integrate the Cactus xcframework

1. Project Navigator → click the blue **EmpirePhone** project at the top →
   **TARGETS / EmpirePhone → General**.
2. Scroll to **Frameworks, Libraries, and Embedded Content**.
3. Drag `ios/cactus-ios.xcframework` from Finder into this list.
4. Set **Embed** to **Embed & Sign**.

## 5. Replace Info.plist contents

Xcode's auto-generated Info.plist doesn't have the microphone permission.

1. In the project navigator, open the generated `Info.plist` (or
   `EmpirePhone/Info.plist` — the one Xcode made, not the one that shipped).
2. Copy the contents of `ios/EmpirePhone/Info.plist` (from this repo) into it,
   replacing the generated contents entirely.
3. Alternatively: replace the file on disk and re-add it.

The critical key is `NSMicrophoneUsageDescription`.

## 6. Signing + deployment target

1. TARGETS / EmpirePhone → **Signing & Capabilities**.
2. **Automatically manage signing: ON**.
3. **Team**: pick your Apple ID.
4. TARGETS / EmpirePhone → **General → Minimum Deployments → iOS 15.0**.

## 7. First build

1. Plug in the iPhone. Unlock it. Tap **Trust this computer**.
2. Top bar → device picker → select your iPhone.
3. **⌘B** to build. First build takes 3–5 minutes (Cactus xcframework is big).
4. Expect zero errors. If you see
   `No such module 'cactus'` → step 4 wasn't done correctly.
5. **⌘R** to install + run.
6. On the phone: first launch will prompt "Untrusted developer". Fix:
   **Settings → General → VPN & Device Management → [your Apple ID] → Trust**.
   Then re-open the app.
7. Splash shows "Loading whisper on Cactus…" for ~2s.

## 8. Demo test

1. Enable **Airplane Mode** on the phone.
2. Hold "HOLD TO SPEAK" — say `is it real leather` → release.
3. Expected within ~1 second:
   - Transcript card: "is it real leather" with a `~250ms` latency badge.
   - Router card: `respond_locally` (green) `0ms` `-$0.00035`.
   - Avatar video plays the pre-rendered wallet answer MP4.

If any step is off, see `ios/README.md` troubleshooting section.

## 9. Common breakages

| Symptom | Fix |
|---|---|
| Build error `No such module 'cactus'` | xcframework not embedded (Step 4). |
| Build error on `import cactus` in Cactus.swift | Drag the .xcframework *before* building; Xcode generates the module map on integration. |
| First launch crashes with "whisper model not bundled" | `Models/` was added as a group, not a folder reference. Delete the group, re-add as folder reference. |
| Avatar shows black frame | `Clips/idle_loop.mp4` missing. Re-run `phase_clips` or AirDrop the file. |
| Microphone button does nothing | `NSMicrophoneUsageDescription` missing from Info.plist. Step 5 wasn't done. |
| Apple cert expired after 7 days (free team) | Re-build + re-install from Xcode. Takes 2 min. |
