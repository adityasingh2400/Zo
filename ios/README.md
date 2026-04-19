# EmpirePhone — standalone iOS demo

> Cactus whisper on an iPhone + the same Swift router port of `backend/agents/router.py`. Push-to-talk voice in, router decision out, pre-rendered avatar plays the answer. **Works in Airplane Mode.**

The laptop dashboard is the seller's studio, running the full Gemma 4 + Cactus stack. This app is proof the Cactus runtime + our router port to mobile without a cloud round-trip.

## Scope (read this first)

The phone does **not** run an on-device LLM classifier. We tested four candidates before shipping this:

| Candidate | Size | Why not |
|---|---|---|
| Gemma 4 E2B | 6.4 GB | Multimodal-only — audio + vision encoders required for init; can't strip. Bundle too big. |
| Gemma 4 E4B | ~10 GB | Same class as E2B. Plus 3–31 s tool-call latency at 1/4 accuracy (Mac probe). |
| Gemma 3n-E2B | 4.5 GB | Text-only but CPU only on phone (no Apple NPU) → 5–12 s per classify on A15. |
| functiongemma-270m | 267 MB | Refuses every prompt (0/4 on demo inputs). |

Our router doesn't actually *need* the LLM. The rule-based decider in `backend/agents/router.py` drives almost every decision from the comment text itself (URL spam cues, compliment words+emoji, objection words, product qa_index keyword match). The `classify.type` LLM signal is used as a supplement, not a primary. The phone ships the rule-based router with a default `classify.type="question"` and still routes every demo case correctly.

What the phone does on-device:
- **Cactus whisper-tiny** for voice transcription (120 MB, Apple NPU)
- **Swift port of the rule-based router** (0 ms decision, 100% accuracy on our demo suite)
- **AVPlayer playback** of the pre-rendered MP4s baked into the bundle

What stays on the Mac:
- Gemma 4 on Cactus (the seller's "studio" side)
- TTS, Wav2Lip / LatentSync render pipeline
- Cloud escalation (Bedrock + ElevenLabs)

## Prerequisites (on your Mac)

- **Xcode 15+** installed from the Mac App Store (Command Line Tools alone aren't enough — you need the full IDE).
- **cmake** (for building the Cactus iOS framework). `brew install cmake`.
- The Cactus repo cloned at `~/cactus/` with the Python CLI available at `~/cactus/venv/bin/cactus`.
- Weights downloaded:
  ```bash
  cd ~/cactus
  ./venv/bin/cactus download google/gemma-4-E2B-it
  ./venv/bin/cactus download openai/whisper-tiny
  ```
- An iPhone 13 or newer (A15+), cable-connected, trusted to the Mac.
- A free Apple ID signed into Xcode (`Xcode → Settings → Accounts`).

## One-time setup

### 1. Build the Cactus iOS XCFramework

```bash
cd ~/cactus/apple
./build.sh
```

Produces `~/cactus/apple/cactus-ios.xcframework`. Takes ~5 minutes.

### 2. Generate the Xcode project

This directory ships the Swift source + resources but **not** the `.xcodeproj` (Xcode regenerates it cleanly from the source tree). From `chicken/ios/`:

1. Open Xcode → File → New → Project → iOS → App
   - Product name: **EmpirePhone**
   - Team: your free Apple ID
   - Bundle ID: `com.codyk.empire`
   - Interface: **SwiftUI**
   - Language: **Swift**
   - Save location: overwrite `chicken/ios/`
2. Delete the auto-generated `ContentView.swift` and `EmpirePhoneApp.swift` — the ones in `ios/EmpirePhone/` replace them.
3. Drag the following into the Xcode sidebar (use "Copy items if needed" OFF, "Create groups" ON):
   - `ios/Cactus.swift`
   - `ios/EmpirePhone/*.swift` (Router, AudioRecorder, CactusRunner, VideoDirector, ContentView, EmpirePhoneApp)
   - `ios/EmpirePhone/products.json`
   - `ios/EmpirePhone/Clips/` (whole folder — use "Create folder references")
   - `ios/EmpirePhone/Models/` after you copy the weights in (step 3)
4. In project Settings:
   - Drag `~/cactus/apple/cactus-ios.xcframework` into "Frameworks, Libraries, and Embedded Content" with "Embed & Sign".
   - Replace the generated `Info.plist` contents with `ios/EmpirePhone/Info.plist` (adds `NSMicrophoneUsageDescription`).
   - Deployment Target: iOS 15.0+.
   - Signing: "Automatically manage signing" with your free Apple ID team.

### 3. Bundle the weights

```bash
mkdir -p ios/EmpirePhone/Models
cp -R ~/cactus/weights/gemma-4-e2b-it ios/EmpirePhone/Models/gemma-4-E2B-it
cp -R ~/cactus/weights/whisper-tiny ios/EmpirePhone/Models/whisper-tiny
```

Drag the `Models/` folder into Xcode as a **folder reference** (blue folder), not a group — so `Bundle.main.path(forResource:"gemma-4-E2B-it", inDirectory:"Models")` resolves.

Expected total bundled size: ~720 MB (600 MB Gemma E2B INT4 + 120 MB whisper-tiny).

### 4. Connect the phone and run

1. Plug iPhone in. Unlock it. Tap "Trust this computer".
2. Xcode → device picker → select your iPhone.
3. ⌘R to build & install. First build takes 3-5 minutes.
4. On the phone: Settings → General → VPN & Device Management → trust your developer cert.
5. Open the app. Splash shows ~10 s while Cactus loads models.

## Demo flow

1. Enable **Airplane Mode** on the phone.
2. Hold the "HOLD TO SPEAK" button.
3. Say **"is it real leather"**, release.
4. Transcript card appears (~250 ms). Classify card (~400 ms). Router card lands `respond_locally` with "0 ms / -$0.00035".
5. The pre-rendered wallet answer plays as a full talking avatar.

All four router paths work:
- `"is it real leather"` → respond_locally → wallet_real_leather.mp4
- `"I love this wallet ❤️"` → play_canned_clip → bridge_compliment_*.mp4
- `"check out my link at cheap.com"` → block_comment → no video
- `"how does this compare to the Apple Watch"` → escalate_to_cloud → "would route to cloud" card

## Troubleshooting

- **"Models not bundled" on launch** — you didn't drag the `Models/` folder as a folder reference. Delete the group in Xcode, re-add with "Create folder references".
- **Build errors about `cactus` module not found** — the xcframework isn't embedded. Check Target → General → Frameworks, Libraries, and Embedded Content.
- **Mic permission denied on first press** — Settings → EmpirePhone → Microphone → toggle on.
- **Avatar doesn't play** — MP4s aren't bundled. Confirm Clips/ is in Xcode as a folder reference; check `Bundle.main.url(forResource:"wallet_real_leather", withExtension:"mp4")` returns non-nil.
- **Free cert expires after 7 days** — re-build+install from Xcode (takes 2 min). Nothing to re-sign manually.

## What this app is not

- **Not running Wav2Lip / LatentSync on the phone.** Real-time lip-sync on A15 would be 30-60s per clip; infeasible. The phone plays pre-rendered MP4s (rendered once on RunPod 5090, baked into the app). The on-device part is *the router*.
- **Not streaming to the laptop.** This is a standalone demo. The dashboard and this app are parallel proofs of the same router running on two devices.
- **Not executing the cloud escalate pipeline.** The phone shows the router decision for `escalate_to_cloud` and stops there — the Claude + TTS + Wav2Lip chain stays on the Mac.
