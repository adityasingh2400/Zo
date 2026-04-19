# EmpirePhone iOS — Session Handoff (2026-04-18)

## Where we left off

App **builds and runs on Cody's iPhone**. The on-device pipeline (HOLD TO SPEAK → AVAudioRecorder → Cactus whisper-tiny → rule-based router → AVPlayer MP4) is wired end-to-end. Last open issue: whisper auto-detects the audio language as Norwegian Nynorsk (`<|nn|>` tokens) instead of English, so the transcript card shows garbled tokens and the router falls through to `escalate_to_cloud` because no English text matches the wallet/compliment/objection keyword rules.

**Last code change**: `CactusRunner.swift` now passes `prompt = "Transcribe the English audio."` as the 3rd arg to `cactusTranscribe` (the non-stream API doesn't read `language` from the options JSON — only the streaming API does). User has not yet rebuilt + tested this change.

## Status

| Piece | State |
|---|---|
| Repo cloned, all 326+ files staged | ✅ |
| Bootstrap script (`scripts/bootstrap_phone_app.sh`) | ✅ committed |
| `cactus-ios.xcframework` (manually assembled from static `libcactus-{device,simulator}.a` + module map) | ✅ in `ios/cactus-ios.xcframework/` (gitignored, 13 MB) |
| `libcurl.xcframework` (wrapped from `~/cactus/libs/curl/ios/{device,simulator}/libcurl.a`) | ✅ in `ios/libcurl.xcframework/` (gitignored) |
| Xcode project (`ios/EmpirePhone.xcodeproj`) | ✅ correctly wired: signing, embedded framework, deployment target iOS 26.3, bundle id `com.codyk.EmpirePhone`, mic permission via `INFOPLIST_KEY_NSMicrophoneUsageDescription`, all 8 frameworks linked |
| iOS app builds in Xcode 26 | ✅ |
| App installs + launches on Cody's iPhone (iOS 26.3.1) | ✅ |
| Mic permission granted | ✅ (`Settings → Privacy & Security → Microphone → EmpirePhone = ON`) |
| AVAudioRecorder captures real audio (~96 KB / 3 s) | ✅ |
| Whisper produces tokens at ~200 ms latency | ✅ |
| Whisper transcribes **in English** | ❌ — outputs `<\|nn\|>` tokens (Norwegian Nynorsk language code) |
| Router matches local rule → plays wallet MP4 | ❌ blocked on whisper output |

## How to resume next session

### 0. Re-open the Xcode project
```bash
open /Users/codykandarian/Desktop/Zo/ios/EmpirePhone.xcodeproj
```

### 1. Rebuild + reinstall the app (latest English-prompt change)
- In Xcode: `⌘R`
- On iPhone: hold "HOLD TO SPEAK" 2–3 s, say "is it real leather", release
- Watch the Transcript card (above the WHISPER JSON). If it shows real English ("is it real leather"), router will then match `wallet_real_leather` rule and play the MP4. **DONE.**

### 2. If whisper still returns `<|nn|>` tokens — next things to try (in order)

The non-stream `cactus_transcribe` C function does NOT read `language` from options JSON. Confirmed by reading `~/cactus/cactus/ffi/cactus_transcribe.cpp` and `parse_inference_options_json` in `cactus_utils.h:1272` — only `temperature`, `top_p`, `min_p`, `repetition_penalty`, `top_k`, `max_tokens`, `use_vad` are parsed. The `prompt` parameter (3rd arg) becomes the task instruction (line 195 of cactus_transcribe.cpp).

**Try A:** Switch from `cactusTranscribe` (one-shot) to **streaming** API (`cactus_stream_transcribe_start`) which DOES parse `language` from options JSON (`cactus_stream.cpp:163: language = json_string(json, "language")`). `Cactus.swift:250` already exposes `cactusStreamTranscribeStart(_ model, _ optionsJson)` — pass `{"language":"en"}`.

**Try B:** Detect language first via `cactusDetectLanguage` (`Cactus.swift:191`), then transcribe with the detected code as prompt.

**Try C:** If A and B both fail to give clean English text, swap whisper-tiny for whisper-base (~150 MB instead of 120 MB) — tiny may just be too small for noisy iPhone-mic audio. Re-run `cactus download openai/whisper-base` in `~/cactus/`, copy weights to `ios/EmpirePhone/Models/whisper-base/`, update `ContentView.swift:226` from `"whisper-tiny"` → `"whisper-base"`.

**Try D:** Pre-process the recorded audio with noise gate / normalize — `AVAudioRecorder` records at whatever ambient level the iPhone mic picks up. Quiet/noisy audio confuses whisper's language detection.

### 3. Once transcription is correct, the rest is wired
- `Router.swift` matches keywords against `products.json` (wallet QA index)
- On match → `VideoDirector` plays `wallet_<answer_id>.mp4` from `Clips/`
- On no match → escalate_to_cloud (in airplane mode, just shows overlay)

## Files changed this session

### `ios/EmpirePhone/AudioRecorder.swift`
**Rewrote entirely**. Was AVAudioEngine + input tap → AVAudioConverter (unreliable on iOS 26, kept losing audio after 0.17 s due to repeated session interruptions). Now uses **`AVAudioRecorder`** writing directly to a temp WAV file at 16 kHz mono int16 LE; `stop()` reads the file, strips the 44-byte WAV header, returns raw PCM bytes. Added `configureSession()` that sets `.playAndRecord`/`.default` with `.defaultToSpeaker, .allowBluetooth` ONCE at app launch (called from `ContentView.bootstrap()`). Diagnostic `print()`s in `start()`/`stop()` show file path and PCM byte count in the Xcode console.

### `ios/EmpirePhone/CactusRunner.swift`
Added `import Combine` (was missing — caused `ObservableObject` conformance error). Changed `cactusTranscribe` call to pass `prompt = "Transcribe the English audio."` (3rd arg) so whisper biases toward English.

### `ios/EmpirePhone/VideoDirector.swift`
Added `import Combine` (same reason).

### `ios/EmpirePhone/ContentView.swift`
Added `try? recorder.configureSession()` in `bootstrap()` after permission grant. Added `director.player.pause()` + 50 ms sleep at start of `beginRecording()` so the AVPlayer doesn't fight the mic recorder for the audio session.

### `ios/EmpirePhone.xcodeproj/project.pbxproj`
Hand-edited for several Xcode 26 issues:
1. Added `Models` as a **folder reference** (PBXFileReference with `lastKnownFileType = folder`) at the project root, copied as a tree into the bundle. Without this, Xcode 26's synchronized-folder system flattened `Models/whisper-tiny/` and `Models/whisper-tiny/vad/` into the same bundle root, causing `config.txt` / `coremldata.bin` / `weight.bin` collisions.
2. Moved `Models/` on disk OUT of the synchronized `EmpirePhone/` folder up to `ios/Models/` so the synchronized-folder system can't see it.
3. `INFOPLIST_KEY_NSMicrophoneUsageDescription` added to both Debug and Release build configs.
4. `FRAMEWORK_SEARCH_PATHS = "$(PROJECT_DIR)"` added to both configs.
5. Frameworks linked in `Frameworks` build phase: `cactus-ios.xcframework`, `libcurl.xcframework`, `CoreML.framework`, `Accelerate.framework`, `Security.framework`, `SystemConfiguration.framework`, `CFNetwork.framework`, `libc++.tbd`. ALL set to **Do Not Embed** (cactus-ios + libcurl are static libraries wrapped in xcframeworks; embedding causes runtime errors).
6. Removed iPad/Mac/Vision destinations (iPhone-only).
7. Lowered iOS deployment target from 26.4 → 26.3 (Cody's iPhone is on iOS 26.3.1).
8. Removed Tests target (was `ZoTests` + `ZoUITests` from earlier wrong-named Xcode project; we redid the whole project as `EmpirePhone` in `~/Desktop/EmpirePhone/` then moved `EmpirePhone.xcodeproj` into `ios/`).

### `~/cactus/apple/CmakeLists.txt` (NOT in this repo — out-of-tree)
Added to **both** `target_compile_options(cactus PRIVATE ...)` AND `set(APPLE_COMMON_OPTIONS ...)`:
```cmake
"SHELL:-include ${CMAKE_OSX_SYSROOT}/usr/include/stdint.h"
```
This was needed because Apple's `<arm_neon.h>` (transitively included from cactus's NEON kernels) uses `int8_t`, `uint64_t`, etc. but doesn't include `<stdint.h>` itself, AND in C++ mode the libcxx wrapper for `<stdint.h>` doesn't put the typedefs in the global namespace where `arm_vector_types.h` looks for them. The full SDK path bypass forces the C `<stdint.h>` (which DOES define globals) to load before any libcxx wrapper interferes.

### `~/cactus/cactus/kernel/kernel_*.cpp` (14 files, NOT in this repo)
Added `#include <stdint.h>` near the top of each file before `<arm_neon.h>`. Helps but not sufficient on its own — the `-include` in CMakeLists is the load-bearing fix.

## Critical gotchas / bug reference

### A. CommandLineTools must NOT be installed when building cactus
`cmake` mixes libcxx headers from `/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1/` with Xcode's iOS SDK, causing `ldiv_t`, `wcschr`, etc. to be undefined in the libcxx wrappers. We moved CommandLineTools out (`sudo mv /Library/Developer/CommandLineTools /tmp/CommandLineTools.bak`) for the cactus build. **You should restore it** when done with cactus rebuilds:
```bash
sudo mv /tmp/CommandLineTools.bak /Library/Developer/CommandLineTools
```

### B. Two Xcode versions are in play
- **Xcode 26.4** (`/Applications/Xcode.app`) — used to build the EmpirePhone iOS app
- **Xcode 16.4** (`/Applications/Xcode-16.4.0.app`) — used to build the cactus-ios.xcframework (Xcode 26's clang21 has the same `arm_vector_types.h` issue but the patches still don't fully resolve it; Xcode 16's clang17 + the `-include` patch works)

Switch with `sudo xcode-select -s <path>/Contents/Developer` between rebuilds.

### C. cactus.framework binary doesn't exist after `~/cactus/apple/build.sh`
The script's framework build phase silently fails (output piped to /dev/null). The cactus.framework dirs end up containing only `Headers/`, no Mach-O binary. **Workaround we used**: assemble the xcframework manually from the static `.a` files:
```bash
APPLE=/Users/codykandarian/cactus/apple
HEADERS=/tmp/cactus_headers_staging
mkdir -p $HEADERS
cp $APPLE/cactus-ios.xcframework/ios-arm64/cactus.framework/Headers/*.h $HEADERS/
# Replace the framework module map with a regular module map (we wrap a static lib, not a dylib)
cat > $HEADERS/module.modulemap <<EOF
module cactus {
    umbrella header "cactus_ffi.h"
    export *
    module * { export * }
}
EOF
rm -rf $APPLE/cactus-ios.xcframework
xcodebuild -create-xcframework \
  -library $APPLE/libcactus-device.a -headers $HEADERS \
  -library $APPLE/libcactus-simulator.a -headers $HEADERS \
  -output $APPLE/cactus-ios.xcframework
cp -R $APPLE/cactus-ios.xcframework /Users/codykandarian/Desktop/Zo/ios/cactus-ios.xcframework
```

### D. iPhone Developer Mode + cert trust
- Settings → Privacy & Security → **Developer Mode** must be ON (requires phone restart)
- Settings → General → VPN & Device Management → trust `codykandarian@comcast.net` developer cert
- Free-team cert expires every 7 days — re-build + re-install from Xcode to refresh

## Repo layout reminder

```
Zo/                                       (this repo, on main branch)
├── ios/                                  (iOS Swift app — created this session)
│   ├── EmpirePhone.xcodeproj/            (committed)
│   ├── EmpirePhone/                      (synchronized folder, all .swift files)
│   ├── Models/                           (gitignored, 66 MB whisper-tiny weights)
│   ├── EmpirePhone/Clips/                (gitignored, 16 MP4s, 66 MB)
│   ├── cactus-ios.xcframework/           (gitignored, 13 MB, manually assembled)
│   ├── libcurl.xcframework/              (gitignored, 2.6 MB, wrapped from cactus libs)
│   ├── libs/                             (gitignored, raw libcurl .a)
│   ├── AGENT_HANDOFF.md                  (original Xcode-GUI walkthrough)
│   └── HANDOFF.md                        (this file — current session state)
├── scripts/bootstrap_phone_app.sh        (committed, fetches iOS source from chicken repo)
├── backend/, dashboard/, mobile/, etc.   (unrelated, not touched this session)
```

## External repos referenced

- `/Users/codykandarian/Desktop/Dev/chicken` — source of iOS Swift code, MP4 clips. On branch `cody/phone-cactus-demo`.
- `/Users/codykandarian/cactus` — cactus library source, build script, vendored libcurl. CMakeLists.txt and 14 kernel files have local mods (not pushed upstream).
