#!/usr/bin/env bash
# ============================================================================
# bootstrap_phone_app.sh
#
# One-shot setup for the EmpirePhone iOS app in a fresh repo. Pulls the Swift
# source I wrote from a public GitHub branch, builds the Cactus iOS
# xcframework, stages model weights, and copies the pre-rendered MP4 clips
# into the right place. Leaves an AGENT_HANDOFF.md with the Xcode-side manual
# steps (project creation + signing) that need a human UI.
#
# USAGE
#   cd /path/to/your/new/repo
#   bash scripts/bootstrap_phone_app.sh
#
# FLAGS
#   --old-repo PATH      Local path to the chicken repo that already has the
#                        MP4 source clips under backend/local_answers/ and
#                        phase0/assets/clips/. Default: ../chicken
#   --cactus-repo PATH   Local path to the Cactus git repo. Default: ~/cactus
#   --skip-xcframework   Skip building cactus-ios.xcframework (useful if you
#                        already have one from another repo)
#   --skip-weights       Skip downloading whisper-tiny via `cactus download`
#   --skip-clips         Skip copying MP4 clips from the old repo
#   --ios-branch BRANCH  Git branch on adityasingh2400/chicken to pull the
#                        ios/ source from. Default: cody/phone-cactus-demo
#   --verbose            Echo every command before running it
#   -h, --help           Show this help
#
# EXIT CODES
#   0  success
#   1  preflight failure (missing Xcode / cmake / git / etc.)
#   2  source fetch failure
#   3  xcframework build failure
#   4  weight download failure
#   5  clip copy failure
#
# WHAT THIS SCRIPT DOES NOT DO
#   - Create the .xcodeproj file. Xcode's "New Project" wizard writes an
#     internally-consistent project file; hand-editing pbxproj is error-prone.
#     The AGENT_HANDOFF.md output walks your agent through the 6-minute Xcode
#     UI step.
#   - Sign the app with your Apple ID. That's a Xcode UI action — your agent
#     does it after opening the project.
#   - Install Xcode. If it's missing the script bails with a clear error.
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------- constants
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly IOS_SOURCE_URL_DEFAULT="https://github.com/adityasingh2400/chicken.git"
readonly IOS_SOURCE_BRANCH_DEFAULT="cody/phone-cactus-demo"
readonly OLD_REPO_DEFAULT="$(cd "$REPO_ROOT/.." && pwd)/chicken"
readonly CACTUS_REPO_DEFAULT="$HOME/cactus"
readonly WHISPER_MODEL="openai/whisper-tiny"
readonly WHISPER_DIR_NAME="whisper-tiny"

# ---------------------------------------------------------------- args
OLD_REPO="$OLD_REPO_DEFAULT"
CACTUS_REPO="$CACTUS_REPO_DEFAULT"
IOS_SOURCE_URL="$IOS_SOURCE_URL_DEFAULT"
IOS_SOURCE_BRANCH="$IOS_SOURCE_BRANCH_DEFAULT"
SKIP_XCFRAMEWORK=0
SKIP_WEIGHTS=0
SKIP_CLIPS=0
VERBOSE=0

# ---------------------------------------------------------------- colors
if [ -t 1 ] && command -v tput >/dev/null 2>&1; then
    readonly BOLD="$(tput bold)"
    readonly DIM="$(tput dim)"
    readonly RED="$(tput setaf 1)"
    readonly GREEN="$(tput setaf 2)"
    readonly YELLOW="$(tput setaf 3)"
    readonly BLUE="$(tput setaf 4)"
    readonly RESET="$(tput sgr0)"
else
    readonly BOLD="" DIM="" RED="" GREEN="" YELLOW="" BLUE="" RESET=""
fi

# ---------------------------------------------------------------- helpers
log_phase() { printf '\n%s════ %s ════%s\n' "$BOLD$BLUE" "$1" "$RESET"; }
log_info()  { printf '%s›%s %s\n' "$DIM" "$RESET" "$1"; }
log_ok()    { printf '%s✓%s %s\n' "$GREEN" "$RESET" "$1"; }
log_warn()  { printf '%s!%s %s\n' "$YELLOW" "$RESET" "$1"; }
log_err()   { printf '%s✗%s %s\n' "$RED" "$RESET" "$1" >&2; }
die()       { log_err "$1"; exit "${2:-1}"; }

run() {
    [ "$VERBOSE" -eq 1 ] && printf '%s$ %s%s\n' "$DIM" "$*" "$RESET" >&2
    "$@"
}

usage() {
    sed -n '/^# USAGE/,/^# EXIT CODES/p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

# ---------------------------------------------------------------- arg parse
while [ $# -gt 0 ]; do
    case "$1" in
        --old-repo)        OLD_REPO="$2"; shift 2 ;;
        --cactus-repo)     CACTUS_REPO="$2"; shift 2 ;;
        --ios-branch)      IOS_SOURCE_BRANCH="$2"; shift 2 ;;
        --skip-xcframework) SKIP_XCFRAMEWORK=1; shift ;;
        --skip-weights)    SKIP_WEIGHTS=1; shift ;;
        --skip-clips)      SKIP_CLIPS=1; shift ;;
        --verbose|-v)      VERBOSE=1; shift ;;
        -h|--help)         usage ;;
        *)                 die "Unknown flag: $1  (try --help)" 1 ;;
    esac
done

# ==================================================== PHASE 1 — PREFLIGHT ====
phase_preflight() {
    log_phase "PHASE 1 / 6 — Preflight"

    # macOS only
    [ "$(uname -s)" = "Darwin" ] || die "This script only runs on macOS (iOS dev host)." 1

    # Xcode full install (not just Command Line Tools)
    if ! xcode-select -p >/dev/null 2>&1; then
        die "Xcode command line tools not found. Install with: xcode-select --install" 1
    fi
    local active
    active="$(xcode-select -p)"
    if [[ "$active" == *CommandLineTools* ]]; then
        die "Only Command Line Tools are active. Install the full Xcode app from the Mac App Store, then run: sudo xcode-select -s /Applications/Xcode.app/Contents/Developer" 1
    fi
    log_ok "Xcode active at: $active"

    # xcodebuild sanity
    if ! xcodebuild -version >/dev/null 2>&1; then
        die "xcodebuild not working. Open Xcode.app once to accept the license, then re-run." 1
    fi
    log_ok "$(xcodebuild -version | head -1)"

    # cmake (for building the xcframework)
    if [ "$SKIP_XCFRAMEWORK" -eq 0 ]; then
        command -v cmake >/dev/null 2>&1 || die "cmake not found. Install with: brew install cmake" 1
        log_ok "cmake: $(cmake --version | head -1)"
    fi

    # git (for fetching ios/ source)
    command -v git >/dev/null 2>&1 || die "git not found. Install Xcode or Homebrew git." 1

    # We need to be inside a repo root (or at least a writeable dir)
    [ -w "$REPO_ROOT" ] || die "Repo root $REPO_ROOT is not writeable." 1
    log_ok "Repo root: $REPO_ROOT"

    # Cactus CLI (via the Python venv Cody's build uses)
    if [ "$SKIP_WEIGHTS" -eq 0 ]; then
        [ -d "$CACTUS_REPO" ] || die "Cactus repo not found at $CACTUS_REPO. Pass --cactus-repo or clone: git clone https://github.com/cactus-compute/cactus $CACTUS_REPO" 1
        [ -x "$CACTUS_REPO/venv/bin/cactus" ] || die "Cactus CLI not found at $CACTUS_REPO/venv/bin/cactus. Run cactus's setup first: cd $CACTUS_REPO && source ./setup && cd .. && cactus build --python" 1
        log_ok "Cactus CLI: $CACTUS_REPO/venv/bin/cactus"
    fi

    # Old repo for MP4 source clips
    if [ "$SKIP_CLIPS" -eq 0 ]; then
        [ -d "$OLD_REPO" ] || die "Old chicken repo not found at $OLD_REPO. Pass --old-repo /path/to/chicken, or run with --skip-clips and copy MP4s manually later." 1
        [ -d "$OLD_REPO/backend/local_answers" ] || die "$OLD_REPO/backend/local_answers/ not found. Did the old repo check out with local_answers gitignored? Regenerate via: cd $OLD_REPO && python scripts/render_local_answers.py" 1
        log_ok "Old repo clip sources: $OLD_REPO"
    fi
}

# =============================================== PHASE 2 — FETCH IOS SOURCE ===
phase_fetch_source() {
    log_phase "PHASE 2 / 6 — Fetch iOS Swift source"

    local target="$REPO_ROOT/ios"
    if [ -d "$target" ] && [ -n "$(ls -A "$target" 2>/dev/null || true)" ]; then
        log_warn "ios/ already has contents; skipping source fetch. Delete ios/ first if you want a clean pull."
        return
    fi

    # Use git sparse-checkout to pull only the ios/ path from the branch.
    local tmp
    tmp="$(mktemp -d -t empire-ios-XXXX)"
    trap "rm -rf '$tmp'" EXIT

    log_info "Cloning $IOS_SOURCE_URL @ $IOS_SOURCE_BRANCH (sparse: ios/)…"
    run git clone --depth 1 --branch "$IOS_SOURCE_BRANCH" --filter=blob:none --sparse \
        "$IOS_SOURCE_URL" "$tmp/repo" >/dev/null 2>&1 \
        || die "git clone failed. If the branch has moved, pass --ios-branch <name>." 2

    run git -C "$tmp/repo" sparse-checkout set ios >/dev/null
    [ -d "$tmp/repo/ios" ] || die "Branch $IOS_SOURCE_BRANCH has no ios/ directory." 2

    mkdir -p "$target"
    run rsync -a "$tmp/repo/ios/" "$target/"

    # Verify key files landed
    local required=(
        "$target/Cactus.swift"
        "$target/EmpirePhone/EmpirePhoneApp.swift"
        "$target/EmpirePhone/ContentView.swift"
        "$target/EmpirePhone/CactusRunner.swift"
        "$target/EmpirePhone/AudioRecorder.swift"
        "$target/EmpirePhone/Router.swift"
        "$target/EmpirePhone/VideoDirector.swift"
        "$target/EmpirePhone/Info.plist"
        "$target/EmpirePhone/products.json"
        "$target/.gitignore"
        "$target/README.md"
    )
    local missing=0
    for f in "${required[@]}"; do
        if [ ! -f "$f" ]; then
            log_err "missing: $f"
            missing=1
        fi
    done
    [ "$missing" -eq 0 ] || die "Source fetch was incomplete." 2

    log_ok "iOS source in place at $target"
    rm -rf "$tmp"; trap - EXIT
}

# =========================================== PHASE 3 — BUILD IOS XCFRAMEWORK ==
phase_build_xcframework() {
    log_phase "PHASE 3 / 6 — Build cactus-ios.xcframework"

    local prebuilt="$CACTUS_REPO/apple/cactus-ios.xcframework"
    local target="$REPO_ROOT/ios/cactus-ios.xcframework"

    if [ "$SKIP_XCFRAMEWORK" -eq 1 ]; then
        log_warn "Skipping xcframework build (--skip-xcframework). Make sure $target exists before opening Xcode."
        return
    fi

    if [ -d "$prebuilt" ]; then
        log_info "Found existing xcframework in Cactus repo; reusing."
    else
        log_info "Building xcframework via $CACTUS_REPO/apple/build.sh (5-10 min)…"
        [ -x "$CACTUS_REPO/apple/build.sh" ] || die "$CACTUS_REPO/apple/build.sh not executable." 3
        (cd "$CACTUS_REPO/apple" && run bash ./build.sh) || die "xcframework build failed." 3
    fi

    [ -d "$prebuilt" ] || die "Build finished but xcframework not found at $prebuilt." 3

    rm -rf "$target"
    run cp -R "$prebuilt" "$target"
    log_ok "xcframework staged at $target"
}

# ================================================= PHASE 4 — WHISPER WEIGHTS ==
phase_weights() {
    log_phase "PHASE 4 / 6 — Whisper weights"

    local weights_dst="$REPO_ROOT/ios/EmpirePhone/Models/$WHISPER_DIR_NAME"
    local weights_src="$CACTUS_REPO/weights/$WHISPER_DIR_NAME"

    if [ "$SKIP_WEIGHTS" -eq 1 ]; then
        log_warn "Skipping whisper download (--skip-weights). Make sure $weights_dst exists before building."
        return
    fi

    if [ ! -d "$weights_src" ]; then
        log_info "Downloading $WHISPER_MODEL via cactus CLI…"
        (cd "$CACTUS_REPO" && run "$CACTUS_REPO/venv/bin/cactus" download "$WHISPER_MODEL") \
            || die "cactus download failed." 4
    else
        log_info "Weights already downloaded at $weights_src; reusing."
    fi

    [ -d "$weights_src" ] || die "Expected weights at $weights_src but directory missing." 4

    mkdir -p "$REPO_ROOT/ios/EmpirePhone/Models"
    rm -rf "$weights_dst"
    run cp -R "$weights_src" "$weights_dst"
    local size
    size="$(du -sh "$weights_dst" | awk '{print $1}')"
    log_ok "Whisper weights staged at $weights_dst ($size)"
}

# =================================================== PHASE 5 — MP4 CLIPS ======
phase_clips() {
    log_phase "PHASE 5 / 6 — Pre-rendered MP4 clips"

    local clips_dst="$REPO_ROOT/ios/EmpirePhone/Clips"

    if [ "$SKIP_CLIPS" -eq 1 ]; then
        log_warn "Skipping clip copy (--skip-clips). Make sure $clips_dst/ has the MP4s before building."
        return
    fi

    mkdir -p "$clips_dst"

    # Wallet answer clips (all 10 that our products.json references).
    local answers_src="$OLD_REPO/backend/local_answers"
    local expected_answers=(
        wallet_real_leather.mp4 wallet_shipping.mp4 wallet_returns.mp4
        wallet_warranty.mp4 wallet_sizing.mp4 wallet_colors.mp4
        wallet_price.mp4 wallet_cards.mp4 wallet_water.mp4 wallet_rfid.mp4
    )
    local missing=0
    for f in "${expected_answers[@]}"; do
        if [ ! -f "$answers_src/$f" ]; then
            log_err "missing: $answers_src/$f"
            missing=1
        fi
    done
    if [ "$missing" -eq 1 ]; then
        die "One or more expected wallet answer clips are missing. Regenerate them in the old repo: cd $OLD_REPO && python scripts/render_local_answers.py" 5
    fi
    for f in "${expected_answers[@]}"; do
        run cp "$answers_src/$f" "$clips_dst/$f"
    done

    # Bridge clips (canned acknowledgements for compliment / objection).
    local bridges_src="$OLD_REPO/phase0/assets/clips"
    if [ ! -d "$bridges_src" ]; then
        log_warn "$bridges_src not found — skipping bridge clips. play_canned_clip routes will fall through to the thinking placeholder."
    else
        # Copy the first 3 compliment clips and 2 objection clips we can find,
        # renaming to the bundle-friendly scheme VideoDirector.swift looks for.
        local comp_idx=0 obj_idx=0
        shopt -s nullglob
        for src in "$bridges_src"/compliment_*.mp4; do
            comp_idx=$((comp_idx + 1))
            [ "$comp_idx" -le 3 ] || break
            run cp "$src" "$clips_dst/bridge_compliment_${comp_idx}.mp4"
        done
        for src in "$bridges_src"/objection_*.mp4; do
            obj_idx=$((obj_idx + 1))
            [ "$obj_idx" -le 2 ] || break
            run cp "$src" "$clips_dst/bridge_objection_${obj_idx}.mp4"
        done
        shopt -u nullglob
        log_ok "Staged $comp_idx compliment + $obj_idx objection bridge clips"
    fi

    # Idle loop substrate.
    local idle_src="$OLD_REPO/phase0/assets/states/idle/idle_calm.mp4"
    if [ -f "$idle_src" ]; then
        run cp "$idle_src" "$clips_dst/idle_loop.mp4"
    else
        log_warn "$idle_src not found — the phone will show a black avatar frame when idle. Copy any idle speaking MP4 to $clips_dst/idle_loop.mp4 to fix."
    fi

    local total
    total="$(find "$clips_dst" -name '*.mp4' | wc -l | tr -d ' ')"
    local size
    size="$(du -sh "$clips_dst" | awk '{print $1}')"
    log_ok "$total clips staged in $clips_dst ($size)"
}

# ============================================ PHASE 6 — AGENT HANDOFF DOC =====
phase_handoff() {
    log_phase "PHASE 6 / 6 — Write AGENT_HANDOFF.md"

    local handoff="$REPO_ROOT/ios/AGENT_HANDOFF.md"
    cat > "$handoff" <<'EOF'
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
EOF

    log_ok "Wrote $handoff"
}

# ============================================================ SUMMARY =========
phase_summary() {
    log_phase "DONE"
    printf '\n%sEmpirePhone scaffolding is ready.%s\n' "$BOLD" "$RESET"
    printf '\n'
    printf 'Tree:\n'
    printf '  %sios/%s\n' "$BOLD" "$RESET"
    printf '  ├── %sAGENT_HANDOFF.md%s  %s← open this in your agent first%s\n' "$GREEN" "$RESET" "$DIM" "$RESET"
    printf '  ├── README.md\n'
    printf '  ├── Cactus.swift\n'
    printf '  ├── cactus-ios.xcframework/  (from ~/cactus/apple/build.sh)\n'
    printf '  └── EmpirePhone/\n'
    printf '      ├── *.swift  (6 files — Router, AudioRecorder, CactusRunner, VideoDirector, ContentView, EmpirePhoneApp)\n'
    printf '      ├── Info.plist\n'
    printf '      ├── products.json\n'
    printf '      ├── Clips/   (16 MP4s, ~65 MB)\n'
    printf '      └── Models/  (whisper-tiny, ~120 MB)\n'
    printf '\n'
    printf '%sNext:%s\n' "$BOLD" "$RESET"
    printf '  1. Open %s%s/ios/AGENT_HANDOFF.md%s in your Xcode-capable coding agent.\n' "$BLUE" "$REPO_ROOT" "$RESET"
    printf '  2. The agent completes the 8 Xcode GUI steps (~10 min).\n'
    printf '  3. Phone is unlocked + trusted to this Mac.\n'
    printf '  4. Cmd+R in Xcode, trust cert on phone, demo.\n'
    printf '\n'
}

main() {
    log_info "EmpirePhone bootstrap starting"
    log_info "  repo root:    $REPO_ROOT"
    log_info "  old repo:     $OLD_REPO"
    log_info "  cactus repo:  $CACTUS_REPO"
    log_info "  ios branch:   $IOS_SOURCE_BRANCH"

    phase_preflight
    phase_fetch_source
    phase_build_xcframework
    phase_weights
    phase_clips
    phase_handoff
    phase_summary
}

main "$@"
