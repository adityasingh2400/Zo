#!/usr/bin/env bash
# scripts/demo_bootstrap.sh — fresh-clone-to-runnable-backend orchestrator.
#
# Checks every prereq the demo needs in dependency order. For things this
# script can safely automate (dep install, env scaffolding) it just runs them.
# For things that need human action (RunPod pod provisioning, .env keys, Xcode
# setup) it prints exact remediation commands and exits non-zero.
#
# Sister script:
#   demo_prewarm.sh — runs AFTER the backend is up, sanity-checks the live
#                     demo path. This script gets you TO the point where
#                     prewarm can run.
#
# Usage:
#   ./scripts/demo_bootstrap.sh             # check + install deps (~2 min fresh)
#   ./scripts/demo_bootstrap.sh --render    # also render local_answers MP4s
#                                           # if missing (~50 min, needs tunnel)
#
# Exit codes:
#   0 — backend is ready to run (you can `make start` next)
#   1 — at least one CRITICAL prereq is missing; see remediation lines above

set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

RENDER=""
[ "${1:-}" = "--render" ] && RENDER=1

PASS=0
FAIL=0
WARN=0
results=()

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; RESET=$'\033[0m'; BOLD=$'\033[1m'
else
  GREEN=""; YELLOW=""; RED=""; RESET=""; BOLD=""
fi

ok()   { results+=("${GREEN}✓${RESET} $1"); PASS=$((PASS+1)); }
fail() { results+=("${RED}✗${RESET} $1"); FAIL=$((FAIL+1)); }
warn() { results+=("${YELLOW}!${RESET} $1"); WARN=$((WARN+1)); }

t0=$(date +%s)

# ── 1. Tooling ──────────────────────────────────────────────────────────────
for tool in python3 node npm ssh curl; do
  if command -v "$tool" >/dev/null 2>&1; then
    ok "$tool installed"
  else
    fail "$tool missing — install via brew (or your package manager) and re-run"
  fi
done

# ── 2. Cactus venv (the on-device LLM runtime) ──────────────────────────────
CACTUS_BIN="$HOME/cactus/venv/bin/cactus"
if [ -x "$CACTUS_BIN" ]; then
  ok "cactus venv at $CACTUS_BIN"
else
  fail "cactus venv missing — install: git clone https://github.com/cactus-compute/cactus ~/cactus && cd ~/cactus && cactus build --python"
fi

# ── 3. Whisper-base weights (on-device transcription) ───────────────────────
WHISPER_DIR="$HOME/cactus/weights/whisper-base"
if [ -d "$WHISPER_DIR" ] && [ -n "$(ls -A "$WHISPER_DIR" 2>/dev/null)" ]; then
  ok "whisper-base weights at $WHISPER_DIR"
else
  fail "whisper-base weights missing — download: $CACTUS_BIN download openai/whisper-base"
fi

# ── 4. .env file + cloud keys ───────────────────────────────────────────────
# Cloud keys (ElevenLabs / RunPod / AWS Bedrock) are WARN-tier, not FAIL-tier:
# per ARCHITECTURE.md the cloud path is a "fast-path upgrade, not a dependency."
# The local-only demo (whisper → router → pre-rendered MP4) needs NONE of these
# at runtime. They only matter for (a) cloud escalate at demo time, and (b)
# re-rendering local_answers from scratch. So missing keys = WARN, not blocker.
if [ -f "$ROOT/.env" ]; then
  ok ".env present"
  set -a; source "$ROOT/.env" 2>/dev/null || true; set +a
  for var in ELEVENLABS_API_KEY RUNPOD_POD_IP AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY; do
    val="${!var:-}"
    if [ -z "$val" ] || [ "$val" = "your_key" ] || [ "$val" = "your_pod_ip" ] || [ "$val" = "your_secret" ]; then
      warn "$var unset — cloud escalate disabled (local demo path still works)"
    else
      ok "$var configured"
    fi
  done
else
  warn ".env missing — copying from .env.example"
  cp "$ROOT/.env.example" "$ROOT/.env"
  fail ".env created from template — edit it (or accept all-defaults for local-only) then re-run"
fi

# ── 5. Backend venv + Python deps ───────────────────────────────────────────
if [ -x "$ROOT/backend/venv/bin/python" ]; then
  ok "backend venv exists"
else
  warn "backend venv missing — running: cd backend && python3 -m venv venv && pip install -r requirements.txt"
  (cd "$ROOT/backend" && python3 -m venv venv && \
    ./venv/bin/pip install --quiet --upgrade pip && \
    ./venv/bin/pip install --quiet -r requirements.txt) \
    && ok "backend deps installed" \
    || fail "backend pip install failed — run scripts/setup.sh manually"
fi

# ── 6. Dashboard node_modules ───────────────────────────────────────────────
if [ -d "$ROOT/dashboard/node_modules" ]; then
  ok "dashboard node_modules present"
else
  warn "dashboard deps missing — running: cd dashboard && npm install"
  (cd "$ROOT/dashboard" && npm install --silent) \
    && ok "dashboard deps installed" \
    || fail "npm install failed — check Node version (need 20+)"
fi

# ── 7. Pod tunnel (Wav2Lip on :8010, LatentSync on :8766) ───────────────────
# Only WARN — the local-only demo path (respond_locally) doesn't need the pod.
# But cloud escalate + first-time render_local_answers does.
if curl -fsS --max-time 2 http://127.0.0.1:8010/health >/dev/null 2>&1; then
  ok "Wav2Lip tunnel up at :8010"
else
  warn "Wav2Lip tunnel down — open in another shell: bash phase0/scripts/open_tunnel.sh  (needs RUNPOD_SSH_KEY + RUNPOD_SSH_PORT in .env)"
fi

# ── 8. Pre-rendered local_answers MP4s ──────────────────────────────────────
LA_DIR="$ROOT/backend/local_answers"
la_count=$(ls "$LA_DIR"/*.mp4 2>/dev/null | wc -l | tr -d ' ')
if [ "${la_count:-0}" -gt 0 ]; then
  ok "local_answers: $la_count MP4(s)"
elif [ -n "$RENDER" ]; then
  warn "no local_answers — rendering now (this takes ~50 min, needs tunnel up)"
  if (cd "$ROOT" && python3 scripts/render_local_answers.py); then
    new_count=$(ls "$LA_DIR"/*.mp4 2>/dev/null | wc -l | tr -d ' ')
    ok "rendered $new_count local_answers MP4(s)"
  else
    fail "render_local_answers.py failed — check tunnel + ELEVENLABS_API_KEY"
  fi
else
  fail "no local_answers — pass --render to generate (~50 min) or copy from another machine"
fi

# ── Report ──────────────────────────────────────────────────────────────────
elapsed=$(($(date +%s)-t0))

echo ""
echo "═══════════════════════════════════════════════════════════════════"
for r in "${results[@]}"; do printf "  %s\n" "$r"; done
echo "═══════════════════════════════════════════════════════════════════"
echo "${BOLD}PASS${RESET}=$PASS  ${BOLD}WARN${RESET}=$WARN  ${BOLD}FAIL${RESET}=$FAIL  (${elapsed}s)"
echo ""

if [ $FAIL -gt 0 ]; then
  printf "%bBOOTSTRAP INCOMPLETE — fix the FAILs above and re-run.%b\n" "$RED$BOLD" "$RESET"
  exit 1
fi

if [ $WARN -gt 0 ]; then
  printf "%bREADY (local-only mode).%b WARNs above are cloud-path features; the local demo (whisper → router → pre-rendered MP4) works without them.\n" "$GREEN$BOLD" "$RESET"
else
  printf "%bREADY (full mode).%b\n" "$GREEN$BOLD" "$RESET"
fi
echo ""
echo "Next:"
echo "  Backend:    make start                       (or: cd backend && source venv/bin/activate && uvicorn main:app --port 8000)"
echo "  Pre-flight: bash scripts/demo_prewarm.sh     (run AFTER the backend is up)"
exit 0
