#!/usr/bin/env bash
# scripts/demo_prewarm.sh — pre-stage sanity check.
#
# Run this 5 minutes before walking on stage. Exit 0 if the demo path is
# hot enough to ship, exit 1 if anything CRITICAL is missing. WARN-level
# items (missing cloudflared, no pre-rendered local answers) don't fail
# the script — they're nice-to-haves the operator may have intentionally
# skipped.
#
# Usage:
#   ./scripts/demo_prewarm.sh                # local-only checks (~3s)
#   ./scripts/demo_prewarm.sh --with-cloud   # also fire a synthetic novel
#                                            # comment to warm Wav2Lip
#                                            # (costs ~$0.00035, takes
#                                            # ~5-10s warm, ~30s cold)
#   ./scripts/demo_prewarm.sh --stage        # full stage-gate: all checks +
#                                            # BRAIN pre-fire (20 seeded
#                                            # comments). WARNs still show
#                                            # but don't block; FAILs do.
#                                            # Run this ~5 min before stage.
#
# Env overrides:
#   BACKEND_URL  default http://localhost:8000

set -uo pipefail
cd "$(dirname "$0")/.."

# Source .env so the aws CLI + ElevenLabs probe see the same creds the
# backend's boto3 + python-dotenv see. `set -a` auto-exports everything
# assigned during sourcing (required — aws CLI reads from environment).
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"
WITH_CLOUD=""
STAGE=""
for arg in "$@"; do
  case "$arg" in
    --with-cloud) WITH_CLOUD=1 ;;
    --stage)      STAGE=1 ;;
    *) ;;
  esac
done

PASS=0
FAIL=0
WARN=0
results=()

# Color helpers — degrade gracefully if NO_COLOR or non-tty.
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; RESET=$'\033[0m'; BOLD=$'\033[1m'
else
  GREEN=""; YELLOW=""; RED=""; RESET=""; BOLD=""
fi

ok()   { results+=("${GREEN}✓${RESET} $1"); PASS=$((PASS+1)); }
fail() { results+=("${RED}✗${RESET} $1"); FAIL=$((FAIL+1)); }
warn() { results+=("${YELLOW}!${RESET} $1"); WARN=$((WARN+1)); }

t0=$(date +%s)

# 1. Backend reachable + responsive.
if state=$(curl -fsS "$BACKEND_URL/api/state" --max-time 3 2>/dev/null); then
  ok "backend reachable at $BACKEND_URL"
else
  fail "backend not reachable at $BACKEND_URL — start it: cd backend && uvicorn main:app --reload"
  state=""
fi

# 2. Active product loaded — router needs product.qa_index for respond_locally.
if [ -n "$state" ]; then
  pname=$(printf '%s' "$state" \
    | python3 -c 'import sys,json
try:
    d=json.load(sys.stdin); pd=d.get("product_data") or {}
    print(pd.get("name") or "")
except Exception: pass' 2>/dev/null)
  if [ -n "$pname" ]; then
    ok "active product loaded: \"$pname\""
  else
    warn "no active product_data — set ACTIVE_PRODUCT_ID or POST /api/sell first (router will cloud-escalate everything)"
  fi
fi

# 3. Bridge clips manifest populated — Director needs these for bridges + intro.
if bridges=$(curl -fsS "$BACKEND_URL/api/bridges" --max-time 3 2>/dev/null); then
  bcount=$(printf '%s' "$bridges" \
    | python3 -c 'import sys,json
try:
    d=json.load(sys.stdin) or {}
    print(sum(len(v) for v in d.values() if isinstance(v, list)))
except Exception: print(0)' 2>/dev/null)
  if [ "${bcount:-0}" -gt 0 ]; then
    ok "bridge clips manifest: $bcount entries"
  else
    warn "bridge clips manifest empty — run: python -m backend.agents.bridge_clips render  (LatentSync, slow) or scripts/render_generic_clips.py (Wav2Lip, fast)"
  fi
fi

# 4. Pre-rendered local answers + warm the file cache by HEAD'ing 3.
local_count=$(ls backend/local_answers/*.mp4 2>/dev/null | wc -l | tr -d ' ')
if [ "${local_count:-0}" -gt 0 ]; then
  ok "local_answers: $local_count MP4(s)"
  for f in $(ls backend/local_answers/*.mp4 2>/dev/null | head -3); do
    name=$(basename "$f")
    if curl -fsSI "$BACKEND_URL/local_answers/$name" --max-time 3 >/dev/null 2>&1; then
      ok "warmed /local_answers/$name"
    else
      warn "could not HEAD /local_answers/$name"
    fi
  done
else
  warn "no pre-rendered local answers — router will cloud-escalate every product question (run: python scripts/render_local_answers.py)"
fi

# 5. Audience comment intake routes are wired. Use GET (not HEAD) — FastAPI
#    doesn't auto-route HEAD to GET handlers, so HEAD returns 405 here.
comment_code=$(curl -s -o /dev/null -w "%{http_code}" "$BACKEND_URL/comment" --max-time 3)
if [ "$comment_code" = "200" ]; then
  ok "/comment form serves (audience QR target)"
else
  fail "/comment form not serving (HTTP $comment_code) — backend may be stale (older build)"
fi

# 6. Stage hotkey endpoint is wired. Drop -f so curl doesn't abort on 503
#    (which is a legitimate "endpoint exists but no clips yet" response).
go_live_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BACKEND_URL/api/go_live" --max-time 3)
if [ "$go_live_code" = "200" ] || [ "$go_live_code" = "503" ]; then
  # 503 is fine — endpoint wired but no clips rendered yet (step 3 flags it).
  ok "POST /api/go_live wired (G hotkey target, HTTP $go_live_code)"
else
  fail "POST /api/go_live not wired (HTTP $go_live_code) — backend may be stale"
fi

# 7. CLI tools for QR + tunnel workflow. Only run in non-stage mode — the
#    hackathon stage demo drives comments from the iPhone directly, so the
#    audience QR is irrelevant and these binaries being absent shouldn't
#    block --stage. Keep the checks in normal mode for audience-QR demos.
if [ -z "$STAGE" ]; then
  if command -v cloudflared >/dev/null 2>&1; then
    ok "cloudflared installed (audience tunnel ready)"
  else
    warn "cloudflared missing — brew install cloudflared (audience QR won't work)"
  fi
  if command -v qrencode >/dev/null 2>&1; then
    ok "qrencode installed (QR PNG generation ready)"
  else
    warn "qrencode missing — brew install qrencode (QR PNG won't generate; manual fallback works)"
  fi
fi

# 8. Optional: synthetic novel comment to warm cloud path end-to-end.
if [ -n "$WITH_CLOUD" ]; then
  echo ""
  echo "Firing synthetic novel comment ('quick prewarm test, ignore') to warm Wav2Lip + Bedrock + ElevenLabs..."
  ct0=$(date +%s)
  if curl -fsS -X POST "$BACKEND_URL/api/comment" \
       -F 'text=quick prewarm test ignore me' --max-time 60 >/dev/null 2>&1; then
    ok "cloud escalate end-to-end: $(($(date +%s)-ct0))s warm path"
  else
    fail "cloud escalate failed (check RunPod tunnel + ELEVENLABS_* + AWS_* in backend/.env)"
  fi
fi

# 9. End-to-end local-route smoke. Hits the on-device classify endpoint with
#    a known-good wallet question, then HEADs the specific MP4 that the
#    router's keyword matcher WOULD dispatch. Catches:
#      - Gemma classify dead / Cactus model not loaded
#      - products.json edited (new qa_index entry) but matching MP4 not rendered
#    Costs ~1-2s to cold-load Gemma if backend just started — that's the point;
#    we want the first real comment to be warm.
known_local_comment="is it real leather"
expected_mp4="/local_answers/wallet_real_leather.mp4"
if classify=$(curl -fsS -X POST "$BACKEND_URL/api/classify_comment" \
                -F "comment=$known_local_comment" --max-time 8 2>/dev/null); then
  label=$(printf '%s' "$classify" \
    | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("label",""))
except Exception: pass' 2>/dev/null)
  if [ "$label" = "question" ]; then
    ok "classify(\"$known_local_comment\") → question (Gemma warm)"
  else
    warn "classify(\"$known_local_comment\") → \"$label\" (expected question)"
  fi
  if curl -fsSI "$BACKEND_URL$expected_mp4" --max-time 3 >/dev/null 2>&1; then
    ok "$expected_mp4 reachable (router would dispatch this for \"$known_local_comment\")"
  else
    fail "$expected_mp4 missing — products.json references it but file not rendered (re-run scripts/render_local_answers.py)"
  fi
else
  fail "/api/classify_comment failed — Cactus may not be loaded (check backend logs)"
fi

# 10. AWS Bedrock creds valid for the cloud-escalate path. Uses the same
#     boto3 the backend uses (from backend/venv) so we're probing with the
#     exact library + auth resolution order that matters. bedrock:List (not
#     bedrock-runtime) is free metadata — doesn't spend tokens.
bedrock_region="${AWS_REGION:-us-east-1}"
bedrock_py="backend/venv/bin/python"
if [ -x "$bedrock_py" ]; then
  if err=$("$bedrock_py" - <<PY 2>&1 >/dev/null
import sys
try:
    import boto3
    boto3.client("bedrock", region_name="$bedrock_region").list_foundation_models()
except Exception as e:
    print(f"{type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
PY
  ); then
    ok "Bedrock creds valid ($bedrock_region)"
  else
    fail "Bedrock creds invalid ($bedrock_region) — fix AWS_* in .env. boto3 said: ${err:-<empty>}"
  fi
else
  warn "backend/venv missing — cannot verify Bedrock creds; run: cd backend && python3 -m venv venv && ./venv/bin/pip install -r requirements.txt"
fi

# 11. ElevenLabs key valid. /v1/voices is a free metadata call. ElevenLabs
#     returns 401 for BOTH "invalid key" and "valid key, missing scope" —
#     the body's `status` field is the only way to tell them apart:
#       "missing_permissions" → key valid, scope missing (fine; Seller uses
#                                the tts scope, not voices_read)
#       anything else on 401  → truly invalid key
#     This is why we can't just check the HTTP code.
if [ -n "${ELEVENLABS_API_KEY:-}" ]; then
  el_resp=$(curl -s -w "\n__HTTP__:%{http_code}" \
    -H "xi-api-key: ${ELEVENLABS_API_KEY}" \
    "https://api.elevenlabs.io/v1/voices" --max-time 5 2>/dev/null || printf '\n__HTTP__:000')
  el_code=$(printf '%s' "$el_resp" | sed -n 's/.*__HTTP__:\([0-9]*\).*/\1/p' | tail -n1)
  el_body=$(printf '%s' "$el_resp" | sed 's/__HTTP__:[0-9]*//')
  case "$el_code" in
    200)
      ok "ElevenLabs key valid (full voices_read scope)"
      ;;
    401|403)
      if printf '%s' "$el_body" | grep -qi 'missing_permissions'; then
        ok "ElevenLabs key valid (voices_read missing — Seller uses the tts scope, which is separate)"
      else
        fail "ElevenLabs key invalid (HTTP $el_code) — fix ELEVENLABS_API_KEY in .env"
      fi
      ;;
    000)
      fail "ElevenLabs unreachable — check network / api.elevenlabs.io status"
      ;;
    *)
      warn "ElevenLabs returned HTTP $el_code (unexpected; key may still work for TTS)"
      ;;
  esac
else
  warn "ELEVENLABS_API_KEY not set — Seller cloud-escalate TTS won't work"
fi

# 12. STAGE MODE: warm Gemma + populate BRAIN by pre-firing comments. This
#     makes the BRAIN panel show real stats from the moment the dashboard
#     opens, so the moat narrative is visible before the first on-stage
#     comment lands. Only runs with --stage.
if [ -n "$STAGE" ]; then
  echo ""
  echo "--stage: pre-firing BRAIN via scripts/prefire_brain.sh..."
  if BACKEND_URL="$BACKEND_URL" bash scripts/prefire_brain.sh >/dev/null 2>&1; then
    brain_total=$(curl -fsS "$BACKEND_URL/api/brain/stats" --max-time 3 2>/dev/null \
      | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("total",0))
except Exception: print(0)' 2>/dev/null)
    if [ "${brain_total:-0}" -ge 20 ]; then
      ok "BRAIN pre-fire: ${brain_total} events logged"
    else
      fail "BRAIN pre-fire ran but only ${brain_total} events — expected ≥20 (check prefire_brain.sh output)"
    fi
  else
    fail "scripts/prefire_brain.sh failed — run it manually to see the error"
  fi
fi

elapsed=$(($(date +%s)-t0))

echo ""
echo "═══════════════════════════════════════════════════════════════════"
for r in "${results[@]}"; do
  printf "  %s\n" "$r"
done
echo "═══════════════════════════════════════════════════════════════════"
echo "${BOLD}PASS${RESET}=$PASS  ${BOLD}WARN${RESET}=$WARN  ${BOLD}FAIL${RESET}=$FAIL  (${elapsed}s)"
echo ""

if [ $FAIL -gt 0 ]; then
  printf "%bDEMO NOT READY — fix the FAILs before walking on stage.%b\n" "$RED$BOLD" "$RESET"
  exit 1
fi

if [ $WARN -gt 0 ]; then
  if [ -n "$STAGE" ]; then
    printf "%bSTAGE READY — %d WARN(s) above are non-blocking for iPhone-driven pitch. Review + continue.%b\n" "$YELLOW" "$WARN" "$RESET"
  else
    printf "%bDEMO PARTIAL — WARNs above are nice-to-have. Continue if intentional.%b\n" "$YELLOW" "$RESET"
  fi
else
  printf "%bDEMO READY — break a leg.%b\n" "$GREEN$BOLD" "$RESET"
fi
echo ""
echo "Operator next steps:"
echo "  1. Audience tunnel + QR:    ./scripts/start_audience_tunnel.sh"
echo "  2. Open stage view:         http://localhost:5173/stage   (then press F for fullscreen)"
echo "  3. Reset cost ticker:       press R inside /stage"
echo "  4. Fire intro clip:         press G inside /stage"
exit 0
