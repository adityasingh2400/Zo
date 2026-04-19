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
#                                            # comment via the SYNCHRONOUS
#                                            # cloud-escalate endpoint —
#                                            # exercises Bedrock + ElevenLabs
#                                            # + the synthetic
#                                            # comment_response_video
#                                            # contract end-to-end
#                                            # (costs ~$0.00035, takes
#                                            # ~5-10s warm, ~30s cold)
#   ./scripts/demo_prewarm.sh --clean        # prune stale renders +
#                                            # response_audio files (no
#                                            # backend calls). Idempotent;
#                                            # safe to run anytime.
#
# Env overrides:
#   BACKEND_URL  default http://localhost:8000

set -uo pipefail
cd "$(dirname "$0")/.."

BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"
WITH_CLOUD=""
CLEAN_ONLY=""
for arg in "$@"; do
  case "$arg" in
    --with-cloud) WITH_CLOUD=1 ;;
    --clean)      CLEAN_ONLY=1 ;;
    *) printf '\033[31mUnknown flag: %s\033[0m\n' "$arg"; exit 1 ;;
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

# ── --clean fast path: prune stale state, then exit. No backend needed. ─────
# Mirrors what the operator should do post-rehearsal: drop the build-up
# of resp_*.mp4 + response_audio/*.mp3 from prior runs (gitignored, but
# eats disk and clutters log/render dirs). Safe to run anytime.
if [ -n "$CLEAN_ONLY" ]; then
  printf "${BOLD}Pruning stale demo artifacts...${RESET}\n"
  pruned=0

  if [ -d backend/renders ]; then
    n=$(find backend/renders -maxdepth 1 -name 'resp_*.mp4' -type f 2>/dev/null | wc -l | tr -d ' ')
    if [ "${n:-0}" -gt 0 ]; then
      find backend/renders -maxdepth 1 -name 'resp_*.mp4' -type f -delete
      printf "  ${GREEN}✓${RESET} pruned %d resp_*.mp4 from backend/renders/\n" "$n"
      pruned=$((pruned+n))
    else
      printf "  ${GREEN}✓${RESET} backend/renders/ already clean (no resp_*.mp4)\n"
    fi
  fi

  if [ -d backend/response_audio ]; then
    n=$(find backend/response_audio -maxdepth 1 -name 'resp_*.mp3' -type f 2>/dev/null | wc -l | tr -d ' ')
    if [ "${n:-0}" -gt 0 ]; then
      find backend/response_audio -maxdepth 1 -name 'resp_*.mp3' -type f -delete
      printf "  ${GREEN}✓${RESET} pruned %d resp_*.mp3 from backend/response_audio/\n" "$n"
      pruned=$((pruned+n))
    else
      printf "  ${GREEN}✓${RESET} backend/response_audio/ already clean (no resp_*.mp3)\n"
    fi
  fi

  printf "\n${BOLD}Total pruned: %d files${RESET}\n" "$pruned"
  exit 0
fi

# 1. Backend reachable + responsive.
if state=$(curl -fsS "$BACKEND_URL/api/state" --max-time 3 2>/dev/null); then
  ok "backend reachable at $BACKEND_URL"
else
  fail "backend not reachable at $BACKEND_URL — start it: cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --ws-max-size 67108864 --reload"
  state=""
fi

# 2a. products.json is empty — judge-item demo invariant. A pre-loaded
#     product would (a) show its name in the BUY card BEFORE the judge's
#     item gets analyzed and (b) make respond_locally fire wrong
#     pre-rendered answers for any comment that keyword-matches the old
#     product. Both are demo-killers.
products_json="backend/data/products.json"
if [ -f "$products_json" ]; then
  if python3 -c "import json,sys; d=json.load(open('$products_json')); sys.exit(0 if d=={} else 1)" 2>/dev/null; then
    ok "products.json is empty (judge-item demo invariant)"
  else
    pcount=$(python3 -c "import json; print(len(json.load(open('$products_json'))))" 2>/dev/null)
    fail "products.json has $pcount product(s) — empty it for judge-item demo: echo '{}' > $products_json"
  fi
else
  warn "products.json missing — create with: echo '{}' > $products_json"
fi

# 2b. Active product loaded — only matters if we INTENDED to ship with a
#     product (e.g. wallet demo). For judge-item, this MUST be empty.
if [ -n "$state" ]; then
  pname=$(printf '%s' "$state" \
    | python3 -c 'import sys,json
try:
    d=json.load(sys.stdin); pd=d.get("product_data") or {}
    print(pd.get("name") or "")
except Exception: pass' 2>/dev/null)
  if [ -n "$pname" ]; then
    warn "pipeline_state.product_data is populated: \"$pname\" — fine if intentional, but BUY card will show this name pre-analysis"
  else
    ok "pipeline_state.product_data is empty (judge-item ready — populates dynamically when judge's item arrives)"
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

# 4a. Pre-rendered local answers (per-product qa_index responses).
#     For judge-item demos these will be 0 by design — every comment
#     cloud-escalates because we don't pre-know the product.
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
  ok "no per-product local_answers (expected for judge-item demo — every comment routes to cloud)"
fi

# 4b. Generic Wav2Lip clips (intro_arbitrary, bridge_arbitrary).
#     These are the FALLBACK clips that fire when the bridge_clips
#     manifest's primary LatentSync clips aren't rendered yet. Without
#     these AND without LatentSync clips, POST /api/go_live returns 503
#     and the avatar can't acknowledge audience comments with a bridge.
generic_dir="backend/local_answers/_generic"
if [ -d "$generic_dir" ]; then
  generic_manifest="$generic_dir/manifest.json"
  if [ -f "$generic_manifest" ]; then
    gcount=$(python3 -c "
import json
try:
    d = json.load(open('$generic_manifest'))
    print(sum(len(v) for v in d.values() if isinstance(v, list)))
except Exception:
    print(0)
" 2>/dev/null)
    if [ "${gcount:-0}" -gt 0 ]; then
      ok "_generic clips manifest: $gcount entries (Wav2Lip tertiary fallback ready)"
    else
      warn "_generic clips manifest empty — run: python scripts/render_generic_clips.py (needed if no LatentSync intro clips exist)"
    fi
  else
    warn "_generic/manifest.json missing — run: python scripts/render_generic_clips.py"
  fi
else
  warn "_generic clips directory missing — run: python scripts/render_generic_clips.py (Wav2Lip fallback for intro/bridge)"
fi

# 4c. Speaking-idle clip the cloud-escalate path emits as the Tier 1
#     loop. If this 404s, the dashboard's <video> element fails to
#     load and the avatar appears frozen during cloud responses (the
#     audio still plays — captions still render — but the visual is
#     dead).
speaking_idle="/states/idle/idle_calm_speaking.mp4"
si_status=$(curl -sS -o /dev/null -w '%{http_code}' "$BACKEND_URL$speaking_idle" --max-time 3 2>/dev/null)
if [ "$si_status" = "200" ]; then
  ok "speaking-idle clip serves: $speaking_idle (cloud-escalate Tier 1 loop)"
else
  fail "speaking-idle clip $speaking_idle returns $si_status — cloud-escalate avatar will be frozen-but-talking"
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

# 7. CLI tools for QR + tunnel workflow.
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

# 8. Optional: synthetic novel comment, hit the SYNCHRONOUS endpoint so
#    we can assert the response shape (not just that the route exists).
#    Calls /api/respond_to_comment which runs the full Bedrock + ElevenLabs
#    + synthetic comment_response_video pipeline and returns the result
#    body — same shape the dashboard contract tests pin.
if [ -n "$WITH_CLOUD" ]; then
  echo ""
  echo "Firing synthetic novel comment via /api/respond_to_comment (SYNC)..."
  echo "  This warms Bedrock Claude + ElevenLabs TTS + the audio dispatch path"
  echo "  end-to-end. ~5-10s warm, ~30s cold. Costs ~\$0.00035."
  ct0=$(date +%s)
  resp_body=$(curl -fsS -X POST "$BACKEND_URL/api/respond_to_comment" \
       -F 'text=quick prewarm test ignore me' --max-time 60 2>/dev/null)
  cstatus=$?
  cms=$(($(date +%s)-ct0))

  if [ $cstatus -ne 0 ] || [ -z "$resp_body" ]; then
    fail "cloud escalate failed (${cms}s) — check RunPod tunnel + ELEVENLABS_* + AWS_* in backend/.env"
  else
    # Parse the JSON response. Validates the full audio-first cloud path:
    # audio_url present, audio_duration_ms > 0, audio_first=true, lip_synced=false
    # (the no-Wav2Lip judge-item-friendly shape).
    parsed=$(printf '%s' "$resp_body" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    audio_url = d.get('audio_url')
    audio_first = d.get('audio_first')
    lip_synced = d.get('lip_synced')
    dur_ms = d.get('audio_duration_ms') or 0
    if audio_first is not True:
        print('FAIL audio_first not True'); sys.exit(2)
    if lip_synced is not False:
        print('FAIL lip_synced not False'); sys.exit(2)
    if not audio_url or '/response_audio/' not in audio_url:
        print('FAIL missing/bad audio_url:', audio_url); sys.exit(2)
    if dur_ms < 500:
        print('FAIL audio_duration_ms too short:', dur_ms); sys.exit(2)
    print(f'audio_url={audio_url} dur={dur_ms}ms')
    sys.exit(0)
except Exception as e:
    print('FAIL parse:', e); sys.exit(2)
" 2>&1)
    pstatus=$?
    if [ $pstatus -eq 0 ]; then
      ok "cloud escalate end-to-end: ${cms}s, $parsed"
    else
      fail "cloud escalate response shape WRONG (${cms}s): $parsed"
    fi
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
  printf "%bDEMO PARTIAL — WARNs above are nice-to-have. Continue if intentional.%b\n" "$YELLOW" "$RESET"
else
  printf "%bDEMO READY — break a leg.%b\n" "$GREEN$BOLD" "$RESET"
fi
echo ""
echo "Operator next steps:"
echo "  1. Audience tunnel + QR:    ./scripts/start_audience_tunnel.sh"
echo "  2. Open stage view:         http://localhost:5173/stage   (then press F for fullscreen)"
echo "  3. Reset cost ticker:       press R inside /stage"
echo "  4. Fire intro clip:         press G inside /stage"
echo ""
echo "Post-rehearsal cleanup:        ./scripts/demo_prewarm.sh --clean"
echo "Full demo runbook:             cat STAGE_RUNBOOK.md"
exit 0
