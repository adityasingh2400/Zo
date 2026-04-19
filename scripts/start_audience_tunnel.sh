#!/usr/bin/env bash
# scripts/start_audience_tunnel.sh
# Bring up the audience-facing comment intake.
#
# Boots a Cloudflare quick-tunnel pointing at the backend (default
# http://localhost:8000), captures the trycloudflare URL, generates a QR
# PNG that points at /comment, and prints both the URL and a terminal-
# rendered QR. Tunnel runs in the foreground — Ctrl-C to tear it down.
#
# Stage usage:
#   1. Run this in a separate terminal before going on stage.
#   2. Drop the printed QR PNG into the intro slide.
#   3. Tape a printout of the same QR to the back of the demo laptop as
#      a fallback — projectors fail, audience phones don't.
#
# Env overrides:
#   BACKEND_PORT  default 8000
#   QR_PNG        default /tmp/empire_audience_qr.png

set -uo pipefail
cd "$(dirname "$0")/.."

BACKEND_PORT="${BACKEND_PORT:-8000}"
QR_PNG="${QR_PNG:-/tmp/empire_audience_qr.png}"

if ! command -v cloudflared >/dev/null 2>&1; then
  printf '\033[31m✗\033[0m cloudflared not installed.\n'
  echo "  Install: brew install cloudflared"
  exit 1
fi

HAS_QRENCODE=1
if ! command -v qrencode >/dev/null 2>&1; then
  HAS_QRENCODE=0
  printf '\033[33m!\033[0m qrencode not installed — terminal QR + PNG will be skipped.\n'
  echo "  Install: brew install qrencode"
fi

TUNNEL_LOG=$(mktemp -t empire_tunnel.XXXXXX)
trap 'cleanup' EXIT INT TERM
cleanup() {
  if [ -n "${TUNNEL_PID:-}" ] && kill -0 "$TUNNEL_PID" 2>/dev/null; then
    echo ""
    echo "Tearing down tunnel (pid=$TUNNEL_PID)..."
    kill "$TUNNEL_PID" 2>/dev/null || true
    wait "$TUNNEL_PID" 2>/dev/null || true
  fi
  rm -f "$TUNNEL_LOG"
}

echo "Starting Cloudflare quick tunnel → http://localhost:${BACKEND_PORT}"
cloudflared tunnel --url "http://localhost:${BACKEND_PORT}" --no-autoupdate \
  > "$TUNNEL_LOG" 2>&1 &
TUNNEL_PID=$!

# Cloudflared prints the trycloudflare URL within a few seconds. Poll for
# up to 30s before giving up — slow networks sometimes need it.
URL=""
for _ in $(seq 1 60); do
  URL=$(grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" 2>/dev/null | head -1)
  [ -n "$URL" ] && break
  if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
    echo ""
    printf '\033[31m✗\033[0m cloudflared exited before producing a tunnel URL. Last log lines:\n'
    tail -20 "$TUNNEL_LOG"
    exit 1
  fi
  sleep 0.5
done

if [ -z "$URL" ]; then
  echo ""
  printf '\033[31m✗\033[0m cloudflared did not produce a tunnel URL within 30s. Last log lines:\n'
  tail -20 "$TUNNEL_LOG"
  exit 1
fi

COMMENT_URL="${URL}/comment"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
printf '  \033[1mAudience comment URL\033[0m\n'
printf '    %s\n' "$COMMENT_URL"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

if [ "$HAS_QRENCODE" -eq 1 ]; then
  qrencode -o "$QR_PNG" -s 14 -m 4 "$COMMENT_URL"
  echo "  QR PNG saved → $QR_PNG"
  echo "  (drop into intro slide; print + tape to laptop as fallback)"
  echo ""
  qrencode -t ansiutf8 "$COMMENT_URL"
  echo ""
fi

echo "Tunnel running (pid=$TUNNEL_PID). Ctrl-C to stop."
echo ""
wait "$TUNNEL_PID"
