import React, { useEffect, useRef, useState } from 'react';
import QRCode from 'qrcode';
import { dlog } from '../lib/dlog';

const API_BASE = `http://${window.location.hostname}:8000`;

/**
 * PhoneQRPanel — replaces drag-drop with a phone-streamed upload.
 *
 * On mount, hits POST /api/phone/session to allocate a session and gets
 * back URLs for both the mobile recorder page and the WebSocket upload
 * endpoint, both pinned to the Mac's LAN IP. We render the recorder URL
 * as a QR code; the operator's phone scans it, the recorder page opens,
 * the operator hits Record, and video chunks stream into the backend
 * over the WS — same end state as if they'd drag-dropped the file.
 *
 * Live status is driven by `phoneUpload` from useEmpireSocket which is
 * fed by the backend's phone_upload_status / phone_upload_complete /
 * phone_upload_failed broadcasts. The operator sees the phone go from
 * "scan to start" → "📱 connected" → "● recording" → "↑ uploading 4.2MB"
 * → "📺 sent! pipeline starting" without ever leaving the dashboard.
 */
export function PhoneQRPanel({ phoneUpload, onUploadComplete, connected }) {
  const canvasRef = useRef(null);
  const [session, setSession] = useState(null);
  const [error, setError] = useState(null);
  // Track WS connection transitions so we can re-allocate the phone
  // session whenever the dashboard's connection drops + reconnects —
  // a backend restart wipes the in-memory session registry, so any
  // previously-issued QR points at a stale session_id and the phone
  // gets a 404 when it scans. Re-allocating on reconnect keeps the
  // QR live across hot-reloads / manual restarts without requiring
  // the operator to refresh the page.
  const lastConnectedRef = useRef(connected);

  async function allocate() {
    try {
      const resp = await fetch(`${API_BASE}/api/phone/session`, { method: 'POST' });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      setSession(data);
      setError(null);
    } catch (e) {
      setError(e.message || 'failed to allocate phone session');
    }
  }

  // Initial allocation on mount.
  useEffect(() => {
    allocate();
  }, []);

  // Re-allocate when the WS reconnects (false → true edge). Covers the
  // backend-restart case where the in-memory session registry was wiped.
  useEffect(() => {
    if (connected && !lastConnectedRef.current) {
      // Skip the very first connect — the initial allocation above
      // covers it. Only re-fire on RE-connects.
      if (session) {
        allocate();
      }
    }
    lastConnectedRef.current = connected;
  }, [connected]);  // eslint-disable-line react-hooks/exhaustive-deps

  // 2. Render the QR whenever the session URL changes. qrcode lib draws
  //    directly onto our canvas — no React re-render needed for the
  //    matrix, just for the surrounding chrome.
  useEffect(() => {
    if (!session?.recorder_url || !canvasRef.current) return;
    QRCode.toCanvas(canvasRef.current, session.recorder_url, {
      width: 240,
      margin: 1,
      color: {
        dark: '#fafafa',
        light: '#0a0a0b',
      },
      errorCorrectionLevel: 'M',
    }).catch(() => {
      setError('QR render failed');
    });
  }, [session?.recorder_url]);

  // 3. Flip hasUploaded the moment the phone connects to the upload WS
  //    — NOT when the upload completes. This matches drag-drop's
  //    semantics exactly: drag-drop sets hasUploaded synchronously on
  //    the drop event (before the file even POSTs), giving LiveStage
  //    several seconds to mount + attach its useAvatarStream listener
  //    before the backend's _play_upload_bridge fires walk_off. The
  //    previous "wait for complete" gating made LiveStage mount AFTER
  //    the backend had already broadcast the walk_off play_clip event,
  //    so useAvatarStream wasn't listening yet and the bridge silently
  //    failed to render. Firing on first connect (status='connected',
  //    which lands the moment the WS opens — well before the operator
  //    even hits Record on the phone) gives the stage 5-10 s of
  //    mount-stabilise time before the pipeline emits anything.
  //    Idempotent — onUploadComplete just calls setHasUploaded(true).
  useEffect(() => {
    if (phoneUpload?.status) {
      dlog('phone', 'status_changed', {
        status: phoneUpload.status,
        bytes: phoneUpload.bytes_received,
        chunks: phoneUpload.chunks_count,
        flipping_hasUploaded: true,
      });
      onUploadComplete?.();
    }
  }, [phoneUpload?.status, onUploadComplete]);

  // ── Render-time state derivation ────────────────────────────────────
  // We have several possible visible states:
  //   - error allocating session    → red banner + retry hint
  //   - waiting for connection      → QR + "scan to record"
  //   - phone connected             → QR dimmed + "📱 connected"
  //   - recording                   → recording chip with bytes
  //   - uploading                   → spinner + bytes
  //   - complete                    → checkmark, will transition out
  let primaryLabel = 'Scan to record';
  let primarySub = 'Open camera, point at the QR';
  let chipColor = null;
  let dimQR = false;
  if (phoneUpload) {
    switch (phoneUpload.status) {
      case 'connected':
        primaryLabel = '📱 Phone connected';
        primarySub = 'Tap the red button to start recording';
        chipColor = '#22c55e';
        dimQR = true;
        break;
      case 'recording':
        primaryLabel = '● Recording…';
        primarySub = `${formatBytes(phoneUpload.bytes_received || 0)} sent`;
        chipColor = '#ef4444';
        dimQR = true;
        break;
      case 'uploading':
        primaryLabel = '↑ Uploading…';
        primarySub = `${formatBytes(phoneUpload.bytes_received || 0)} — pipeline starting`;
        chipColor = '#7c3aed';
        dimQR = true;
        break;
      case 'complete':
        primaryLabel = '✓ Sent';
        primarySub = 'Pipeline running, watch the avatar';
        chipColor = '#22c55e';
        dimQR = true;
        break;
      case 'failed':
        primaryLabel = '✗ Upload failed';
        primarySub = phoneUpload.error || 'try again';
        chipColor = '#ef4444';
        dimQR = false;
        break;
      default:
        break;
    }
  }

  return (
    <div style={styles.root}>
      <div style={styles.qrWrap}>
        <canvas
          ref={canvasRef}
          style={{ ...styles.qr, opacity: dimQR ? 0.35 : 1 }}
        />
        {chipColor && (
          <span
            style={{
              ...styles.chip,
              background: chipColor,
              color: '#0a0a0b',
            }}
          >
            {phoneUpload?.status?.toUpperCase()}
          </span>
        )}
      </div>
      <div style={styles.label}>{primaryLabel}</div>
      <div style={styles.sub}>{primarySub}</div>
      {session && (
        <div style={styles.url}>
          {/* Plaintext URL fallback in case QR scanning is flaky.
              Operator can type it into Safari manually. */}
          {session.recorder_url}
        </div>
      )}
      {error && (
        <div style={styles.error}>
          {error}. Backend on LAN reachable?
        </div>
      )}
    </div>
  );
}

function formatBytes(n) {
  if (!n) return '0 B';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(2)} MB`;
}

const styles = {
  root: {
    display: 'flex', flexDirection: 'column',
    alignItems: 'center', gap: 14,
    color: '#fafafa',
    pointerEvents: 'none',
  },
  qrWrap: {
    position: 'relative',
    padding: 14,
    background: 'rgba(15,15,18,0.85)',
    border: '1px solid #27272a',
    borderRadius: 16,
    boxShadow: '0 8px 28px rgba(0,0,0,0.5)',
  },
  qr: {
    display: 'block',
    transition: 'opacity 240ms ease',
  },
  chip: {
    position: 'absolute',
    top: -10, right: -10,
    fontSize: 9, fontWeight: 900,
    letterSpacing: 1.2,
    padding: '4px 9px',
    borderRadius: 999,
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    boxShadow: '0 2px 12px rgba(0,0,0,0.5)',
  },
  label: {
    fontSize: 18,
    fontWeight: 800,
    letterSpacing: 0.5,
    color: '#fafafa',
    fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif',
  },
  sub: {
    fontSize: 11,
    fontWeight: 600,
    letterSpacing: 1.2,
    color: '#71717a',
    textTransform: 'uppercase',
  },
  url: {
    fontSize: 10,
    color: '#52525b',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    marginTop: 4,
    maxWidth: 280,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  error: {
    fontSize: 11,
    color: '#fca5a5',
    background: 'rgba(127,29,29,0.4)',
    padding: '4px 10px',
    borderRadius: 6,
    border: '1px solid rgba(239,68,68,0.4)',
  },
};
