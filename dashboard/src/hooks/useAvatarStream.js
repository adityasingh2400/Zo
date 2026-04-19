import { useEffect, useRef, useState, useCallback } from 'react';

/**
 * useAvatarStream — listens to play_clip events on the existing EMPIRE
 * WebSocket and exposes the current desired clip for each video tier.
 *
 * Architecture (see seamless_avatar_continuity plan):
 *   - Tier 0 = always-on idle layer underneath. Director rotates this.
 *   - Tier 1 = reactive layer. Bridges, responses, pitches.
 *
 * The hook does not own the <video> elements. It owns *what should be there*
 * and a `clipAck` callback for the LiveStage to report playback status back.
 *
 * It deliberately does NOT open its own WebSocket. It piggybacks on the
 * shared empire socket so we don't double-subscribe and so server-side
 * broadcast deduplication just works.
 */

const WS_URL = `ws://${window.location.hostname}:8000/ws/dashboard`;

// Mirrors of backend constants in agents/avatar_director.py
export const TIER1_CROSSFADE_MS = 350;
export const TIER0_CROSSFADE_MS = 600;
export const TIER1_FADEOUT_MS = 500;

/**
 * @returns {{
 *   tier0: { intent: string, url: string, loop: boolean, fadeMs: number, ts: number } | null,
 *   tier1: { intent: string, url: string, loop: boolean, fadeMs: number, ts: number } | null,
 *   sendAck: (intent: string, url: string, status: 'started'|'ended'|'stalled'|'skipped') => void,
 *   sendStageReady: () => void,
 * }}
 */
export function useAvatarStream({ wsRef } = {}) {
  const [tier0, setTier0] = useState(null);
  const [tier1, setTier1] = useState(null);
  const localWsRef = useRef(null);

  // If parent didn't pass a shared socket, open our own (for storybook etc.).
  useEffect(() => {
    if (wsRef?.current) return;
    const ws = new WebSocket(WS_URL);
    ws.onmessage = handleMessage;
    localWsRef.current = ws;
    return () => ws.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const sendAck = useCallback((intent, url, status) => {
    const ws = wsRef?.current || localWsRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'clip_ack', intent, url, status, client_ts: Date.now(),
      }));
    }
  }, [wsRef]);

  const sendStageReady = useCallback(() => {
    const ws = wsRef?.current || localWsRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'stage_ready', tier0_playing: true }));
    }
  }, [wsRef]);

  // Single message handler for both shared + local sockets.
  function handleMessage(e) {
    let msg;
    try { msg = JSON.parse(e.data); } catch { return; }
    if (msg.type !== 'play_clip') return;
    const clip = {
      intent: msg.intent,
      url: msg.url,
      loop: !!msg.loop,
      fadeMs: msg.fade_ms ?? (msg.layer === 'tier1' ? TIER1_CROSSFADE_MS : TIER0_CROSSFADE_MS),
      ts: msg.ts,
      mode: msg.mode || 'crossfade',
      // Audio-first metadata. `muted` tells LiveStage to set incomingEl.muted=true
      // and skip the volume ramp (audio is coming from a separate <audio>
      // element). `expectedDurationMs` enables the duration handshake on
      // canplaythrough — if the video duration drifts >150ms from this,
      // LiveStage rejects the video and lets the standalone audio play alone.
      muted: !!msg.muted,
      expectedDurationMs: msg.expected_duration_ms ?? null,
    };
    if (msg.layer === 'tier0') setTier0(clip);
    else if (msg.layer === 'tier1') setTier1(clip);
  }

  // If the parent passed a shared wsRef, attach a listener to it. We use
  // addEventListener so we don't clobber its existing onmessage handler.
  useEffect(() => {
    const ws = wsRef?.current;
    if (!ws) return;
    ws.addEventListener('message', handleMessage);
    return () => ws.removeEventListener('message', handleMessage);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wsRef]);

  return { tier0, tier1, sendAck, sendStageReady };
}
