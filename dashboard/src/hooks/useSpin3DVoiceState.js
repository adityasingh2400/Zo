import { useEffect, useRef, useState } from 'react';

/**
 * useSpin3DVoiceState — derive a Spin3D `state` prop ('idle' | 'thinking' |
 * 'responding') from the shared dashboard WebSocket. Keeps the rim light,
 * spin speed, and bloom on every <Spin3D> on the page in lockstep with the
 * avatar's voice pipeline without prop-drilling through every container.
 *
 * Originally lived inside ProductPanel as a private helper. Lifted here
 * (with no behavior change) so the new TikTokShopOverlay product showcase
 * can use the same source of truth — both panels drive identical state
 * out of one shared WS subscription.
 *
 * Dep includes `connected`, not just `wsRef`. wsRef is a stable ref object
 * whose identity never changes, and useEmpireSocket assigns wsRef.current
 * inside its OWN useEffect — which fires AFTER child effects (effects run
 * bottom-up). On first mount this effect would see wsRef.current=null,
 * bail, and never re-trigger; the rim light would silently never react to
 * voice events. `connected` flips false→true via ws.onopen, which gives
 * this effect the trigger to re-run with wsRef.current populated.
 *
 * Safety auto-clear after 12s so a dropped follow-up event never sticks
 * the spin in 'responding' for the rest of the demo.
 */
export function useSpin3DVoiceState({ wsRef, connected }) {
  const [state, setState] = useState('idle');
  const clearTimerRef = useRef(null);
  useEffect(() => {
    const ws = wsRef?.current;
    if (!ws) return;
    function setStateSafe(s) {
      setState(s || 'idle');
      if (clearTimerRef.current) clearTimeout(clearTimerRef.current);
      if (s && s !== 'idle') {
        clearTimerRef.current = setTimeout(() => setState('idle'), 12_000);
      }
    }
    function onMessage(e) {
      let msg;
      try { msg = JSON.parse(e.data); } catch { return; }
      switch (msg.type) {
        case 'voice_state':
          // Director-driven, authoritative.
          setStateSafe(msg.state || 'idle');
          break;
        case 'voice_transcript':
          setStateSafe('thinking');
          break;
        case 'routing_decision':
          setStateSafe('responding');
          break;
        case 'comment_response_video':
          setStateSafe('idle');
          break;
      }
    }
    ws.addEventListener('message', onMessage);
    return () => {
      ws.removeEventListener('message', onMessage);
      if (clearTimerRef.current) clearTimeout(clearTimerRef.current);
    };
  }, [wsRef, connected]);
  return state;
}
