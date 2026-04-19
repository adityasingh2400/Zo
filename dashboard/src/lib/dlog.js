// dlog — fire-and-forget POSTer that mirrors browser-side debug events
// into the backend logger. Goal: a single `tail -f backend.log` shows
// pipeline traces (server) AND audio/video lifecycle events (client) under
// the SAME trace ids, so iterating on a bug doesn't require swiveling
// between two consoles.
//
// Usage:
//   import { dlog } from '../lib/dlog';
//   dlog('tier1', 'play_started', { url, fadeMs, intent });
//
// All calls are non-blocking. Failures (network down, backend not up yet)
// are silently swallowed — the dev experience must never degrade because a
// debug log POST didn't make it.

const API = `http://${window.location.hostname}:8000`;

// Pull a trace id from the most recent server event we've seen in this
// session, if any. Lets a dashboard event line up with the pipeline that
// triggered it. Updated by setTrace() which the message-listener wiring
// in LiveStage / useEmpireSocket can call when broadcasting events with
// trace ids attached.
let _lastTrace = null;
export function setTrace(t) { if (t) _lastTrace = t; }

// Subscribers receive every dlog call client-side so a visible
// EventLogHUD can render the live trace alongside the avatar without
// needing a console open. Listeners are stored in a Set so adding /
// removing one is O(1) and we don't leak refs.
const _listeners = new Set();
export function subscribe(fn) {
  _listeners.add(fn);
  return () => _listeners.delete(fn);
}

export function dlog(src, msg, data = {}) {
  // 1. Mirror to the browser console so the trace is visible without
  //    opening backend logs. Single-line format with a ▶ prefix so it's
  //    grep-able vs other console noise.
  try {
    // eslint-disable-next-line no-console
    console.info(`▶ [${src}] ${msg}`, data);
  } catch {}

  // 2. Notify in-page subscribers (EventLogHUD).
  const evt = { src, msg, data, ts: Date.now() };
  for (const fn of _listeners) {
    try { fn(evt); } catch {}
  }

  // 3. Mirror to backend so a tail -f covers BOTH sides of the trace.
  try {
    fetch(`${API}/api/dashboard_log`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ src, msg, data, trace: _lastTrace }),
      keepalive: true,  // survives the page being closed mid-flight
    }).catch(() => {});
  } catch {
    // POST construction itself failed — usually means we're in a teardown
    // path. Nothing to do; the dev console still has the same info.
  }
}
