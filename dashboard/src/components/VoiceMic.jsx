import React, { useEffect, useRef, useState } from 'react';

/**
 * Push-to-talk mic. Hold-to-record (mouse/touch), release-to-upload.
 * POSTs WebM/Opus to /api/voice_comment — the backend transcribes via
 * whisper on Cactus (<200ms) and fires a voice_transcript WS event that
 * the dashboard consumes through useEmpireSocket.
 *
 * Four local states: idle | recording | transcribing | error.
 * We don't render the transcript here — the ChatPanel picks it up via
 * pendingComments so voice and typed comments look identical.
 *
 * Props:
 *   voiceTranscript: the latest {text, source, ms} from the hook. Used
 *     to show a small "heard: ..." chip for a beat after release so the
 *     user sees what the backend heard in case they need to retry.
 */
export function VoiceMic({ voiceTranscript }) {
  const [state, setState] = useState('idle');
  const [elapsedMs, setElapsedMs] = useState(0);
  const [lastError, setLastError] = useState(null);
  const recorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);
  const startedAtRef = useRef(0);
  const timerRef = useRef(null);

  // Clear the "heard: ..." chip after 4s of idle.
  const [recentTranscript, setRecentTranscript] = useState(null);
  useEffect(() => {
    if (!voiceTranscript?.text) return;
    setRecentTranscript(voiceTranscript);
    const t = setTimeout(() => setRecentTranscript(null), 4000);
    return () => clearTimeout(t);
  }, [voiceTranscript]);

  // Stop the mic stream + timer. Does NOT clear chunksRef — the caller owns
  // clearing that AFTER reading the collected audio.
  function releaseStream() {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    setElapsedMs(0);
  }

  async function beginRecording() {
    if (state !== 'idle') return;
    setLastError(null);
    console.log('[voicemic] requesting mic…');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // Prefer opus in webm — whisper handles it cleanly via ffmpeg server-side.
      const candidates = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/mp4',
      ];
      const mimeType = candidates.find(m => MediaRecorder.isTypeSupported(m)) || '';
      const rec = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);
      recorderRef.current = rec;
      chunksRef.current = [];
      rec.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          chunksRef.current.push(e.data);
          console.log('[voicemic] chunk', e.data.size, 'bytes');
        }
      };
      rec.onstop = handleStop;
      // Request a chunk every 500ms so we always have data even on super-short
      // holds, and so the first chunk arrives before stop() is called.
      rec.start(500);
      startedAtRef.current = Date.now();
      timerRef.current = setInterval(
        () => setElapsedMs(Date.now() - startedAtRef.current),
        100,
      );
      setState('recording');
      console.log('[voicemic] recording started, mime=', mimeType || '(default)');
    } catch (err) {
      console.error('[voicemic] getUserMedia failed', err);
      setLastError(err.name === 'NotAllowedError' ? 'mic_blocked' : 'mic_unavailable');
      setState('error');
      releaseStream();
      chunksRef.current = [];
      // Auto-dismiss error so user can retry.
      setTimeout(() => setState('idle'), 2000);
    }
  }

  function stopRecording() {
    if (state !== 'recording') return;
    const rec = recorderRef.current;
    if (!rec || rec.state === 'inactive') {
      releaseStream();
      chunksRef.current = [];
      setState('idle');
      return;
    }
    console.log('[voicemic] stopping recorder');
    rec.stop();  // triggers ondataavailable (final flush) then onstop
  }

  async function handleStop() {
    const rec = recorderRef.current;
    const mime = rec?.mimeType || 'audio/webm';
    const duration = Date.now() - startedAtRef.current;
    // Release mic + timer BEFORE we copy the chunks out. But DO NOT clear
    // chunksRef here — we need it for the blob below.
    releaseStream();
    recorderRef.current = null;

    if (duration < 150) {
      // Released in a flash — treat as a misclick and drop.
      console.log('[voicemic] release <150ms, dropping');
      chunksRef.current = [];
      setState('idle');
      return;
    }

    // Snapshot + clear so re-entry can't poison the next recording.
    const chunks = chunksRef.current;
    chunksRef.current = [];
    const blob = new Blob(chunks, { type: mime });
    console.log(`[voicemic] built blob: ${blob.size} bytes from ${chunks.length} chunks`);

    if (blob.size === 0) {
      console.warn('[voicemic] empty blob — nothing to upload');
      setLastError('empty_recording');
      setState('error');
      setTimeout(() => setState('idle'), 2000);
      return;
    }

    setState('transcribing');
    const fd = new FormData();
    const ext = mime.startsWith('audio/webm') ? 'webm' : 'bin';
    fd.append('audio', blob, `voice.${ext}`);

    const url = `http://${window.location.hostname}:8000/api/voice_comment`;
    console.log('[voicemic] POST', url);
    try {
      const resp = await fetch(url, { method: 'POST', body: fd });
      console.log('[voicemic] response status', resp.status);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const body = await resp.json();
      console.log('[voicemic] response body', body);
      // The WS voice_transcript event drives the dashboard UI.
    } catch (err) {
      console.error('[voicemic] upload failed', err);
      setLastError('upload_failed');
      setState('error');
      setTimeout(() => setState('idle'), 2000);
      return;
    }
    setState('idle');
  }

  // Handle pointer leave mid-hold: treat as release so we don't leave the
  // recorder running if the user drags off the button.
  function handlePointerLeave(e) {
    if (state === 'recording') stopRecording();
  }

  const label =
    state === 'recording' ? `● Recording ${(elapsedMs / 1000).toFixed(1)}s — release to send` :
    state === 'transcribing' ? 'Transcribing…' :
    state === 'error' ? (lastError === 'mic_blocked' ? 'Mic blocked — allow access' :
                         lastError === 'upload_failed' ? 'Upload failed — tap retry' :
                         lastError === 'empty_recording' ? 'No audio captured — retry' :
                         'Mic unavailable') :
    '🎙 Hold to speak';

  const bg =
    state === 'recording' ? '#ef4444' :
    state === 'transcribing' ? '#7c3aed' :
    state === 'error' ? '#dc2626' :
    '#27272a';
  const color =
    state === 'idle' ? '#a1a1aa' : '#fafafa';

  return (
    <div style={styles.wrap}>
      <button
        type="button"
        style={{ ...styles.btn, background: bg, color, borderColor: bg }}
        onPointerDown={beginRecording}
        onPointerUp={stopRecording}
        onPointerLeave={handlePointerLeave}
        onContextMenu={(e) => e.preventDefault()}
        disabled={state === 'transcribing'}
        aria-pressed={state === 'recording'}
      >
        {label}
      </button>
      {recentTranscript?.text && (
        <span style={styles.chip}>
          heard
          <span style={styles.chipText}>"{recentTranscript.text}"</span>
          <span style={styles.chipMeta}>
            · {recentTranscript.source === 'cactus_on_device' ? 'on-device' : recentTranscript.source}
            {' · '}
            {recentTranscript.ms}ms
          </span>
        </span>
      )}
    </div>
  );
}

const styles = {
  wrap: { display: 'flex', alignItems: 'center', gap: 10, minWidth: 0 },
  btn: {
    border: '1px solid #3f3f46',
    borderRadius: 8,
    padding: '10px 16px',
    fontSize: 14,
    fontWeight: 600,
    cursor: 'pointer',
    userSelect: 'none',
    touchAction: 'none',
    whiteSpace: 'nowrap',
    transition: 'background 120ms, border-color 120ms',
  },
  chip: {
    display: 'inline-flex', alignItems: 'baseline', gap: 6,
    background: '#18181b', border: '1px solid #27272a', borderRadius: 999,
    padding: '6px 12px', fontSize: 12, color: '#71717a',
    maxWidth: 360, overflow: 'hidden', textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  chipText: { color: '#fafafa', fontWeight: 600 },
  chipMeta: { color: '#52525b' },
};
