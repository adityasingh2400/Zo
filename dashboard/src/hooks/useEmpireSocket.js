import { useState, useEffect, useRef, useCallback } from 'react';

// Backend's /ws/dashboard requires `?token=<value>` when WS_SHARED_SECRET
// is set in backend .env (Sprint 2.6). We forward VITE_WS_TOKEN if defined;
// blank token is fine in dev mode (backend default-allows when secret unset).
const WS_TOKEN = import.meta.env?.VITE_WS_TOKEN || '';
const WS_URL = `ws://${window.location.hostname}:8000/ws/dashboard`
  + (WS_TOKEN ? `?token=${encodeURIComponent(WS_TOKEN)}` : '');
// Live-mode states: how the dashboard chrome interprets `status`.
// idle → analyzing/creating (INTRO) → selling/live (PITCH/LIVE), with BRIDGE
// inserted whenever a comment response is mid-flight.
export const LIVE_STAGES = ['INTRO', 'BRIDGE', 'PITCH', 'LIVE'];

export function useEmpireSocket() {
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState('idle');
  const [productData, setProductData] = useState(null);
  const [productPhoto, setProductPhoto] = useState(null);
  const [salesScript, setSalesScript] = useState(null);
  const [agentLog, setAgentLog] = useState([]);
  const [latestAudio, setLatestAudio] = useState(null);
  const [commentResponse, setCommentResponse] = useState(null);
  const [transcript, setTranscript] = useState(null);
  const [pitchVideoUrl, setPitchVideoUrl] = useState(null);
  const [responseVideo, setResponseVideo] = useState(null); // {url, comment, response, total_ms, ...}
  // Audio-first dispatch payload: {url, comment, response, word_timings,
  // expected_duration_ms, intent, ts, seq}. seq monotonically increases so
  // LiveStage can swap the playing <audio> when a fresh dispatch arrives
  // mid-playback (e.g. operator fires a new comment while one is playing).
  // Cleared by the matching comment_response_video or video_failed event so
  // KaraokeCaptions stop tracking once the audio source is exhausted.
  const [audioResponse, setAudioResponse] = useState(null);
  // Pitch dispatch — fires when run_sell_pipeline → _run_audio_first_pitch
  // → Director.dispatch_audio_first_pitch broadcasts pitch_audio. Driven
  // exclusively by the video-upload pipeline (chat doesn't trigger pitches).
  // Same shape as audioResponse but tagged so LiveStage knows to mount the
  // TranslationChip overlay's pitch variant (the audio is the freshly-
  // generated ~30s pitch, not a live response). Cleared after the audio
  // ends or pitch_audio_end fires.
  const [pitchAudio, setPitchAudio] = useState(null);
  const audioSeqRef = useRef(0);
  const [liveStage, setLiveStage] = useState('INTRO');
  const [pendingComments, setPendingComments] = useState([]); // [{id, text, t0}]
  const [view3d, setView3d] = useState(null); // {kind, frames|url, ms, source}
  const [transcriptExtract, setTranscriptExtract] = useState(null); // on-device structured pitch hints
  // Voice flow state — driven by Cody's voice + router events (additive).
  // Lifecycle: VoiceMic.jsx may flip to 'transcribing' optimistically; server
  // broadcasts then walk through 'thinking' → 'responding' → null.
  const [voiceState, setVoiceState] = useState(null); // null | 'transcribing' | 'thinking' | 'responding'
  const [voiceTranscript, setVoiceTranscript] = useState(null); // {text, source, ms} — latest mic-in transcript
  // Router telemetry: last decision (drives the badge), the rolling window of
  // recent decisions (drives the RoutingPanel list), and rolled-up counters.
  // local = any tool that avoids a cloud round-trip (respond_locally /
  // play_canned_clip / block_comment). cloud = escalate_to_cloud.
  const [routingDecision, setRoutingDecision] = useState(null);
  const [routingDecisions, setRoutingDecisions] = useState([]); // newest-last
  const [routingStats, setRoutingStats] = useState({
    total: 0, local: 0, cloud: 0, cost_saved_usd: 0,
  });
  const voiceStateTimerRef = useRef(null);
  const wsRef = useRef(null);

  // Helper: set voice state with a safety auto-clear so a dropped follow-up
  // event can never leave the pill stuck on stage.
  const setVoiceStateSafe = useCallback((s) => {
    setVoiceState(s);
    if (voiceStateTimerRef.current) clearTimeout(voiceStateTimerRef.current);
    if (s) {
      voiceStateTimerRef.current = setTimeout(() => setVoiceState(null), 12_000);
    }
  }, []);

  const connect = useCallback(() => {
    const ws = new WebSocket(WS_URL);

    ws.onopen = () => setConnected(true);
    ws.onclose = () => {
      setConnected(false);
      setTimeout(connect, 2000);
    };

    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);

      switch (msg.type) {
        case 'state_sync':
          setStatus(msg.state.status);
          setProductData(msg.state.product_data);
          setSalesScript(msg.state.sales_script || null);
          setPitchVideoUrl(msg.state.pitch_video_url || null);
          if (msg.state.last_response_video_url) {
            setResponseVideo(prev => prev || { url: msg.state.last_response_video_url });
          }
          setAgentLog(msg.state.agent_log || []);
          if (msg.state.view_3d) setView3d(msg.state.view_3d);
          if (msg.state.transcript_extract) setTranscriptExtract(msg.state.transcript_extract);
          break;
        case 'view_3d':
          setView3d(msg.data);
          break;
        case 'transcript_extract':
          setTranscriptExtract(msg.data);
          break;
        case 'agent_log':
          setAgentLog(prev => [...prev.slice(-49), msg.entry]);
          break;
        case 'product_data':
          setProductData(msg.data);
          break;
        case 'product_photo':
          setProductPhoto(msg.photo);
          break;
        case 'sales_script':
          setSalesScript(msg.script);
          break;
        case 'status':
          setStatus(msg.status);
          break;
        case 'tts_audio':
          setLatestAudio({ audio: msg.audio, format: msg.format });
          break;
        case 'comment_response':
          // Text-only response (legacy path); the *_video event is the real one.
          setCommentResponse(msg);
          if (msg.audio) setLatestAudio({ audio: msg.audio, format: msg.format });
          break;
        case 'comment_response_audio':
          // Audio-first dispatch — TTS bytes are saved + ready to play. The
          // visible video crossfades in later (comment_response_video below)
          // with audio_already_playing=true so the dashboard mutes that
          // video element and lets the standalone <audio> finish the audio
          // track. KaraokeCaptions reads word_timings to highlight word-by-word.
          if (msg.url) {
            audioSeqRef.current += 1;
            setAudioResponse({
              ...msg,
              seq: audioSeqRef.current,
              receivedAt: Date.now(),
            });
            // We're in the middle of responding — pill goes responding so the
            // LIVE indicator transitions correctly when video lands.
            setVoiceStateSafe('responding');
            // Drop the pending chip immediately on audio arrival so the
            // floating "Reading..." overlay doesn't keep showing while we
            // hear the answer (was previously cleared by the video event).
            setPendingComments(prev => prev.filter(p => p.text !== msg.comment));
          }
          break;
        case 'comment_response_video':
          setResponseVideo(msg);
          setCommentResponse(msg);
          setLiveStage('BRIDGE');
          // Drop any pending entry that matches this comment so the chip clears.
          setPendingComments(prev => prev.filter(p => p.text !== msg.comment));
          // Voice flow lands here if this response was triggered by voice.
          // Clearing voiceState lets the LIVE pill take over visually.
          setVoiceStateSafe(null);
          // Audio-first: the audio session is owned by the audio element, but
          // the comment_response_video event is the canonical "render done"
          // signal that the audience-facing video is on stage. Don't clear
          // audioResponse here — KaraokeCaptions need to keep tracking until
          // audio actually ends (handled inside LiveStage on audio.ended).
          break;
        case 'comment_response_video_failed':
          // Audio-first background Wav2Lip failed. Audio still plays out via
          // the existing <audio> element; we just won't ever crossfade the
          // visual layer. Clear pending chip so it doesn't hang.
          setPendingComments(prev => prev.filter(p => p.text !== msg.comment));
          setVoiceStateSafe(null);
          break;
        case 'pitch_audio':
          // 30s pre-rendered Veo pitch — audio + word_timings + chip should
          // all start at once. Stored separately from audioResponse so a live
          // comment that arrives mid-pitch doesn't preempt the pitch by
          // overwriting the audio source.
          if (msg.url) {
            audioSeqRef.current += 1;
            setPitchAudio({
              ...msg,
              seq: audioSeqRef.current,
              receivedAt: Date.now(),
            });
            setLiveStage('PITCH');
          }
          break;
        case 'pitch_audio_end':
          // Pitch ended naturally (audio.ended on the dashboard) — backend
          // can also broadcast this proactively to clear the chip if needed.
          setPitchAudio(null);
          break;
        case 'voice_state':
          // Director-driven explicit voice state. Authoritative.
          setVoiceStateSafe(msg.state || null);
          break;
        case 'pitch_video':
          setPitchVideoUrl(msg.url);
          setLiveStage('PITCH');
          break;
        case 'transcript':
          setTranscript(msg.text);
          break;
        case 'routing_decision':
          // Router has picked a tool. Keep the last decision (drives the
          // badge), a rolling 50-item window (drives RoutingPanel feed), and
          // rolled-up counters (KPI cards).
          //
          // Attach a stable sequence id so React keys don't shift when new
          // decisions arrive — otherwise old rows remount and their pulse
          // animation re-fires, which looks like every row is blinking.
          setRoutingDecision(msg);
          setRoutingDecisions(prev => {
            const nextSeq = (prev[prev.length - 1]?.seq ?? 0) + 1;
            return [...prev.slice(-49), {
              ...msg,
              seq: nextSeq,
              receivedAt: Date.now(),
            }];
          });
          setRoutingStats(prev => {
            // Reject non-finite cost_saved_usd (NaN / Infinity) — if a
            // malformed broadcast ever poisons the accumulator, the panel
            // would show "$NaN" for the rest of the session.
            const saved = Number(msg.cost_saved_usd);
            const delta = Number.isFinite(saved) && saved >= 0 ? saved : 0;
            return {
              total: prev.total + 1,
              local: prev.local + (msg.tool !== 'escalate_to_cloud' ? 1 : 0),
              cloud: prev.cloud + (msg.tool === 'escalate_to_cloud' ? 1 : 0),
              cost_saved_usd: prev.cost_saved_usd + delta,
            };
          });
          // Advance the voice-state pill — the router has picked a tool, so
          // the avatar is about to respond.
          setVoiceStateSafe('responding');
          break;
        case 'comment_blocked':
          // Router decided the comment is spam. Drop its pending chip so
          // LiveStage's "Reading..." overlay clears. No video, no bubble.
          setPendingComments(prev => prev.filter(p => p.text !== msg.comment));
          break;
        case 'comment_failed':
          // Downstream render failed (Wav2Lip, TTS, Claude, etc.). Drop the
          // matching pending chip so LiveStage's "Reading..." overlay clears.
          setPendingComments(prev => prev.filter(p => p.text !== msg.comment));
          // If Claude drafted text before the downstream blew up, show it
          // as a degraded reply — better than nothing while the pod is down.
          if (msg.response) {
            const degraded = {
              type: 'comment_failed',
              comment: msg.comment,
              response: msg.response,
              degraded: true,
              total_ms: 0,
            };
            setCommentResponse(degraded);
            setResponseVideo(prev => prev ?? degraded);
          }
          break;
        case 'audience_comment':
          // QR-submitted comment from a phone in the room (username from
          // the form), OR an operator-typed test comment echoed back from
          // the simulate_comment WS handler (username='operator'). The
          // backend has already (a) broadcast this informational event and
          // (b) handed the text to run_routed_comment so the standard
          // router + cost ticker + comment_response_video chain fires
          // identically to a typed comment.
          //
          // Surface as pendingComments so the home dashboard's ChatPanel
          // shows the same "AI Seller (rendering…) responding to '<text>'"
          // placeholder that typed comments produce, AND so the /stage
          // TikTokShopOverlay chat rail renders a bubble (single-source
          // path — see TikTokShopOverlay.jsx). Cleared by the matching
          // comment_response_video (filter by text).
          //
          // Dedup: when the operator types in ChatPanel, sendComment
          // optimistically pushes pendingComments (so the pending pill
          // appears instantly) AND fires simulate_comment over the WS.
          // The backend echoes it back as audience_comment with the same
          // client_id. We swallow the echo when its client_id matches a
          // local optimistic pending entry. This is targeted on client_id
          // (not text), so two different audience phones submitting the
          // same text within a few seconds each get their own bubble —
          // they have no client_id and hit the normal append path.
          if (msg.text) {
            const audId = `aud_${msg.ts || Date.now()}`;
            setPendingComments(prev => {
              if (msg.client_id && prev.some(p => p.id === msg.client_id)) {
                return prev;
              }
              return [...prev, {
                id: audId, text: msg.text, t0: Date.now(),
                source: 'audience', username: msg.username || 'guest',
              }];
            });
          }
          break;
        case 'force_phase':
          // TransportControls fired POST /api/director/force_phase. Backend
          // broadcasts this event with the target phase so every connected
          // dashboard updates in sync (not just the one that clicked).
          if (msg.phase && LIVE_STAGES.includes(msg.phase)) {
            setLiveStage(msg.phase);
          }
          break;
        case 'on_air':
          // Operator pressed On Air. Currently soft (doesn't gate pipeline)
          // — Item 5 wires distribution fanout to this flag.
          if (typeof msg.on === 'boolean') {
            setStatus(msg.on ? 'live' : 'idle');
          }
          break;
        case 'voice_transcript':
          // Fires within ~200ms of push-to-talk release. Drop empty
          // transcripts (no_speech / transcription_failed) — the endpoint
          // has already short-circuited those; the dashboard shouldn't
          // flash an empty bubble.
          if (msg.text) {
            setVoiceTranscript({ text: msg.text, source: msg.source, ms: msg.ms });
            // Advance the voice-state pill — whisper has text, router is
            // next.
            setVoiceStateSafe('thinking');
            // Add as a pending comment so ChatPanel renders the
            // "AI Seller (rendering...)" placeholder alongside the bubble,
            // matching the typed-comment UX. Cleared by comment_response_video.
            const id = `v_${Date.now()}`;
            setPendingComments(prev => [...prev, {
              id, text: msg.text, t0: Date.now(),
              source: 'voice', voiceSource: msg.source, voiceMs: msg.ms,
            }]);
          }
          break;
      }
    };

    wsRef.current = ws;
  }, []);

  useEffect(() => {
    connect();
    return () => wsRef.current?.close();
  }, [connect]);

  // Derive a coarse live stage from backend status when no explicit stage event
  // has updated us recently.
  useEffect(() => {
    if (status === 'idle') setLiveStage('INTRO');
    else if (status === 'analyzing' || status === 'creating') setLiveStage('INTRO');
    else if (status === 'live') setLiveStage(prev => (prev === 'BRIDGE' ? prev : 'LIVE'));
    else if (status === 'selling') setLiveStage('PITCH');
  }, [status]);

  const sendComment = useCallback((text) => {
    if (!text?.trim()) return;
    const id = `c_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    setPendingComments(prev => [...prev, { id, text: text.trim(), t0: Date.now() }]);
    // client_id round-trips through the backend so the audience_comment
    // echo handler can dedup THIS optimistic entry without false-positiving
    // on a real audience phone that happened to type the same text within
    // a few seconds. See the audience_comment case below for the dedup.
    wsRef.current?.send(JSON.stringify({
      type: 'simulate_comment',
      text: text.trim(),
      client_id: id,
    }));
  }, []);

  const sendSell = useCallback((voiceText) => {
    wsRef.current?.send(JSON.stringify({ type: 'simulate_sell', voice_text: voiceText }));
  }, []);

  return {
    connected, status, productData, productPhoto, salesScript,
    agentLog, latestAudio, commentResponse, transcript,
    pitchVideoUrl, responseVideo, liveStage, setLiveStage, pendingComments,
    view3d, transcriptExtract,
    // Voice + routing surface
    voiceState, setVoiceState: setVoiceStateSafe,
    voiceTranscript, routingDecision, routingDecisions, routingStats,
    // Audio-first surface
    audioResponse, setAudioResponse, pitchAudio, setPitchAudio,
    sendComment, sendSell,
    wsRef, // exposed so useAvatarStream can attach an extra message listener
  };
}
