import { useState, useEffect, useRef, useCallback } from 'react';

const WS_URL = `ws://${window.location.hostname}:8000/ws/dashboard`;
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
  const [liveStage, setLiveStage] = useState('INTRO');
  const [pendingComments, setPendingComments] = useState([]); // [{id, text, t0}]
  const [view3d, setView3d] = useState(null); // {kind, frames|url, ms, source}
  const [transcriptExtract, setTranscriptExtract] = useState(null); // on-device structured pitch hints
  const wsRef = useRef(null);

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
        case 'comment_response_video':
          setResponseVideo(msg);
          setCommentResponse(msg);
          setLiveStage('BRIDGE');
          // Drop any pending entry that matches this comment so the chip clears.
          setPendingComments(prev => prev.filter(p => p.text !== msg.comment));
          break;
        case 'pitch_video':
          setPitchVideoUrl(msg.url);
          setLiveStage('PITCH');
          break;
        case 'transcript':
          setTranscript(msg.text);
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
    const id = `c_${Date.now()}`;
    setPendingComments(prev => [...prev, { id, text: text.trim(), t0: Date.now() }]);
    wsRef.current?.send(JSON.stringify({ type: 'simulate_comment', text: text.trim() }));
  }, []);

  const sendSell = useCallback((voiceText) => {
    wsRef.current?.send(JSON.stringify({ type: 'simulate_sell', voice_text: voiceText }));
  }, []);

  return {
    connected, status, productData, productPhoto, salesScript,
    agentLog, latestAudio, commentResponse, transcript,
    pitchVideoUrl, responseVideo, liveStage, setLiveStage, pendingComments,
    view3d, transcriptExtract,
    sendComment, sendSell,
    wsRef, // exposed so useAvatarStream can attach an extra message listener
  };
}
