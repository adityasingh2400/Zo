import { useState, useEffect, useRef, useCallback } from 'react';

const WS_URL = `ws://${window.location.hostname}:8000/ws/dashboard`;

export function useEmpireSocket() {
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState('idle');
  const [productData, setProductData] = useState(null);
  const [productPhoto, setProductPhoto] = useState(null);
  const [salesScript, setSalesScript] = useState(null);
  const [agentLog, setAgentLog] = useState([]);
  const [latestAudio, setLatestAudio] = useState(null);
  const [commentResponse, setCommentResponse] = useState(null);
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
          setAgentLog(msg.state.agent_log || []);
          break;
        case 'agent_log':
          setAgentLog(prev => [...prev, msg.entry]);
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
          setCommentResponse(msg);
          setLatestAudio({ audio: msg.audio, format: msg.format });
          break;
      }
    };

    wsRef.current = ws;
  }, []);

  useEffect(() => {
    connect();
    return () => wsRef.current?.close();
  }, [connect]);

  const sendComment = useCallback((text) => {
    wsRef.current?.send(JSON.stringify({ type: 'simulate_comment', text }));
  }, []);

  const sendSell = useCallback((voiceText) => {
    wsRef.current?.send(JSON.stringify({ type: 'simulate_sell', voice_text: voiceText }));
  }, []);

  return {
    connected, status, productData, productPhoto, salesScript,
    agentLog, latestAudio, commentResponse, sendComment, sendSell,
  };
}
