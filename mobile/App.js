import React, { useState, useRef, useEffect } from 'react';
import {
  StyleSheet, View, Text, TouchableOpacity, SafeAreaView, StatusBar, Alert,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { Audio } from 'expo-av';

const BACKEND_WS = 'ws://192.168.0.117:8000/ws/phone'; // Change to your laptop IP

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState('Disconnected');
  const [recording, setRecording] = useState(null);
  const cameraRef = useRef(null);
  const wsRef = useRef(null);

  useEffect(() => {
    connectWS();
    return () => wsRef.current?.close();
  }, []);

  function connectWS() {
    const ws = new WebSocket(BACKEND_WS);
    ws.onopen = () => { setConnected(true); setStatus('Connected'); };
    ws.onclose = () => { setConnected(false); setStatus('Disconnected'); setTimeout(connectWS, 3000); };
    ws.onerror = () => setStatus('Connection error');
    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.type === 'status') setStatus(msg.message || msg.status);
      } catch {}
    };
    wsRef.current = ws;
  }

  async function captureAndSell() {
    if (!cameraRef.current || !wsRef.current) return;
    setStatus('Capturing...');

    const photo = await cameraRef.current.takePictureAsync({
      base64: true,
      quality: 0.7,
    });

    wsRef.current.send(JSON.stringify({
      type: 'sell_command',
      frame: photo.base64,
      voice_text: 'sell this',
    }));

    setStatus('Sent to EMPIRE!');
  }

  async function startVoiceCapture() {
    try {
      await Audio.requestPermissionsAsync();
      await Audio.setAudioModeAsync({ allowsRecordingIOS: true, playsInSilentModeIOS: true });
      const { recording: rec } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );
      setRecording(rec);
      setStatus('Recording...');
    } catch (err) {
      Alert.alert('Error', 'Could not start recording');
    }
  }

  async function stopVoiceCapture() {
    if (!recording) return;
    setStatus('Processing voice...');
    await recording.stopAndUnloadAsync();
    const uri = recording.getURI();
    setRecording(null);

    // For the demo, we capture photo + send with voice text
    // Full implementation would send audio to Cactus on MacBook
    await captureAndSell();
  }

  if (!permission) return <View />;
  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.container}>
        <Text style={styles.title}>EMPIRE needs camera access</Text>
        <TouchableOpacity style={styles.btn} onPress={requestPermission}>
          <Text style={styles.btnText}>Grant Permission</Text>
        </TouchableOpacity>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" />

      <View style={styles.header}>
        <Text style={styles.title}>EMPIRE</Text>
        <View style={[styles.dot, { backgroundColor: connected ? '#4ade80' : '#ef4444' }]} />
        <Text style={styles.status}>{status}</Text>
      </View>

      <CameraView ref={cameraRef} style={styles.camera} facing="back">
        <View style={styles.overlay}>
          <View style={styles.crosshair} />
        </View>
      </CameraView>

      <View style={styles.controls}>
        <TouchableOpacity
          style={[styles.btn, styles.sellBtn]}
          onPress={captureAndSell}
        >
          <Text style={styles.btnText}>📸 SELL THIS</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.btn, recording ? styles.stopBtn : styles.voiceBtn]}
          onPressIn={startVoiceCapture}
          onPressOut={stopVoiceCapture}
        >
          <Text style={styles.btnText}>
            {recording ? '🔴 Release to Send' : '🎤 Hold to Speak'}
          </Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  header: {
    flexDirection: 'row', alignItems: 'center', padding: 16, gap: 8,
  },
  title: { color: '#fff', fontSize: 24, fontWeight: '800', letterSpacing: 2 },
  dot: { width: 8, height: 8, borderRadius: 4 },
  status: { color: '#a1a1aa', fontSize: 14 },
  camera: { flex: 1 },
  overlay: {
    flex: 1, justifyContent: 'center', alignItems: 'center',
  },
  crosshair: {
    width: 200, height: 200, borderWidth: 2, borderColor: 'rgba(255,255,255,0.3)',
    borderRadius: 12,
  },
  controls: { padding: 16, gap: 12 },
  btn: {
    padding: 16, borderRadius: 12, alignItems: 'center',
  },
  sellBtn: { backgroundColor: '#7c3aed' },
  voiceBtn: { backgroundColor: '#1d4ed8' },
  stopBtn: { backgroundColor: '#dc2626' },
  btnText: { color: '#fff', fontSize: 18, fontWeight: '700' },
});
