# EMPIRE — Build Architecture (v3, post-eng-review)

## What Runs Where

```
PHONE (Expo Go app)              MACBOOK (Cactus + Python + React)     RUNPOD (A100 GPU)
├── Camera capture         ws→   ├── Cactus CLI + Gemma 4 E4B          ├── LiveTalking
├── Mic relay              ws→   │   ├── Voice intent parsing           │   ├── Receives TTS audio
└── Branded UI                   │   ├── Product vision (fast, free)    │   ├── Lip-syncs avatar
                                 │   └── Comment classification         │   └── WebRTC → dashboard
                                 ├── FastAPI orchestrator               │
                                 │   ├── /analyze → Claude Vision       ├── TripoSR
                                 │   ├── /sell → Claude + ElevenLabs    │   └── Image → 3D mesh
                                 │   ├── /respond → Claude + ElevenLabs │
                                 │   └── /content → rembg               │
                                 └── Vite + React dashboard             │
                                     ├── Avatar (WebRTC from RunPod)    │
                                     ├── Product (3D model + photos)    │
                                     ├── Agent log (real-time steps)    │
                                     └── Chat panel                     │
```

## Key Architecture Decisions

1. **Gemma 4 runs on MacBook, not phone.** Cactus supports Apple Silicon.
   Phone is a lightweight Expo Go capture app (camera + mic over WebSocket).
   Eliminates React Native native module risk. Saves 3-5 hours.

2. **LiveTalking on RunPod A100 ($1.19/hr) for real-time lip sync.**
   Full photorealistic avatar with lip-synced speech.
   WebRTC output streams directly to dashboard.

3. **Gemma 4 does FIRST product analysis (fast, free, on-device).**
   Claude Vision does SECOND richer analysis (cloud, $0.01).
   Dashboard shows both with speed comparison. Proves on-device value.

4. **TripoSR on same RunPod pod.** Single product photo → 3D mesh in 0.5s.
   Auto-rotates in dashboard (not user-controlled, since target is livestream).

5. **Agent activity log IS the UX for wait time.**
   Each step logged with timestamps: "EYES: analyzing... 0.3s... found leather bag"
   Turns 5-8 second pipeline into the most impressive part of the demo.

## Data Flow: "Sell This"

```
1. [PHONE] User points camera, says "sell this for $49"
   → Sends frame + audio to MacBook via WebSocket

2. [MACBOOK] Gemma 4 (Cactus) processes voice + image
   → Parses intent: SELL, price: $49, product: visible
   → Vision: "leather crossbody bag, brass hardware" (0.3s, free)
   → Dashboard shows: "EYES: product detected (0.3s, on-device)"

3. [MACBOOK] Claude Vision API for rich analysis
   → Detailed JSON: materials, selling points, dimensions (2-3s)
   → Dashboard shows: "EYES: deep analysis complete (2.1s, cloud)"

4. [MACBOOK] Claude generates 30-second sales script
   → References specific visual details from analysis (1-2s)

5. [MACBOOK] ElevenLabs streaming TTS
   → Script → audio stream, first bytes in ~75ms

6. [RUNPOD] LiveTalking receives audio stream
   → Lip-synced avatar video → WebRTC to dashboard
   → Avatar starts talking about THIS specific product

7. [DASHBOARD] All panels update:
   → Avatar video (selling), product photos (rembg), 3D model (TripoSR)
   → Agent log shows full pipeline with timing

Total: ~5-8 seconds from voice command to avatar selling.
```

## Data Flow: Comment Response

```
1. [MACBOOK] Receives typed comment (simulated chat)
2. [MACBOOK] Gemma 4 classifies: QUESTION about materials
   → Drafts response on-device (0.3s)
3. [MACBOOK] Claude refines response with product context (0.5-2s)
4. [MACBOOK] ElevenLabs → audio
5. [RUNPOD] LiveTalking → lip-synced response
6. [DASHBOARD] Avatar responds: "Great question, this is full-grain leather..."

Total: ~3-6 seconds from comment to avatar speaking.
```

## Build Order (10 hours)

### Hours 1-2: Parallel Spike
- Install Cactus on MacBook: `brew install cactus-compute/cactus/cactus`
- Test Gemma 4 voice: "sell this for $49" → intent parsed?
- Test Gemma 4 vision: product photo → description?
- Deploy LiveTalking on RunPod A100: Docker, test audio → video

### Hours 3-4: Backend Core
- FastAPI scaffold
- WebSocket endpoint for phone
- /analyze: Gemma 4 (fast) + Claude Vision (rich)
- /sell: Claude script + ElevenLabs TTS + RunPod relay
- Test each endpoint in isolation

### Hours 5-6: End-to-End Pipeline
- Phone (Expo Go) sends frame + audio → MacBook processes → RunPod renders
- Full pipeline: "sell this" → avatar selling the product
- Agent log populating in real-time

### Hours 7-8: Comment Response + Dashboard
- /respond endpoint
- React dashboard with 4 panels
- WebRTC from RunPod embedded in dashboard
- Chat panel with simulated comments

### Hour 9: 3D + Polish
- TripoSR on RunPod → GLB → Three.js auto-rotate
- rembg product photo on dashboard
- Error handling + retry logic

### Hour 10: Rehearsal
- Full demo 3x with timing
- Pre-generate fallback content for one product
- Practice 2-minute script

## Fallback Chain

```
Avatar:     RunPod LiveTalking → HeyGen API (5 min free) → static image + audio
TTS:        ElevenLabs (33M credits) → Edge TTS (free)
Analysis:   Gemma 4 on-device (fast) + Claude Vision (rich) → Gemma 4 only
Scripts:    Claude API → Gemma 4 on-device
3D Model:   TripoSR on RunPod → skip (not critical path)
WiFi:       USB tethering as primary if venue WiFi is bad
```

## API Keys Needed

| Service | Purpose | Status |
|---------|---------|--------|
| Cactus + Gemma 4 | On-device inference | Free, `brew install` |
| Anthropic (Claude) | Product analysis + scripts | Need API key |
| ElevenLabs | TTS | Have 33M credits |
| RunPod | GPU for LiveTalking + TripoSR | Need account, ~$10-20 total |
