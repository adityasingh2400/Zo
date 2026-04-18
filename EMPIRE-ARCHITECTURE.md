# EMPIRE — Build Architecture (v2, post-review)

## Quick Reference: What Runs Where

```
PHONE (Gemma 4 on Cactus)          LAPTOP (Python + React)
├── Voice command listener    ←ws→  ├── FastAPI orchestrator
├── Camera product detection        ├── Claude Vision (EYES)
├── Comment processing              ├── Claude text gen (scripts)
└── Intent → route to agents        ├── ElevenLabs (voice)
                                    ├── HeyGen LiveAvatar (avatar)
                                    ├── rembg (photo cleanup)
                                    └── React dashboard
```

## Build Order (Do This Exact Sequence)

### TONIGHT: Spike (2 hours)
```bash
brew install cactus-compute/cactus/cactus
cactus run google/gemma-4-E4B-it
```
- Test: "sell this for $49" → does it understand?
- Test: point at 3 products → describe them
- Sign up: Claude API, ElevenLabs, HeyGen

### WEDNESDAY: EYES (5 hours)
- FastAPI scaffold
- WebSocket phone ↔ laptop
- Claude Vision product analysis
- Test: frame → JSON

### THURSDAY: SELLER (10 hours)
- Claude → sales script
- ElevenLabs → speech
- HeyGen → avatar
- Comment → response loop
- Test: product JSON → avatar selling + responding

### FRIDAY: Dashboard + Polish (7 hours)
- React dashboard
- All panels wired up
- rembg product photos
- End-to-end rehearsal

### SATURDAY: At Hackathon (3+ hours)
- Final polish
- Demo rehearsal
- Stretch goals if time

## Critical Path
Phases 1-3 = working demo (no dashboard needed, show on terminal)
Phase 4 = pretty demo
Phase 5+ = wow factor (3D model, showroom)
