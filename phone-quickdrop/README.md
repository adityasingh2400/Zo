# phone-quickdrop

Lightweight phone → laptop video drop. No app to install. Phone records,
file lands in `~/Desktop/PhoneCaptures/` the moment you tap Stop.

## Why

AirDrop has a discovery + handshake step that runs after you tap Send,
so even small clips feel slow. quickdrop keeps a WebSocket open the
whole time you're recording and streams chunks every 250 ms — so when
you tap Stop, ~95% of the bytes are already on your laptop and only the
last quarter-second has to fly. On LAN that's effectively instant.

## Run

```bash
cd phone-quickdrop
npm install
npm start
```

The server prints a URL and a QR code:

```
phone-quickdrop is up.
  output dir : /Users/you/Desktop/PhoneCaptures
  port       : 8443

  Open this on your phone (same Wi-Fi or USB tether):
    https://192.168.1.42:8443

  QR (scan with your phone camera):
    █▀▀▀▀▀█  ▀▀ ▄ █▀▀▀▀▀█
    ...
```

Open it on your phone, accept the self-signed cert warning **once**, allow
camera + mic, tap the red button. Tap again to stop. The file is on
your laptop before the button finishes its animation.

## Wired (USB) mode

Faster, more reliable, totally optional.

- **iPhone:** plug in via USB → Settings → Personal Hotspot → enable.
  Your laptop now sits on the iPhone's tether net (e.g. `172.20.10.x`).
  Restart `npm start` so the cert picks up the new IP, then open the
  URL on the phone — same code path.
- **Android:** plug in via USB → Settings → Network → USB tethering.
  Same flow.

The WebSocket doesn't care whether the underlying link is Wi-Fi, USB
tether, or carrier pigeon, as long as the phone can reach the laptop's
IP on port 8443.

## Where files land

`~/Desktop/PhoneCaptures/<timestamp>_<random>.<ext>`

Override with `QUICKDROP_OUT=/some/dir npm start`.

## Wire protocol

One WebSocket can handle many recordings back-to-back. Per recording:

1. Text frame: `{"type":"start","sessionId":"…","mime":"video/mp4","ext":"mp4","ts":…}`
2. N binary frames (raw `MediaRecorder` chunks)
3. Text frame: `{"type":"end","sessionId":"…","totalBytes":…}`

Server acks: `hello`, `started`, `progress` (every ~1 MB), `saved`, `error`.

## Smoke test

Boots a synthetic phone client, sends 1 MiB of random bytes, checks the
file landed at the right size:

```bash
npm start                  # in one shell
npm run smoke              # in another
# → smoke: OK — round-trip 1,048,576 bytes in ~250ms
```

## Files

- `server.js` — Express + ws + auto self-signed cert + per-recording file writer.
- `public/index.html` — single-file phone UI (no build step).
- `scripts/smoke.js` — end-to-end synthetic-phone test.
