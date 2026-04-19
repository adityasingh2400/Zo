#!/usr/bin/env node
/**
 * phone-quickdrop server
 *
 * Single HTTPS server. Three things on one URL:
 *
 *   /         → laptop dashboard (QR + live recordings list)
 *   /phone    → phone recorder UI (what the QR encodes)
 *   /ws       → WebSocket: phones upload, dashboards subscribe
 *
 * Self-signed cert is generated on first run and cached in .certs/, so
 * each device only has to accept the warning once.
 *
 * Wire protocol (per recording, on a single phone WebSocket):
 *   1. text: {"type":"start","sessionId":"…","mime":"video/mp4","ext":"mp4","ts":…}
 *   2. N binary frames: raw MediaRecorder chunks
 *   3. text: {"type":"end","sessionId":"…","totalBytes":…}  (or "abort")
 *
 * Observer subscription (dashboard):
 *   1. text: {"type":"subscribe_observer"}
 *   → server sends {"type":"snapshot","recordings":[…],"phonesConnected":N}
 *   → server pushes {"type":"recording_saved",…},
 *     {"type":"phone_status",connected:N} on every change.
 */

const fs = require("fs");
const os = require("os");
const path = require("path");
const https = require("https");
const crypto = require("crypto");
const { exec } = require("child_process");
const express = require("express");
const { WebSocketServer } = require("ws");
const selfsigned = require("selfsigned");
const QRCode = require("qrcode");
const qrcodeTerminal = require("qrcode-terminal");

const PORT = parseInt(process.env.PORT || "8443", 10);
const PROJECT_DIR = __dirname;
const CERT_DIR = path.join(PROJECT_DIR, ".certs");
const PUBLIC_DIR = path.join(PROJECT_DIR, "public");
const OUTPUT_DIR =
  process.env.QUICKDROP_OUT ||
  path.join(os.homedir(), "Desktop", "PhoneCaptures");
const RECENT_LIMIT = 50;

fs.mkdirSync(OUTPUT_DIR, { recursive: true });
fs.mkdirSync(CERT_DIR, { recursive: true });

// ---------- local IPs ----------
function getLocalIPv4s() {
  const out = [];
  const ifaces = os.networkInterfaces();
  for (const name of Object.keys(ifaces)) {
    for (const info of ifaces[name] || []) {
      if (info.family === "IPv4" && !info.internal) {
        out.push({ name, address: info.address });
      }
    }
  }
  return out;
}
function pickPrimaryIP(ips) {
  return (
    ips.find((i) => i.name === "en0") ||
    ips.find((i) => i.name === "en1") ||
    ips[0] ||
    null
  );
}

// ---------- self-signed cert ----------
function loadOrGenerateCert(ips) {
  const keyPath = path.join(CERT_DIR, "key.pem");
  const certPath = path.join(CERT_DIR, "cert.pem");
  const metaPath = path.join(CERT_DIR, "meta.json");

  const wantSans = new Set(["localhost", "127.0.0.1", ...ips.map((i) => i.address)]);
  let regenerate = true;

  if (fs.existsSync(keyPath) && fs.existsSync(certPath) && fs.existsSync(metaPath)) {
    try {
      const meta = JSON.parse(fs.readFileSync(metaPath, "utf8"));
      const haveSans = new Set(meta.sans || []);
      const ageDays = (Date.now() - (meta.createdAt || 0)) / 86_400_000;
      const sansMatch =
        wantSans.size === haveSans.size && [...wantSans].every((s) => haveSans.has(s));
      if (sansMatch && ageDays < 300) regenerate = false;
    } catch {
      regenerate = true;
    }
  }

  if (regenerate) {
    const attrs = [{ name: "commonName", value: "phone-quickdrop" }];
    const altNames = [...wantSans].map((value) => {
      const isIp = /^\d+\.\d+\.\d+\.\d+$/.test(value);
      return { type: isIp ? 7 : 2, [isIp ? "ip" : "value"]: value };
    });
    const pems = selfsigned.generate(attrs, {
      keySize: 2048,
      days: 365,
      algorithm: "sha256",
      extensions: [
        { name: "basicConstraints", cA: true },
        { name: "subjectAltName", altNames },
      ],
    });
    fs.writeFileSync(keyPath, pems.private);
    fs.writeFileSync(certPath, pems.cert);
    fs.writeFileSync(
      metaPath,
      JSON.stringify({ createdAt: Date.now(), sans: [...wantSans] }, null, 2)
    );
  }

  return { key: fs.readFileSync(keyPath), cert: fs.readFileSync(certPath) };
}

// ---------- helpers ----------
function tsName() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` +
    `T${pad(d.getHours())}-${pad(d.getMinutes())}-${pad(d.getSeconds())}` +
    `_${crypto.randomBytes(2).toString("hex")}`
  );
}
function fmtBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(2)} MB`;
}
function safeExt(input) {
  const e = String(input || "").toLowerCase().replace(/[^a-z0-9]/g, "");
  return e ? e.slice(0, 6) : "bin";
}
function safeLabel(input) {
  return String(input || "")
    .toLowerCase()
    .replace(/[^a-z0-9-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 32);
}

// ---------- observer broadcast ----------
const observers = new Set();
const phones = new Set();
const recentRecordings = []; // newest first

function broadcastToObservers(obj) {
  const payload = JSON.stringify(obj);
  for (const ws of observers) {
    if (ws.readyState === ws.OPEN) {
      try { ws.send(payload); } catch { /* ignore */ }
    }
  }
}
function pushPhoneStatus() {
  broadcastToObservers({ type: "phone_status", connected: phones.size });
}

// ---------- per-recording session state ----------
class RecordingSession {
  constructor({ sessionId, mime, ext, label, startedAt, peerLabel }) {
    this.sessionId = sessionId;
    this.mime = mime || "application/octet-stream";
    this.ext = safeExt(ext);
    this.label = safeLabel(label);
    this.startedAt = startedAt || Date.now();
    this.peerLabel = peerLabel;
    const base = this.label
      ? `${tsName()}_${this.label}.${this.ext}`
      : `${tsName()}.${this.ext}`;
    this.filePath = path.join(OUTPUT_DIR, base);
    this.fileName = base;
    this.bytes = 0;
    this.chunks = 0;
    this.stream = fs.createWriteStream(this.filePath);
    this.closed = false;
  }
  appendBinary(buf) {
    if (this.closed) return;
    this.bytes += buf.byteLength;
    this.chunks += 1;
    this.stream.write(Buffer.from(buf));
  }
  finish(reason = "ok") {
    return new Promise((resolve) => {
      if (this.closed) return resolve();
      this.closed = true;
      this.stream.end(() => {
        const ms = Date.now() - this.startedAt;
        const sizeStr = fmtBytes(this.bytes);
        const tag = reason === "ok" ? "saved" : reason;

        if ((reason === "abort" || reason === "partial") && this.bytes === 0) {
          fs.unlink(this.filePath, () => {});
          console.log(
            `[${this.peerLabel}] ${tag} ${this.sessionId} (empty, deleted)`
          );
          resolve();
          return;
        }

        console.log(
          `[${this.peerLabel}] ${tag} ${this.fileName}  ${sizeStr}  ` +
            `${this.chunks} chunks  ${ms}ms`
        );
        console.log(`           → ${this.filePath}`);

        const record = {
          fileName: this.fileName,
          filePath: this.filePath,
          bytes: this.bytes,
          chunks: this.chunks,
          mime: this.mime,
          ext: this.ext,
          ms,
          ts: Date.now(),
          peer: this.peerLabel,
          partial: reason !== "ok",
        };
        recentRecordings.unshift(record);
        if (recentRecordings.length > RECENT_LIMIT) {
          recentRecordings.length = RECENT_LIMIT;
        }
        broadcastToObservers({ type: "recording_saved", recording: record });
        resolve();
      });
    });
  }
}

// ---------- HTTP/WebSocket server ----------
async function start() {
  const ips = getLocalIPv4s();
  const primary = pickPrimaryIP(ips);
  const { key, cert } = loadOrGenerateCert(ips);

  const phoneHost = primary ? primary.address : "localhost";
  const phoneUrl = `https://${phoneHost}:${PORT}/phone`;
  const dashUrl = `https://localhost:${PORT}/`;

  // Pre-render QR as inline SVG.
  const qrSvg = await QRCode.toString(phoneUrl, {
    type: "svg",
    errorCorrectionLevel: "M",
    margin: 1,
    color: { dark: "#ffffff", light: "#00000000" },
  });

  // Read dashboard template once at boot, substitute.
  const dashTpl = fs.readFileSync(path.join(PUBLIC_DIR, "desktop.html"), "utf8");
  const renderDashboard = () => {
    return dashTpl
      .replace(/\{\{QR_SVG\}\}/g, qrSvg)
      .replace(/\{\{PHONE_URL\}\}/g, phoneUrl)
      .replace(/\{\{OUTPUT_DIR\}\}/g, OUTPUT_DIR)
      .replace(/\{\{PORT\}\}/g, String(PORT));
  };

  const app = express();
  app.disable("x-powered-by");
  app.get("/", (_req, res) => {
    res.setHeader("Content-Type", "text/html; charset=utf-8");
    res.setHeader("Cache-Control", "no-store");
    res.send(renderDashboard());
  });
  app.get("/phone", (_req, res) => {
    res.setHeader("Cache-Control", "no-store");
    res.sendFile(path.join(PUBLIC_DIR, "phone.html"));
  });
  app.get("/qr.svg", (_req, res) => {
    res.setHeader("Content-Type", "image/svg+xml; charset=utf-8");
    res.setHeader("Cache-Control", "no-store");
    res.send(qrSvg);
  });
  app.get("/health", (_req, res) =>
    res.json({ ok: true, outputDir: OUTPUT_DIR, phoneUrl })
  );
  app.post("/api/reveal", express.json({ limit: "8kb" }), (req, res) => {
    // Only allow revealing files inside OUTPUT_DIR.
    const target = path.resolve(String(req.body?.path || ""));
    if (!target.startsWith(path.resolve(OUTPUT_DIR) + path.sep) || !fs.existsSync(target)) {
      return res.status(400).json({ ok: false, reason: "invalid_path" });
    }
    if (process.platform === "darwin") {
      exec(`open -R ${JSON.stringify(target)}`);
    } else if (process.platform === "win32") {
      exec(`explorer.exe /select,${JSON.stringify(target)}`);
    } else {
      exec(`xdg-open ${JSON.stringify(path.dirname(target))}`);
    }
    res.json({ ok: true });
  });

  const server = https.createServer({ key, cert }, app);
  const wss = new WebSocketServer({
    server,
    path: "/ws",
    maxPayload: 64 * 1024 * 1024,
  });

  wss.on("connection", (ws, req) => {
    const peer = req.socket.remoteAddress?.replace(/^::ffff:/, "") || "client";
    let role = null; // "phone" | "observer"
    let session = null;

    const ack = (obj) => {
      try { ws.send(JSON.stringify(obj)); } catch { /* ignore */ }
    };

    ws.on("message", async (data, isBinary) => {
      // Binary frames are recording chunks (phone role only).
      if (isBinary) {
        if (role !== "phone" || !session) {
          ack({ type: "error", reason: "binary_without_session" });
          return;
        }
        session.appendBinary(data);
        if (session.bytes - (session._lastAckBytes || 0) > 1024 * 1024) {
          session._lastAckBytes = session.bytes;
          ack({
            type: "progress",
            sessionId: session.sessionId,
            bytes: session.bytes,
            chunks: session.chunks,
          });
        }
        return;
      }

      let msg;
      try { msg = JSON.parse(data.toString()); } catch {
        ack({ type: "error", reason: "bad_json" });
        return;
      }

      if (msg.type === "subscribe_observer") {
        role = "observer";
        observers.add(ws);
        ack({
          type: "snapshot",
          recordings: recentRecordings.slice(0, RECENT_LIMIT),
          phonesConnected: phones.size,
          outputDir: OUTPUT_DIR,
          phoneUrl,
        });
        console.log(`[${peer}] observer connected`);
        return;
      }

      if (msg.type === "start") {
        if (role !== "phone") {
          role = "phone";
          phones.add(ws);
          pushPhoneStatus();
          console.log(`[${peer}] phone connected`);
        }
        if (session && !session.closed) await session.finish("abort");
        session = new RecordingSession({
          sessionId: msg.sessionId || crypto.randomUUID(),
          mime: msg.mime,
          ext: msg.ext,
          label: msg.label,
          startedAt: Date.now(),
          peerLabel: peer,
        });
        console.log(
          `[${peer}] start ${session.sessionId} → ${session.fileName} (${session.mime})`
        );
        ack({ type: "started", sessionId: session.sessionId, fileName: session.fileName });
        return;
      }
      if (msg.type === "end") {
        if (!session) { ack({ type: "error", reason: "end_without_start" }); return; }
        const expected = msg.totalBytes;
        await session.finish("ok");
        ack({
          type: "saved",
          sessionId: session.sessionId,
          fileName: session.fileName,
          bytes: session.bytes,
          chunks: session.chunks,
          expectedBytes: typeof expected === "number" ? expected : undefined,
        });
        session = null;
        return;
      }
      if (msg.type === "abort") {
        if (session) { await session.finish("abort"); session = null; }
        ack({ type: "aborted" });
        return;
      }
      if (msg.type === "ping") { ack({ type: "pong", ts: Date.now() }); return; }

      ack({ type: "error", reason: `unknown_type:${msg.type}` });
    });

    ws.on("close", async () => {
      if (session && !session.closed) {
        await session.finish("partial");
        session = null;
      }
      if (role === "phone") {
        phones.delete(ws);
        pushPhoneStatus();
        console.log(`[${peer}] phone disconnected`);
      } else if (role === "observer") {
        observers.delete(ws);
        console.log(`[${peer}] observer disconnected`);
      }
    });
    ws.on("error", (err) => {
      console.error(`[${peer}] ws error:`, err.message);
    });

    ack({ type: "hello", outputDir: OUTPUT_DIR, phoneUrl });
  });

  server.on("error", (err) => {
    if (err.code === "EADDRINUSE") {
      console.error(`\nPort ${PORT} is already in use. Set PORT=<other> and try again.\n`);
      process.exit(1);
    }
    throw err;
  });

  server.listen(PORT, "0.0.0.0", () => {
    const otherUrls = ips
      .map((i) => `https://${i.address}:${PORT}/`)
      .filter((u) => u !== `https://${phoneHost}:${PORT}/`);

    console.log("\nphone-quickdrop is up.");
    console.log(`  output dir : ${OUTPUT_DIR}`);
    console.log(`  port       : ${PORT}\n`);
    console.log("  \u001b[1;32m▶  Open this on your laptop (one-time cert warning):\u001b[0m");
    console.log(`     \u001b[1;36m${dashUrl}\u001b[0m\n`);
    console.log("  The dashboard shows a big QR + live recordings list.");
    console.log("  Scan the QR with your phone to start recording.\n");
    console.log(`  (QR encodes: ${phoneUrl})\n`);
    console.log("  Terminal QR (fallback if you don't want to open the dashboard):\n");
    qrcodeTerminal.generate(phoneUrl, { small: true });
    if (otherUrls.length) {
      console.log("\n  Other dashboard addresses:");
      for (const u of otherUrls) console.log(`    ${u}`);
    }
    console.log("");
  });
}

start().catch((err) => {
  console.error("startup failed:", err);
  process.exit(1);
});
