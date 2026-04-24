#!/usr/bin/env node
/**
 * Reconnect smoke test.
 *
 * Simulates a phone whose WebSocket dies in the middle of a recording:
 *   1. Open WS, send "start" with sessionId S
 *   2. Send N/2 chunks
 *   3. Abruptly close the socket (not a graceful "end")
 *   4. Open a new WS, send "start" with same sessionId and resume:true
 *   5. Expect ack { type:"started", resumed:true, bytes: <what server has} }
 *   6. Send the remaining N/2 chunks, then "end"
 *   7. Verify disk file is exactly the concatenation of all chunks, in order
 *
 * Usage:
 *   node scripts/smoke-reconnect.js                 # defaults to localhost:8443
 *   PORT=8444 node scripts/smoke-reconnect.js       # worktree port
 */

const fs = require("fs");
const path = require("path");
const os = require("os");
const crypto = require("crypto");
const WebSocket = require("ws");

const PORT = parseInt(process.env.PORT || "8443", 10);
const HOST = process.env.HOST || "localhost";
const URL = `wss://${HOST}:${PORT}/ws`;
const OUTPUT_DIR =
  process.env.QUICKDROP_OUT ||
  path.join(os.homedir(), "Desktop", "PhoneCaptures");

const CHUNK_BYTES = 128 * 1024;
const CHUNK_COUNT = 8; // 4 before disconnect, 4 after
const TOTAL = CHUNK_BYTES * CHUNK_COUNT;

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

function connect() {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(URL, { rejectUnauthorized: false });
    ws.once("open", () => resolve(ws));
    ws.once("error", reject);
  });
}

function collectMessages(ws) {
  const messages = [];
  ws.on("message", (data, isBinary) => {
    if (isBinary) return;
    try { messages.push(JSON.parse(data.toString())); } catch {}
  });
  return messages;
}

async function waitFor(messages, type, timeoutMs = 2000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const m = messages.find((x) => x.type === type);
    if (m) return m;
    await sleep(10);
  }
  return null;
}

async function main() {
  console.log(`smoke-reconnect: connecting to ${URL}`);

  // Pre-generate all chunks so we know exactly what bytes should land on disk.
  const chunks = [];
  for (let i = 0; i < CHUNK_COUNT; i++) {
    chunks.push(crypto.randomBytes(CHUNK_BYTES));
  }
  const expectedBuffer = Buffer.concat(chunks);

  const sessionId = "reconnect-" + crypto.randomBytes(3).toString("hex");
  const label = "reconnect-test";

  // ---- phase 1: open, start, send half, kill ----
  let ws = await connect();
  let msgs = collectMessages(ws);

  ws.send(JSON.stringify({
    type: "start",
    sessionId,
    mime: "video/mp4",
    ext: "mp4",
    label,
    ts: Date.now(),
  }));

  const started1 = await waitFor(msgs, "started", 2000);
  if (!started1 || started1.resumed) {
    console.error(`smoke-reconnect: FAIL — expected fresh 'started' ack, got ${JSON.stringify(started1)}`);
    process.exit(2);
  }
  const fileName = started1.fileName;
  console.log(`smoke-reconnect: started ${sessionId} → ${fileName}`);

  for (let i = 0; i < CHUNK_COUNT / 2; i++) {
    ws.send(chunks[i]);
    await sleep(15);
  }

  // Give the server a beat to flush the binary frames to disk before we cut
  // the socket — otherwise we're testing TCP close semantics, not resume.
  await sleep(80);

  // Abrupt close — terminate, not close() — so server sees an unclean drop.
  ws.terminate();
  console.log(`smoke-reconnect: killed WS after ${CHUNK_COUNT / 2} chunks`);

  // Wait briefly for server to register the close and orphan the session.
  await sleep(200);

  // ---- phase 2: reconnect with resume ----
  ws = await connect();
  msgs = collectMessages(ws);

  ws.send(JSON.stringify({
    type: "start",
    sessionId,
    mime: "video/mp4",
    ext: "mp4",
    resume: true,
    ts: Date.now(),
  }));

  const started2 = await waitFor(msgs, "started", 2000);
  if (!started2) {
    console.error("smoke-reconnect: FAIL — no 'started' ack after resume");
    process.exit(3);
  }
  if (!started2.resumed) {
    console.error(`smoke-reconnect: FAIL — expected resumed:true, got ${JSON.stringify(started2)}`);
    process.exit(4);
  }
  const halfBytes = (CHUNK_COUNT / 2) * CHUNK_BYTES;
  if (typeof started2.bytes !== "number" || started2.bytes < 1 || started2.bytes > halfBytes) {
    console.error(
      `smoke-reconnect: FAIL — resume bytes unreasonable: got ${started2.bytes}, expected 1..${halfBytes}`
    );
    process.exit(5);
  }
  console.log(
    `smoke-reconnect: resumed ${sessionId} (server already has ${started2.bytes} bytes)`
  );

  // ---- phase 3: send the rest ----
  for (let i = CHUNK_COUNT / 2; i < CHUNK_COUNT; i++) {
    ws.send(chunks[i]);
    await sleep(15);
  }
  ws.send(JSON.stringify({
    type: "end",
    sessionId,
    totalBytes: TOTAL,
  }));

  const saved = await waitFor(msgs, "saved", 3000);
  ws.close();
  if (!saved) {
    console.error("smoke-reconnect: FAIL — no 'saved' ack");
    process.exit(6);
  }

  // ---- phase 4: verify on-disk bytes ----
  const filePath = path.join(OUTPUT_DIR, fileName);
  if (!fs.existsSync(filePath)) {
    console.error(`smoke-reconnect: FAIL — file missing: ${filePath}`);
    process.exit(7);
  }
  const onDisk = fs.readFileSync(filePath);
  if (onDisk.length !== TOTAL) {
    console.error(
      `smoke-reconnect: FAIL — size mismatch: disk=${onDisk.length} expected=${TOTAL}`
    );
    process.exit(8);
  }
  if (!onDisk.equals(expectedBuffer)) {
    // Find first divergence for debug
    let i = 0;
    while (i < onDisk.length && onDisk[i] === expectedBuffer[i]) i++;
    console.error(
      `smoke-reconnect: FAIL — content mismatch (first diff at byte ${i})`
    );
    process.exit(9);
  }

  try { fs.unlinkSync(filePath); } catch {}

  console.log(
    `smoke-reconnect: OK — ${TOTAL.toLocaleString()} bytes landed correctly across a mid-session disconnect.`
  );
}

main().catch((err) => {
  console.error("smoke-reconnect: error", err);
  process.exit(1);
});
