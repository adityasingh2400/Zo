#!/usr/bin/env node
/**
 * End-to-end smoke test. Mimics a phone uploading a video over the WS:
 *   start → N binary chunks → end → expect "saved" ack → check file on disk.
 *
 * Usage:
 *   node scripts/smoke.js                 # uses https://localhost:8443
 *   PORT=8443 node scripts/smoke.js
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

// ~ 1 MiB of pseudo-random bytes split into 8 chunks
const CHUNK_BYTES = 128 * 1024;
const CHUNK_COUNT = 8;
const TOTAL = CHUNK_BYTES * CHUNK_COUNT;

function makeChunks() {
  const chunks = [];
  for (let i = 0; i < CHUNK_COUNT; i++) {
    chunks.push(crypto.randomBytes(CHUNK_BYTES));
  }
  return chunks;
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function main() {
  console.log(`smoke: connecting to ${URL}`);

  const ws = new WebSocket(URL, { rejectUnauthorized: false });

  await new Promise((resolve, reject) => {
    ws.once("open", resolve);
    ws.once("error", reject);
  });

  const sessionId = "smoke-" + crypto.randomBytes(3).toString("hex");
  const label = "smoke-test";
  const messages = [];
  let savedMsg = null;
  let startedMsg = null;

  ws.on("message", (data, isBinary) => {
    if (isBinary) return;
    const msg = JSON.parse(data.toString());
    messages.push(msg);
    if (msg.type === "started") startedMsg = msg;
    if (msg.type === "saved") savedMsg = msg;
  });

  ws.send(
    JSON.stringify({
      type: "start",
      sessionId,
      mime: "video/mp4",
      ext: "mp4",
      label,
      ts: Date.now(),
    })
  );

  // wait briefly for "started"
  for (let i = 0; i < 30 && !startedMsg; i++) await sleep(20);
  if (!startedMsg) {
    console.error("smoke: never got 'started' ack");
    process.exit(2);
  }
  console.log(`smoke: started session ${sessionId} → ${startedMsg.fileName}`);

  const startedAt = Date.now();
  const chunks = makeChunks();
  for (const c of chunks) {
    ws.send(c);
    // tiny gap to mimic real recorder cadence
    await sleep(20);
  }

  ws.send(
    JSON.stringify({
      type: "end",
      sessionId,
      totalBytes: TOTAL,
    })
  );

  // wait for saved ack
  for (let i = 0; i < 100 && !savedMsg; i++) await sleep(20);
  ws.close();

  if (!savedMsg) {
    console.error("smoke: never got 'saved' ack");
    process.exit(3);
  }

  const elapsed = Date.now() - startedAt;
  console.log(
    `smoke: server saved ${savedMsg.fileName} (${savedMsg.bytes} bytes, ${savedMsg.chunks} chunks)  ${elapsed}ms`
  );

  const filePath = path.join(OUTPUT_DIR, savedMsg.fileName);
  if (!fs.existsSync(filePath)) {
    console.error(`smoke: FAIL — file missing on disk: ${filePath}`);
    process.exit(4);
  }
  const stat = fs.statSync(filePath);
  if (stat.size !== TOTAL) {
    console.error(`smoke: FAIL — size mismatch: disk=${stat.size} expected=${TOTAL}`);
    process.exit(5);
  }
  if (savedMsg.bytes !== TOTAL) {
    console.error(`smoke: FAIL — ack bytes mismatch: ack=${savedMsg.bytes} expected=${TOTAL}`);
    process.exit(6);
  }

  // clean up the smoke artifact
  try { fs.unlinkSync(filePath); } catch {}

  console.log(`smoke: OK — round-trip ${TOTAL.toLocaleString()} bytes in ${elapsed}ms.`);
}

main().catch((err) => {
  console.error("smoke: error", err);
  process.exit(1);
});
