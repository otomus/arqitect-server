/**
 * Synaptic Bridge — WebSocket server that bridges Redis Pub/Sub to the browser.
 * Subscribes to brain:* and nerve:* channels, forwards to connected clients.
 * Accepts tasks from browser and publishes to brain:task channel.
 */

const { WebSocketServer } = require("ws");
const { createClient } = require("redis");
const http = require("http");
const https = require("https");
const fs = require("fs");
const path = require("path");
const os = require("os");

const PORT = 3000;
const REDIS_CHANNELS = [
  "brain:thought",
  "brain:action",
  "brain:response",
  "nerve:result",
  "nerve:qualification",
  "system:status",
  "system:kill",
  "memory:update",
  "sense:calibration",
  "sense:sight:frame",
  "sense:stt:result",
  "brain:audio",
  "sense:config",
  "brain:task",
  "mcp:tool_call",
];

// --- SSL support: use HTTPS when cert/key are configured ---
const SSL_CERT = process.env.SSL_CERT || "";
const SSL_KEY = process.env.SSL_KEY || "";

function createServer(handler) {
  if (SSL_CERT && SSL_KEY) {
    return https.createServer({
      cert: fs.readFileSync(SSL_CERT),
      key: fs.readFileSync(SSL_KEY),
    }, handler);
  }
  return http.createServer(handler);
}

// --- HTTP(S) server for serving the dashboard ---
const server = createServer((req, res) => {
  let filePath;
  if (req.url === "/" || req.url === "/index.html") {
    filePath = path.join(__dirname, "index.html");
  } else {
    filePath = path.join(__dirname, req.url);
  }

  // Prevent path traversal — resolved path must stay within __dirname
  const resolved = path.resolve(filePath);
  if (!resolved.startsWith(path.resolve(__dirname))) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }

  const ext = path.extname(filePath);
  const mimeTypes = {
    ".html": "text/html",
    ".js": "text/javascript",
    ".css": "text/css",
  };

  fs.readFile(resolved, (err, data) => {
    if (err) {
      res.writeHead(404);
      res.end("Not found");
      return;
    }
    res.writeHead(200, { "Content-Type": mimeTypes[ext] || "text/plain" });
    res.end(data);
  });
});

// --- WebSocket server ---
const wss = new WebSocketServer({ server });

// --- Redis clients ---
let redisSub;
let redisPub;

async function startRedis() {
  redisSub = createClient();
  redisPub = createClient();

  redisSub.on("error", (err) => console.error("[REDIS-SUB]", err.message));
  redisPub.on("error", (err) => console.error("[REDIS-PUB]", err.message));

  await redisSub.connect();
  await redisPub.connect();

  // Subscribe to all channels
  for (const channel of REDIS_CHANNELS) {
    await redisSub.subscribe(channel, (message, ch) => {
      let data;
      try {
        data = JSON.parse(message);
      } catch (e) {
        console.warn(`[BRIDGE] Malformed message on ${ch}:`, e.message);
        return;
      }
      // Resolve image_path to base64 for browser clients
      if (data.media && data.media.image_path && !data.media.image_b64) {
        try {
          if (fs.existsSync(data.media.image_path)) {
            const buf = fs.readFileSync(data.media.image_path);
            data.media.image_b64 = buf.toString("base64");
          }
        } catch (e) {
          console.warn("[BRIDGE] Failed to read image_path:", e.message);
        }
      }
      const payload = JSON.stringify({ channel: ch, data });
      wss.clients.forEach((client) => {
        if (client.readyState === 1) {
          client.send(payload);
        }
      });
    });
  }

  console.log("[BRIDGE] Subscribed to Redis channels:", REDIS_CHANNELS.join(", "));
}

// --- System stats for Pain Gauge ---
function getSystemStats() {
  const cpus = os.cpus();
  const loadAvg = os.loadavg();
  const totalMem = os.totalmem();
  const freeMem = os.freemem();
  return {
    cpuLoad: loadAvg[0] / cpus.length, // normalized 0-1ish
    memoryUsed: ((totalMem - freeMem) / totalMem) * 100,
    totalMem: Math.round(totalMem / 1024 / 1024),
    freeMem: Math.round(freeMem / 1024 / 1024),
    uptime: os.uptime(),
  };
}

// Broadcast system stats every 2 seconds
setInterval(async () => {
  const stats = getSystemStats();
  const payload = JSON.stringify({ channel: "system:stats", data: stats });
  wss.clients.forEach((client) => {
    if (client.readyState === 1) {
      client.send(payload);
    }
  });
}, 2000);

// Broadcast memory state every 3 seconds (poll Redis session)
setInterval(async () => {
  if (!redisPub) return;
  try {
    const session = await redisPub.hGetAll("synapse:session");
    const convoRaw = await redisPub.lRange("synapse:conversation", -10, -1);
    const conversation = convoRaw.map((s) => { try { return JSON.parse(s); } catch { return null; } }).filter(Boolean);
    const payload = JSON.stringify({
      channel: "memory:state",
      data: { session, conversation },
    });
    wss.clients.forEach((client) => {
      if (client.readyState === 1) {
        client.send(payload);
      }
    });
  } catch (err) {
    // ignore
  }
}, 3000);

// Broadcast sense calibration status every 5 seconds (poll Redis hash)
setInterval(async () => {
  if (!redisPub) return;
  try {
    const raw = await redisPub.hGetAll("synapse:sense_calibration");
    if (!raw || Object.keys(raw).length === 0) return;
    const senses = {};
    for (const [name, json] of Object.entries(raw)) {
      try { senses[name] = JSON.parse(json); } catch { /* skip */ }
    }
    const payload = JSON.stringify({ channel: "sense:status", data: senses });
    wss.clients.forEach((client) => {
      if (client.readyState === 1) {
        client.send(payload);
      }
    });
  } catch (err) {
    // ignore
  }
}, 5000);

// --- Handle incoming WebSocket messages ---
wss.on("connection", (ws) => {
  console.log("[BRIDGE] Client connected");

  // Send current stats immediately
  ws.send(JSON.stringify({ channel: "system:stats", data: getSystemStats() }));

  // Send current calibration status immediately
  (async () => {
    if (!redisPub) return;
    try {
      const raw = await redisPub.hGetAll("synapse:sense_calibration");
      if (raw && Object.keys(raw).length > 0) {
        const senses = {};
        for (const [name, json] of Object.entries(raw)) {
          try { senses[name] = JSON.parse(json); } catch { /* skip */ }
        }
        ws.send(JSON.stringify({ channel: "sense:status", data: senses }));
      }
    } catch { /* ignore */ }

    // Send saved sense config values
    try {
      const config = await redisPub.hGetAll("synapse:sense_config");
      if (config && Object.keys(config).length > 0) {
        ws.send(JSON.stringify({ channel: "sense:saved_config", data: config }));
      }
    } catch { /* ignore */ }

    // Send nerve status on connect
    try {
      const nerveRaw = await redisPub.get("synapse:nerve_status");
      if (nerveRaw) {
        ws.send(JSON.stringify({ channel: "nerve:qualification", data: JSON.parse(nerveRaw) }));
      }
    } catch { /* ignore */ }
  })();

  ws.on("message", async (raw) => {
    try {
      const msg = JSON.parse(raw.toString());

      if (msg.type === "task") {
        // Publish task to Redis for the brain to pick up
        const payload = {
          task: msg.task,
          source: "dashboard",
          connector_user_id: msg.user_id || "",
        };
        if (msg.location) payload.location = msg.location;
        await redisPub.publish("brain:task", JSON.stringify(payload));
        console.log("[BRIDGE] Task published:", msg.task);
      } else if (msg.type === "peek") {
        // Request a sight peek (screenshot or camera)
        const source = msg.source || "screenshot";
        await redisPub.publish("sense:peek", JSON.stringify({ sense: "sight", source }));
        console.log(`[BRIDGE] Sight peek requested (source: ${source})`);
      } else if (msg.type === "voice") {
        // Voice message — audio base64 for STT
        await redisPub.publish("sense:voice", JSON.stringify({ audio_b64: msg.audio_b64, mime: msg.mime || "audio/webm" }));
        console.log("[BRIDGE] Voice message forwarded for STT");
      } else if (msg.type === "image") {
        // Image message — base64 for sight analysis
        await redisPub.publish("sense:image", JSON.stringify({ image_b64: msg.image_b64, prompt: msg.prompt || "Describe this image in detail." }));
        console.log("[BRIDGE] Image forwarded for sight analysis");
      } else if (msg.type === "sense_config") {
        // Sense configuration — save to Redis hash and publish for Python side
        const configKey = `${msg.sense}.${msg.key}`;
        await redisPub.hSet("synapse:sense_config", configKey, msg.value);
        await redisPub.publish("sense:config", JSON.stringify({
          sense: msg.sense, key: msg.key, value: msg.value,
        }));
        console.log(`[BRIDGE] Sense config: ${configKey} = ${msg.value}`);
      } else if (msg.type === "nerve_details") {
        // Fetch nerve details from Redis hash
        const name = msg.name;
        if (name) {
          const raw = await redisPub.hGet("synapse:nerve_details", name);
          if (raw) {
            ws.send(JSON.stringify({ channel: "nerve:details", data: JSON.parse(raw) }));
          } else {
            ws.send(JSON.stringify({ channel: "nerve:details", data: { name, error: "not found" } }));
          }
        }
      } else if (msg.type === "kill") {
        // Kill switch — publish kill signal then stop Redis
        await redisPub.publish("system:kill", JSON.stringify({ reason: "Kill switch activated" }));
        console.log("[BRIDGE] KILL SWITCH ACTIVATED");
      }
    } catch (err) {
      console.error("[BRIDGE] Bad message:", err.message);
    }
  });

  ws.on("close", () => console.log("[BRIDGE] Client disconnected"));
});

// --- Start ---
startRedis()
  .then(() => {
    const scheme = SSL_CERT ? "https" : "http";
    server.listen(PORT, () => {
      console.log(`[BRIDGE] Synapse-UI: ${scheme}://localhost:${PORT}`);
      console.log(`[BRIDGE] WebSocket ready on same port (${SSL_CERT ? "wss" : "ws"})`);
    });
  })
  .catch((err) => {
    console.error("[BRIDGE] Failed to connect to Redis:", err.message);
    process.exit(1);
  });
