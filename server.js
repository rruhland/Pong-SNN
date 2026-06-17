const http = require("http");
const fs = require("fs");
const path = require("path");

const root = __dirname;
const webRoot = path.join(root, "web");

const settings = {
  width: 800,
  height: 450,
  paddleWidth: 10,
  paddleHeight: 80,
  ballSize: 10,
  pollMs: 16,
  statePushMs: 16,
  fixedDt: 1 / 60,
};

const state = {
  sessionId: "local-1",
  resetToken: 0,
  seed: Date.now() >>> 0,
  mode: "human",
  apiDirection: 0,
  authoritativeTick: 0,
  eventSeq: 0,
  events: [],
  inputSeq: 0,
  inputEvents: [],
  score: { left: 0, right: 0 },
  running: false,
  ball: { x: settings.width / 2, y: settings.height / 2, vx: 0, vy: 0 },
  paddles: {
    leftY: (settings.height - settings.paddleHeight) / 2,
    rightY: (settings.height - settings.paddleHeight) / 2,
  },
  appliedInputSeq: 0,
  appliedEventSeq: 0,
  frameSeq: 0,
  renderedAt: 0,
  eventCamera: null,
  source: "backend",
  updatedAt: Date.now() / 1000,
};

const contentTypes = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
};

function sendJson(response, payload, status = 200) {
  const body = Buffer.from(JSON.stringify(payload));
  response.writeHead(status, {
    "Content-Type": "application/json",
    "Content-Length": body.length,
    "Access-Control-Allow-Origin": "*",
    "Cache-Control": "no-store",
  });
  response.end(body);
}

function readJson(request) {
  return new Promise((resolve, reject) => {
    let body = "";
    request.setEncoding("utf8");
    request.on("data", (chunk) => {
      body += chunk;
      if (body.length > 1_000_000) {
        reject(new Error("request body is too large"));
        request.destroy();
      }
    });
    request.on("end", () => {
      if (!body) {
        resolve({});
        return;
      }
      try {
        resolve(JSON.parse(body));
      } catch {
        reject(new Error("request body must be valid JSON"));
      }
    });
    request.on("error", reject);
  });
}

function parseDirection(value) {
  const numeric = Number.parseInt(value, 10);
  if (![-1, 0, 1].includes(numeric)) {
    throw new Error("direction must be -1, 0, or 1");
  }
  return numeric;
}

function parseTick(value, fallback) {
  if (value === undefined || value === null) {
    return fallback;
  }
  const numeric = Number.parseInt(value, 10);
  if (!Number.isFinite(numeric) || numeric < 0) {
    throw new Error("tick must be a non-negative integer");
  }
  return numeric;
}

function copyState() {
  return JSON.parse(JSON.stringify({ ...state, tick: state.authoritativeTick, settings }));
}

function normalizeEventCamera(value) {
  if (!value || typeof value !== "object") return null;
  const width = Number.parseInt(value.width, 10);
  const height = Number.parseInt(value.height, 10);
  if (!Number.isFinite(width) || width <= 0 || !Number.isFinite(height) || height <= 0) {
    return null;
  }
  const maxIndex = width * height;
  const pixels = Array.isArray(value.pixels)
    ? value.pixels
        .map((pixel) => Number.parseInt(pixel, 10))
        .filter((pixel) => Number.isInteger(pixel) && pixel >= 0 && pixel < maxIndex)
    : [];
  return {
    width,
    height,
    tick: Number.parseInt(value.tick ?? state.authoritativeTick, 10) || 0,
    frameSeq: Number.parseInt(value.frameSeq ?? state.frameSeq, 10) || 0,
    resetToken: value.resetToken,
    renderedAt: Number(value.renderedAt ?? state.renderedAt) || 0,
    source: typeof value.source === "string" ? value.source : "game",
    pixels,
    count: pixels.length,
  };
}

function sessionPayload() {
  return {
    sessionId: state.sessionId,
    resetToken: state.resetToken,
    seed: state.seed,
    tick: state.authoritativeTick,
    authoritativeTick: state.authoritativeTick,
    mode: state.mode,
    running: state.running,
    apiDirection: state.apiDirection,
    latestEventSeq: state.eventSeq,
    latestInputSeq: state.inputSeq,
    settings,
  };
}

function compactEvents() {
  const minTick = Math.max(0, state.authoritativeTick - 1200);
  if (state.events.length > 2500) {
    state.events = state.events.filter((event) => event.tick >= minTick);
    state.inputEvents = state.events.filter((event) => event.type === "input");
  }
}

function addSessionEvent(body, defaultType = "input") {
  const type = body.type || defaultType;
  if (!["start", "input", "pause"].includes(type)) {
    throw new Error('type must be "start", "input", or "pause"');
  }
  const tick = parseTick(body.tick, state.authoritativeTick + 1);
  const event = {
    seq: state.eventSeq + 1,
    type,
    tick,
    sessionId: state.sessionId,
    receivedAt: Date.now() / 1000,
  };
  if (type === "input") {
    event.direction = parseDirection(body.direction);
    event.source = body.source === "human" ? "human" : "api";
    state.apiDirection = event.direction;
    state.inputSeq = event.seq;
  }
  state.eventSeq = event.seq;
  state.events.push(event);
  state.events.sort((a, b) => (a.tick - b.tick) || (a.seq - b.seq));
  state.inputEvents = state.events.filter((item) => item.type === "input");
  if (type === "start") {
    state.running = true;
  } else if (type === "pause") {
    state.running = false;
  }
  state.updatedAt = Date.now() / 1000;
  compactEvents();
  return event;
}

function resetGameState(resetScore = true) {
  state.sessionId = `local-${Date.now().toString(36)}`;
  state.resetToken += 1;
  state.seed = Date.now() >>> 0;
  state.authoritativeTick = 0;
  state.eventSeq = 0;
  state.events = [];
  state.inputSeq = 0;
  state.inputEvents = [];
  state.apiDirection = 0;
  state.running = false;
  if (resetScore) {
    state.score = { left: 0, right: 0 };
  }
  state.ball = { x: settings.width / 2, y: settings.height / 2, vx: 0, vy: 0 };
  state.paddles = {
    leftY: (settings.height - settings.paddleHeight) / 2,
    rightY: (settings.height - settings.paddleHeight) / 2,
  };
  state.appliedInputSeq = 0;
  state.appliedEventSeq = 0;
  state.frameSeq = 0;
  state.renderedAt = 0;
  state.eventCamera = null;
  state.source = "backend";
  state.updatedAt = Date.now() / 1000;
}

function serveFile(response, filePath) {
  fs.readFile(filePath, (error, body) => {
    if (error) {
      response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
      response.end("Not found");
      return;
    }
    response.writeHead(200, {
      "Content-Type": contentTypes[path.extname(filePath)] || "application/octet-stream",
      "Content-Length": body.length,
      "Cache-Control": "no-store",
    });
    response.end(body);
  });
}

async function handlePost(request, response, route) {
  try {
    const body = await readJson(request);
    if (route === "/api/control-mode") {
      if (!["human", "api"].includes(body.mode)) {
        throw new Error('mode must be "human" or "api"');
      }
      state.mode = body.mode;
      state.updatedAt = Date.now() / 1000;
      sendJson(response, sessionPayload());
      return;
    }
    if (route === "/api/input") {
      const event = addSessionEvent({ ...body, type: "input" }, "input");
      sendJson(response, { ...sessionPayload(), event });
      return;
    }
    if (route === "/api/start") {
      const event = addSessionEvent({ ...body, type: "start" }, "start");
      sendJson(response, { ...sessionPayload(), event });
      return;
    }
    if (route === "/api/event") {
      const event = addSessionEvent(body);
      sendJson(response, { ...sessionPayload(), event });
      return;
    }
    if (route === "/api/state") {
      if (Number.isFinite(Number(body.tick))) {
        state.authoritativeTick = Math.max(state.authoritativeTick, Number.parseInt(body.tick, 10));
      }
      if (body.score) {
        state.score = {
          left: Number.parseInt(body.score.left ?? state.score.left, 10),
          right: Number.parseInt(body.score.right ?? state.score.right, 10),
        };
      }
      if ("running" in body) {
        state.running = Boolean(body.running);
      }
      if (body.ball) {
        state.ball = {
          x: Number(body.ball.x ?? state.ball.x),
          y: Number(body.ball.y ?? state.ball.y),
          vx: Number(body.ball.vx ?? state.ball.vx),
          vy: Number(body.ball.vy ?? state.ball.vy),
        };
      }
      if (body.paddles) {
        state.paddles = {
          leftY: Number(body.paddles.leftY ?? state.paddles.leftY),
          rightY: Number(body.paddles.rightY ?? state.paddles.rightY),
        };
      }
      if (Number.isFinite(Number(body.appliedInputSeq))) {
        state.appliedInputSeq = Math.max(state.appliedInputSeq, Number.parseInt(body.appliedInputSeq, 10));
      }
      if (Number.isFinite(Number(body.appliedEventSeq))) {
        state.appliedEventSeq = Math.max(state.appliedEventSeq, Number.parseInt(body.appliedEventSeq, 10));
      }
      if (Number.isFinite(Number(body.frameSeq))) {
        state.frameSeq = Math.max(state.frameSeq, Number.parseInt(body.frameSeq, 10));
      }
      if (Number.isFinite(Number(body.renderedAt))) {
        state.renderedAt = Number(body.renderedAt);
      }
      if ("eventCamera" in body) {
        state.eventCamera = normalizeEventCamera(body.eventCamera);
      }
      if (typeof body.source === "string") {
        state.source = body.source;
      }
      state.updatedAt = Date.now() / 1000;
      compactEvents();
      sendJson(response, copyState());
      return;
    }
    if (route === "/api/reset") {
      resetGameState(body.resetScore !== false);
      sendJson(response, copyState());
      return;
    }
    sendJson(response, { error: "Not found" }, 404);
  } catch (error) {
    sendJson(response, { error: error.message }, 400);
  }
}

function handleRequest(request, response) {
  const url = new URL(request.url, "http://127.0.0.1");
  const route = decodeURIComponent(url.pathname);

  if (request.method === "OPTIONS") {
    response.writeHead(204, {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    });
    response.end();
    return;
  }

  if (request.method === "POST") {
    handlePost(request, response, route);
    return;
  }

  if (route === "/" || route === "/index.html") {
    serveFile(response, path.join(webRoot, "index.html"));
    return;
  }
  if (route === "/visualizer" || route === "/visualizer.html") {
    serveFile(response, path.join(webRoot, "visualizer.html"));
    return;
  }
  if (route === "/api/config" || route === "/api/session") {
    sendJson(response, sessionPayload());
    return;
  }
  if (route === "/api/inputs") {
    const hasSinceSeq = url.searchParams.has("sinceSeq");
    const sinceSeq = Number.parseInt(url.searchParams.get("sinceSeq") || "0", 10);
    const sinceTick = Number.parseInt(url.searchParams.get("sinceTick") || "-1", 10);
    const events = state.inputEvents.filter((event) => {
      if (hasSinceSeq && Number.isFinite(sinceSeq)) {
        return event.seq > sinceSeq;
      }
      return event.tick > sinceTick;
    });
    sendJson(response, { ...sessionPayload(), events });
    return;
  }
  if (route === "/api/events") {
    const hasSinceSeq = url.searchParams.has("sinceSeq");
    const sinceSeq = Number.parseInt(url.searchParams.get("sinceSeq") || "0", 10);
    const sinceTick = Number.parseInt(url.searchParams.get("sinceTick") || "-1", 10);
    const events = state.events.filter((event) => {
      if (hasSinceSeq && Number.isFinite(sinceSeq)) {
        return event.seq > sinceSeq;
      }
      return event.tick > sinceTick;
    });
    sendJson(response, { ...sessionPayload(), events });
    return;
  }
  if (route === "/api/state") {
    sendJson(response, copyState());
    return;
  }
  if (route.startsWith("/web/")) {
    const filePath = path.resolve(webRoot, route.slice("/web/".length));
    if (filePath.startsWith(webRoot + path.sep)) {
      serveFile(response, filePath);
      return;
    }
  }
  sendJson(response, { error: "Not found" }, 404);
}

const host = process.env.HOST || "127.0.0.1";
const port = Number.parseInt(process.env.PORT || "8000", 10);

http.createServer(handleRequest).listen(port, host, () => {
  console.log(`Pong SNN server listening at http://${host}:${port}`);
});
