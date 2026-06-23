const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");

const FRAME_MS = 1000 / 60;
const MAX_STEPS_PER_FRAME = 5;

const client = {
  sim: null,
  eventCamera: null,
  sessionId: null,
  resetToken: null,
  seed: null,
  keys: new Set(),
  pendingEvents: [],
  lastEventSeq: 0,
  lastPublishedDirection: 0,
  stateChannel: "BroadcastChannel" in window ? new BroadcastChannel("pong-snn-state") : null,
  pollMs: 120,
  eventPollMs: 8,
  postMs: 16,
  simulationSpeed: 1,
  frameSeq: 0,
  frameHistory: [],
  pendingWorldFrames: new Map(),
  postingObservation: false,
  lastPostAt: -Infinity,
  lastFrameAt: null,
  accumulatorMs: 0,
};

function createLocalSimulation(seed = 1, settings = PongCore.DEFAULT_SETTINGS) {
  client.seed = seed;
  client.sim = PongCore.createSimulation({ seed, settings });
  if (!client.eventCamera) {
    client.eventCamera = EventCamera.create({
      width: settings.width,
      height: settings.height,
      source: "event-camera",
      onEventCamera: acceptEventCamera,
    });
  } else {
    client.eventCamera.width = settings.width;
    client.eventCamera.height = settings.height;
    client.eventCamera.reset();
  }
  client.pendingEvents = [];
  client.lastEventSeq = 0;
  client.lastPublishedDirection = 0;
  client.frameSeq = 0;
  client.frameHistory = [];
  client.pendingWorldFrames.clear();
}

function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(window.innerWidth * dpr);
  canvas.height = Math.floor(window.innerHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  render();
}

function currentKeyboardDirection() {
  const up = client.keys.has("ArrowUp") || client.keys.has("KeyW");
  const down = client.keys.has("ArrowDown") || client.keys.has("KeyS");
  if (up && !down) return -1;
  if (down && !up) return 1;
  return 0;
}

function normalizeSessionSettings(settings) {
  return { ...PongCore.DEFAULT_SETTINGS, ...(settings || {}) };
}

function mergeSettings(settings) {
  if (!client.sim) return;
  client.sim.settings = normalizeSessionSettings(settings);
  client.sim.left.speed = 320;
  client.sim.right.speed = 360;
  client.sim.left.x = 24;
  client.sim.right.x = client.sim.settings.width - 34;
  client.simulationSpeed = Number(settings?.simulationSpeed ?? client.simulationSpeed ?? 1);
}

function resetFromSession(session) {
  const settings = normalizeSessionSettings(session.settings);
  client.sessionId = session.sessionId;
  client.resetToken = session.resetToken;
  createLocalSimulation(Number(session.seed || 1), settings);
  client.pollMs = Number(session.settings?.pollMs || client.pollMs);
  client.eventPollMs = Number(session.settings?.eventPollMs || client.eventPollMs);
  client.postMs = Number(session.settings?.statePushMs || client.postMs);
  client.simulationSpeed = Number(session.settings?.simulationSpeed || 1);
}

function normalizeEvent(rawEvent) {
  if (!rawEvent || typeof rawEvent !== "object") return null;
  const tick = Number(rawEvent.tick ?? 0);
  const type = rawEvent.type || "input";
  if (!Number.isFinite(tick) || !["start", "pause", "input"].includes(type)) return null;
  return {
    ...rawEvent,
    type,
    tick: Math.max(0, Math.floor(tick)),
    seq: Number(rawEvent.seq || 0),
    direction: type === "input" ? Math.max(-1, Math.min(1, Number(rawEvent.direction || 0))) : undefined,
  };
}

function queueEvents(events) {
  for (const rawEvent of events || []) {
    const event = normalizeEvent(rawEvent);
    if (!event) continue;
    if (event.seq > 0) {
      client.lastEventSeq = Math.max(client.lastEventSeq, event.seq);
    }
    client.pendingEvents.push(event);
  }
  client.pendingEvents.sort((a, b) => (a.tick - b.tick) || ((a.seq || 0) - (b.seq || 0)));
}

async function pollSession() {
  try {
    const response = await fetch("/api/session", { cache: "no-store" });
    if (response.ok) {
      const session = await response.json();
      const isReset =
        !client.sim ||
        client.sessionId !== session.sessionId ||
        client.resetToken !== session.resetToken;
      if (isReset) {
        resetFromSession(session);
      } else {
        mergeSettings(session.settings);
        client.pollMs = Number(session.settings?.pollMs || client.pollMs);
        client.eventPollMs = Number(session.settings?.eventPollMs || client.eventPollMs);
        client.postMs = Number(session.settings?.statePushMs || client.postMs);
      }
    }
  } catch {
    // The world keeps running locally if the stream server is briefly unavailable.
  } finally {
    window.setTimeout(pollSession, client.pollMs);
  }
}

async function pollEvents() {
  try {
    const response = await fetch(`/api/events?sinceSeq=${client.lastEventSeq}`, { cache: "no-store" });
    if (response.ok) {
      const payload = await response.json();
      queueEvents(payload.events || []);
    }
  } catch {
    // Hold the last actuator direction until the event stream returns.
  } finally {
    window.setTimeout(pollEvents, client.eventPollMs);
  }
}

async function postJson(path, body = {}) {
  try {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      cache: "no-store",
    });
    return response.ok ? response.json() : null;
  } catch {
    return null;
  }
}

function localEvent(type, body = {}) {
  return {
    localId: `${type}:${performance.now()}:${Math.random()}`,
    type,
    tick: Math.floor(client.sim?.tick || 0),
    ...body,
  };
}

async function startBackendGame() {
  if (!client.sim) return;
  queueEvents([localEvent("start")]);
  await postJson("/api/start", { tick: Math.floor(client.sim.tick) });
}

async function publishInput(direction, source = "human") {
  if (!client.sim || direction === client.lastPublishedDirection) return;
  client.lastPublishedDirection = direction;
  queueEvents([localEvent("input", { direction, source })]);
  await postJson("/api/input", {
    tick: Math.floor(client.sim.tick),
    direction,
    source,
  });
}

function handleInputChange() {
  publishInput(currentKeyboardDirection(), "human");
}

function makeWorldFrame() {
  const snapshot = PongCore.snapshot(client.sim);
  const renderedAt = performance.now();
  return {
    ...snapshot,
    sessionId: client.sessionId,
    resetToken: client.resetToken,
    seed: client.seed,
    authoritativeTick: snapshot.tick,
    frameSeq: client.frameSeq,
    renderedAt,
    source: "game-renderer",
  };
}

function acceptEventCamera(eventCamera) {
  const worldFrame = client.pendingWorldFrames.get(eventCamera.frameSeq);
  if (!worldFrame) return;
  client.pendingWorldFrames.delete(eventCamera.frameSeq);
  const state = { ...worldFrame, eventCamera };
  broadcastState(state);
  publishObservation(state);

  const minFrameSeq = eventCamera.frameSeq - 12;
  for (const frameSeq of client.pendingWorldFrames.keys()) {
    if (frameSeq < minFrameSeq) {
      client.pendingWorldFrames.delete(frameSeq);
    }
  }
}

function observeWorldFrame(state) {
  client.pendingWorldFrames.set(state.frameSeq, state);
  client.eventCamera.observeCanvas(canvas, {
    width: state.settings.width,
    height: state.settings.height,
    tick: state.tick,
    frameSeq: state.frameSeq,
    resetToken: state.resetToken,
    renderedAt: state.renderedAt,
    source: "event-camera",
  });
}

function broadcastState(state) {
  client.frameHistory.push(state);
  if (client.frameHistory.length > 180) {
    client.frameHistory.shift();
  }
  if (client.stateChannel) {
    client.stateChannel.postMessage(state);
  }
  window.__pongLatestFrame = state;
  window.__pongFrameHistory = client.frameHistory;
}

function publishObservation(state, now = performance.now()) {
  if (client.postingObservation || now - client.lastPostAt < client.postMs) return;
  client.postingObservation = true;
  client.lastPostAt = now;
  postJson("/api/world/state", state).finally(() => {
    client.postingObservation = false;
  });
}

function stepWorld() {
  if (!client.sim) {
    createLocalSimulation(1, PongCore.DEFAULT_SETTINGS);
  }
  const due = [];
  const future = [];
  for (const event of client.pendingEvents) {
    if (event.tick <= client.sim.tick) {
      due.push(event);
    } else {
      future.push(event);
    }
  }
  client.pendingEvents = future;
  PongCore.step(client.sim, due, { simulationSpeed: client.simulationSpeed });
}

function render() {
  if (!client.sim) {
    createLocalSimulation(1, PongCore.DEFAULT_SETTINGS);
  }
  PongCore.draw(ctx, window.innerWidth, window.innerHeight, client.sim);
}

function frame(now) {
  if (client.lastFrameAt === null) {
    client.lastFrameAt = now;
  }
  client.accumulatorMs += Math.min(250, now - client.lastFrameAt);
  client.lastFrameAt = now;

  let steps = 0;
  while (client.accumulatorMs >= FRAME_MS && steps < MAX_STEPS_PER_FRAME) {
    stepWorld();
    client.accumulatorMs -= FRAME_MS;
    steps += 1;
  }

  if (steps > 0) {
    render();
    const state = makeWorldFrame();
    observeWorldFrame(state);
    client.frameSeq += 1;
  }

  requestAnimationFrame(frame);
}

window.addEventListener("resize", resize);
window.addEventListener("keydown", (event) => {
  client.keys.add(event.code);
  startBackendGame();
  handleInputChange();
  if (["ArrowUp", "ArrowDown", "Space"].includes(event.code)) {
    event.preventDefault();
  }
});
window.addEventListener("keyup", (event) => {
  client.keys.delete(event.code);
  handleInputChange();
  if (["ArrowUp", "ArrowDown", "Space"].includes(event.code)) {
    event.preventDefault();
  }
});

createLocalSimulation(1, PongCore.DEFAULT_SETTINGS);
resize();
window.__pongClient = client;
pollSession();
pollEvents();
requestAnimationFrame(frame);
