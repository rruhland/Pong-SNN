const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");

const client = {
  sim: PongCore.createSimulation(),
  eventCamera: PongCore.createEventCamera(),
  accumulator: 0,
  lastTime: performance.now(),
  mode: "api",
  resetToken: null,
  sessionId: null,
  seed: null,
  keys: new Set(),
  timelineEvents: [],
  pendingEvents: [],
  currentDirection: 0,
  lastPublishedDirection: 0,
  startRequested: false,
  eventChannel: "BroadcastChannel" in window ? new BroadcastChannel("pong-snn-events") : null,
  stateChannel: "BroadcastChannel" in window ? new BroadcastChannel("pong-snn-state") : null,
  pollMs: 16,
  statePushMs: 16,
  lastStatePush: 0,
  frameSeq: 0,
  frameHistory: [],
};

function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(window.innerWidth * dpr);
  canvas.height = Math.floor(window.innerHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function currentKeyboardDirection() {
  const up = client.keys.has("ArrowUp") || client.keys.has("KeyW");
  const down = client.keys.has("ArrowDown") || client.keys.has("KeyS");
  if (up && !down) return -1;
  if (down && !up) return 1;
  return 0;
}

function mergeEvents(events) {
  const bySeq = new Map(client.timelineEvents.filter((event) => event.seq > 0).map((event) => [event.seq, event]));
  const localEvents = client.timelineEvents.filter((event) => !event.seq || event.seq < 0);
  for (const event of events) {
    if (Number.isFinite(event.seq)) {
      bySeq.set(event.seq, event);
    }
  }
  client.timelineEvents = [...bySeq.values(), ...localEvents].sort((a, b) => (a.tick - b.tick) || (a.seq - b.seq));
  client.timelineEvents = client.timelineEvents.filter((event) => event.tick >= client.sim.tick - 180);
}

function broadcastEvent(event) {
  if (client.eventChannel) {
    client.eventChannel.postMessage(event);
  }
}

function frameState(now = performance.now()) {
  const state = {
    ...PongCore.snapshot(client.sim),
    sessionId: client.sessionId,
    resetToken: client.resetToken,
    seed: client.seed,
    mode: client.mode,
    frameSeq: client.frameSeq,
    renderedAt: now,
    source: "game",
  };
  state.eventCamera = PongCore.captureEventFrame(client.eventCamera, state, {
    resetToken: client.resetToken,
    frameSeq: client.frameSeq,
    renderedAt: now,
  });
  return state;
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

async function publishEvent(event, endpoint = "/api/event") {
  const localEvent = { ...event, seq: -performance.now(), localId: `${event.type}:${event.tick}:${performance.now()}` };
  client.pendingEvents.push(localEvent);
  mergeEvents([localEvent]);
  broadcastEvent(localEvent);

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(event),
      cache: "no-store",
    });
    if (response.ok) {
      const payload = await response.json();
      mergeEvents([payload.event]);
      broadcastEvent(payload.event);
      client.pendingEvents = client.pendingEvents.filter((pending) => pending !== localEvent);
      return payload.event;
    }
  } catch {
    // Local play should keep moving even if one event publish misses.
  }
  return localEvent;
}

async function publishInput(direction, source = "human") {
  const tick = client.sim.tick + 1;
  client.lastPublishedDirection = direction;
  return publishEvent({ type: "input", tick, direction, source }, "/api/input");
}

async function pushState(state = frameState()) {
  try {
    await fetch("/api/state", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state),
      cache: "no-store",
    });
  } catch {
    // Rendering should not pause if a checkpoint update is dropped.
  }
}

async function pollSession() {
  try {
    const response = await fetch("/api/session", { cache: "no-store" });
    if (!response.ok) return;
    const session = await response.json();
    client.mode = session.mode === "api" ? "api" : "human";
    client.pollMs = session.settings?.pollMs || client.pollMs;
    client.statePushMs = session.settings?.statePushMs || client.statePushMs;

    const needsReset =
      client.resetToken === null ||
      client.resetToken !== session.resetToken ||
      client.sessionId !== session.sessionId;

    if (needsReset) {
      client.resetToken = session.resetToken;
      client.sessionId = session.sessionId;
      client.seed = session.seed;
      client.timelineEvents = [];
      client.pendingEvents = [];
      client.currentDirection = 0;
      client.lastPublishedDirection = 0;
      client.startRequested = false;
      client.eventCamera = PongCore.createEventCamera();
      client.frameSeq = 0;
      client.accumulator = 0;
      PongCore.resetSimulation(client.sim, session.seed);
      const state = frameState();
      broadcastState(state);
      await pushState(state);
    }
  } catch {
    // The game remains playable if the backend misses a session poll.
  } finally {
    window.setTimeout(pollSession, client.pollMs);
  }
}

async function pollInputs() {
  try {
    const sinceSeq = Math.max(0, client.sim.appliedEventSeq);
    const response = await fetch(`/api/events?sinceSeq=${sinceSeq}&sinceTick=${Math.max(0, client.sim.tick - 180)}`, {
      cache: "no-store",
    });
    if (response.ok) {
      const payload = await response.json();
      mergeEvents(payload.events || []);
    }
  } catch {
    // A missed input poll will be picked up on the next request.
  } finally {
    window.setTimeout(pollInputs, client.pollMs);
  }
}

async function start() {
  if (!client.sim.running && !client.startRequested) {
    client.startRequested = true;
    await publishEvent({ type: "start", tick: client.sim.tick + 1 }, "/api/start");
  }
}

function handleInputChange() {
  if (client.mode !== "human") return;
  const direction = currentKeyboardDirection();
  if (direction !== client.lastPublishedDirection) {
    publishInput(direction, "human");
  }
}

function fixedUpdate(now) {
  const dt = Math.min((now - client.lastTime) / 1000, 0.1);
  client.lastTime = now;
  client.accumulator += dt;

  while (client.accumulator >= client.sim.settings.fixedDt) {
    PongCore.step(client.sim, client.timelineEvents);
    client.accumulator -= client.sim.settings.fixedDt;
  }

  PongCore.draw(ctx, window.innerWidth, window.innerHeight, client.sim);
  client.frameSeq += 1;
  const state = frameState(now);
  broadcastState(state);

  if (now - client.lastStatePush >= client.statePushMs) {
    client.lastStatePush = now;
    pushState(state);
  }

  requestAnimationFrame(fixedUpdate);
}

window.addEventListener("resize", resize);
if (client.eventChannel) {
  client.eventChannel.addEventListener("message", (event) => {
    mergeEvents([event.data]);
  });
}
window.addEventListener("keydown", (event) => {
  client.keys.add(event.code);
  start();
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

resize();
PongCore.draw(ctx, window.innerWidth, window.innerHeight, client.sim);
window.__pongClient = client;
broadcastState(frameState());
pollSession();
pollInputs();
requestAnimationFrame(fixedUpdate);
