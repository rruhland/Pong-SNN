const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");

const client = {
  latestState: null,
  sessionId: null,
  resetToken: null,
  keys: new Set(),
  lastPublishedDirection: 0,
  stateChannel: "BroadcastChannel" in window ? new BroadcastChannel("pong-snn-state") : null,
  pollMs: 16,
  frameHistory: [],
};

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

function normalizeState(rawState) {
  if (!rawState) return null;
  const settings = { ...PongCore.DEFAULT_SETTINGS, ...(rawState.settings || {}) };
  const paddles = rawState.paddles || {};
  const ball = rawState.ball || {};
  const score = rawState.score || {};
  const tick = Number(rawState.tick ?? rawState.authoritativeTick ?? 0);
  return {
    ...rawState,
    tick,
    authoritativeTick: Number(rawState.authoritativeTick ?? tick),
    frameSeq: Number(rawState.frameSeq ?? tick),
    settings,
    running: Boolean(rawState.running),
    score: {
      left: Number(score.left ?? 0),
      right: Number(score.right ?? 0),
    },
    ball: {
      x: Number(ball.x ?? settings.width / 2),
      y: Number(ball.y ?? settings.height / 2),
      vx: Number(ball.vx ?? 0),
      vy: Number(ball.vy ?? 0),
    },
    paddles: {
      leftY: Number(paddles.leftY ?? (settings.height - settings.paddleHeight) / 2),
      rightY: Number(paddles.rightY ?? (settings.height - settings.paddleHeight) / 2),
    },
    source: rawState.source || "backend",
  };
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

function acceptState(rawState) {
  const state = normalizeState(rawState);
  if (!state) return;
  const isReset = state.resetToken !== undefined && state.resetToken !== client.resetToken;
  const isNewSession = client.sessionId && state.sessionId && state.sessionId !== client.sessionId;
  if (
    client.latestState &&
    !isReset &&
    !isNewSession &&
    state.frameSeq < client.latestState.frameSeq &&
    state.tick <= client.latestState.tick
  ) {
    return;
  }
  client.latestState = state;
  client.sessionId = state.sessionId || client.sessionId;
  client.resetToken = state.resetToken ?? client.resetToken;
  broadcastState(state);
}

async function pollState() {
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (response.ok) {
      const state = await response.json();
      client.pollMs = state.settings?.pollMs || client.pollMs;
      acceptState(state);
    }
  } catch {
    // The canvas remains on the last good backend frame.
  } finally {
    window.setTimeout(pollState, client.pollMs);
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

async function startBackendGame() {
  const tick = Number(client.latestState?.tick ?? 0);
  await postJson("/api/start", { tick: Number.isFinite(tick) ? Math.floor(tick) + 1 : undefined });
}

async function publishInput(direction, source = "human") {
  if (direction === client.lastPublishedDirection) return;
  client.lastPublishedDirection = direction;
  const tick = Number(client.latestState?.tick ?? 0);
  await postJson("/api/input", {
    tick: Number.isFinite(tick) ? Math.floor(tick) + 1 : undefined,
    direction,
    source,
  });
}

function handleInputChange() {
  publishInput(currentKeyboardDirection(), "human");
}

function render() {
  const state =
    client.latestState ||
    normalizeState({
      settings: PongCore.DEFAULT_SETTINGS,
      score: { left: 0, right: 0 },
      ball: {
        x: PongCore.DEFAULT_SETTINGS.width / 2,
        y: PongCore.DEFAULT_SETTINGS.height / 2,
        vx: 0,
        vy: 0,
      },
      paddles: {
        leftY: (PongCore.DEFAULT_SETTINGS.height - PongCore.DEFAULT_SETTINGS.paddleHeight) / 2,
        rightY: (PongCore.DEFAULT_SETTINGS.height - PongCore.DEFAULT_SETTINGS.paddleHeight) / 2,
      },
    });
  PongCore.drawState(ctx, window.innerWidth, window.innerHeight, state);
}

function frame() {
  render();
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

resize();
window.__pongClient = client;
pollState();
requestAnimationFrame(frame);
