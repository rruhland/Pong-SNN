const canvas = document.getElementById("preview");
const ctx = canvas.getContext("2d");
const scoreEl = document.getElementById("score");
const resetButton = document.getElementById("reset");

const viewer = {
  latestState: null,
  lastRenderedFrameSeq: -1,
  resetToken: null,
  sessionId: null,
  seed: null,
  stateChannel: "BroadcastChannel" in window ? new BroadcastChannel("pong-snn-state") : null,
  pollMs: 16,
  backendPollMs: 16,
  source: "waiting",
};

function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(canvas.clientWidth * dpr);
  canvas.height = Math.floor(canvas.clientHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  render();
}

function normalizeState(rawState, source) {
  if (!rawState) return null;
  const settings = { ...PongCore.DEFAULT_SETTINGS, ...(rawState.settings || {}) };
  const paddles = rawState.paddles || {};
  const ball = rawState.ball || {};
  const score = rawState.score || {};
  const tick = Number(rawState.tick ?? rawState.authoritativeTick ?? 0);

  return {
    ...rawState,
    source,
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
    eventCamera: normalizeEventCamera(rawState.eventCamera, settings),
  };
}

function normalizeEventCamera(rawEventCamera, settings) {
  if (!rawEventCamera || !Array.isArray(rawEventCamera.pixels)) return null;
  const width = Math.max(1, Math.floor(Number(rawEventCamera.width || settings.width)));
  const height = Math.max(1, Math.floor(Number(rawEventCamera.height || settings.height)));
  const maxIndex = width * height;
  const pixels = rawEventCamera.pixels
    .map((pixel) => Number(pixel))
    .filter((pixel) => Number.isInteger(pixel) && pixel >= 0 && pixel < maxIndex);

  return {
    ...rawEventCamera,
    width,
    height,
    pixels,
    count: pixels.length,
  };
}

function drawEventOverlay(ctx, canvasWidth, canvasHeight, eventCamera) {
  if (!eventCamera || eventCamera.pixels.length === 0) return;
  const scaleX = canvasWidth / eventCamera.width;
  const scaleY = canvasHeight / eventCamera.height;
  const flushRun = (start, end, row) => {
    const x = start - row * eventCamera.width;
    ctx.fillRect(x * scaleX, row * scaleY, (end - start + 1) * scaleX, scaleY);
  };
  let runStart = -1;
  let runEnd = -1;
  let runRow = -1;

  ctx.fillStyle = "rgba(255, 0, 0, .25)";
  for (const pixel of eventCamera.pixels) {
    const row = Math.floor(pixel / eventCamera.width);
    if (runStart >= 0 && row === runRow && pixel === runEnd + 1) {
      runEnd = pixel;
      continue;
    }
    if (runStart >= 0) {
      flushRun(runStart, runEnd, runRow);
    }
    runStart = pixel;
    runEnd = pixel;
    runRow = row;
  }
  if (runStart >= 0) {
    flushRun(runStart, runEnd, runRow);
  }
}

function acceptState(rawState, source) {
  const state = normalizeState(rawState, source);
  if (!state) return;

  const isNewSession = Boolean(viewer.sessionId && state.sessionId && state.sessionId !== viewer.sessionId);
  const isResetState = state.resetToken !== undefined && state.resetToken !== viewer.resetToken;
  const sameSession = !viewer.sessionId || !state.sessionId || state.sessionId === viewer.sessionId || isResetState;

  if (!sameSession) return;

  if (
    viewer.latestState &&
    !isNewSession &&
    !isResetState &&
    state.frameSeq < viewer.latestState.frameSeq &&
    state.tick <= viewer.latestState.tick
  ) {
    return;
  }

  viewer.latestState = state;
  viewer.source = source;
  viewer.sessionId = state.sessionId || viewer.sessionId;
  viewer.resetToken = state.resetToken ?? viewer.resetToken;
  viewer.seed = state.seed ?? viewer.seed;
  window.__pongViewerState = state;
}

function render() {
  if (!viewer.latestState) {
    PongCore.drawState(ctx, canvas.clientWidth, canvas.clientHeight, {
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
    scoreEl.textContent = "0:0";
    return;
  }

  PongCore.drawState(ctx, canvas.clientWidth, canvas.clientHeight, viewer.latestState);
  drawEventOverlay(ctx, canvas.clientWidth, canvas.clientHeight, viewer.latestState.eventCamera);
  scoreEl.textContent = `${viewer.latestState.score.left}:${viewer.latestState.score.right}`;
  viewer.lastRenderedFrameSeq = viewer.latestState.frameSeq;
}

async function pollSession() {
  try {
    const response = await fetch("/api/session", { cache: "no-store" });
    if (response.ok) {
      const session = await response.json();
      viewer.pollMs = session.settings?.pollMs || viewer.pollMs;
      viewer.backendPollMs = session.settings?.statePushMs || viewer.backendPollMs;
      const needsReset =
        viewer.resetToken === null ||
        viewer.resetToken !== session.resetToken ||
        viewer.sessionId !== session.sessionId;

      if (needsReset) {
        viewer.resetToken = session.resetToken;
        viewer.sessionId = session.sessionId;
        viewer.seed = session.seed;
        viewer.latestState = null;
      }
    }
  } finally {
    window.setTimeout(pollSession, viewer.pollMs);
  }
}

async function pollBackendState() {
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (response.ok) {
      acceptState(await response.json(), "backend");
    }
  } finally {
    window.setTimeout(pollBackendState, viewer.backendPollMs);
  }
}

async function resetGame() {
  const response = await fetch("/api/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ resetScore: true }),
    cache: "no-store",
  });
  if (response.ok) {
    acceptState(await response.json(), "backend");
    render();
  }
}

function frame() {
  render();
  requestAnimationFrame(frame);
}

resetButton.addEventListener("click", resetGame);
if (viewer.stateChannel) {
  viewer.stateChannel.addEventListener("message", (event) => {
    acceptState(event.data, "game-channel");
  });
}
window.addEventListener("resize", resize);

window.__pongViewer = viewer;
resize();
pollSession();
pollBackendState();
requestAnimationFrame(frame);
