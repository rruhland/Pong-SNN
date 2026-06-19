const canvas = document.getElementById("preview");
const ctx = canvas.getContext("2d");
const scoreEl = document.getElementById("score");
const resetButton = document.getElementById("reset");
const startButton = document.getElementById("start");
const pauseButton = document.getElementById("pause");
const saveButton = document.getElementById("saveNetwork");
const loadButton = document.getElementById("loadNetwork");
const savesSelect = document.getElementById("savedNetworks");
const trailInput = document.getElementById("trailMs");
const trailValue = document.getElementById("trailValue");
const simSpeedInput = document.getElementById("simSpeed");
const simSpeedValue = document.getElementById("simSpeedValue");
const networkCanvas = document.getElementById("networkDesign");
const networkCtx = networkCanvas?.getContext("2d");
const outputBars = {
  up: document.getElementById("barUp"),
  down: document.getElementById("barDown"),
  stay: document.getElementById("barStay"),
};
const winnerEl = document.getElementById("winner");
const eventCountEl = document.getElementById("eventCount");
const spikeCountsEl = document.getElementById("spikeCounts");
const stdpCountsEl = document.getElementById("stdpCounts");
const rewardReadoutEl = document.getElementById("rewardReadout");
const eligibilityReadoutEl = document.getElementById("eligibilityReadout");
const hitMissReadoutEl = document.getElementById("hitMissReadout");
const learningReadoutEl = document.getElementById("learningReadout");
const deviceEl = document.getElementById("device");
const trainStateEl = document.getElementById("trainState");

const viewer = {
  latestState: null,
  lastRenderedFrameSeq: -1,
  resetToken: null,
  sessionId: null,
  seed: null,
  stateChannel: "BroadcastChannel" in window ? new BroadcastChannel("pong-snn-state") : null,
  pollMs: 16,
  backendPollMs: 8,
  backendPollInFlight: false,
  lastBackendPollAt: -Infinity,
  source: "waiting",
  eventTrail: [],
  trailMs: Number(trailInput?.value || 120),
  simulationSpeed: Number(simSpeedInput?.value || 1),
  speedPostTimer: null,
  lastTrailKey: null,
  snnStatus: null,
  snnPollMs: 120,
};

function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(canvas.clientWidth * dpr);
  canvas.height = Math.floor(canvas.clientHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (networkCanvas && networkCtx) {
    networkCanvas.width = Math.floor(networkCanvas.clientWidth * dpr);
    networkCanvas.height = Math.floor(networkCanvas.clientHeight * dpr);
    networkCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
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

function colorForEventAge(ageRatio) {
  const clamped = Math.max(0, Math.min(1, ageRatio));
  const green = clamped < 0.5 ? Math.round(330 * clamped) : Math.round(165 + 180 * (clamped - 0.5));
  const alpha = Math.max(0, 1 - clamped * 0.85);
  return `rgba(255, ${green}, 0, ${alpha.toFixed(3)})`;
}

function drawEventPixels(ctx, canvasWidth, canvasHeight, eventCamera, fillStyle) {
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

  ctx.fillStyle = fillStyle;
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

function pruneEventTrail(now = performance.now()) {
  viewer.eventTrail = viewer.eventTrail.filter((entry) => now - entry.receivedAt <= viewer.trailMs);
}

function rememberEventCamera(state) {
  const eventCamera = state.eventCamera;
  if (!eventCamera || eventCamera.pixels.length === 0) return;
  const key = `${state.resetToken ?? ""}:${eventCamera.frameSeq ?? state.frameSeq}:${eventCamera.tick ?? state.tick}`;
  if (key === viewer.lastTrailKey) return;

  viewer.lastTrailKey = key;
  viewer.eventTrail.push({
    key,
    receivedAt: performance.now(),
    eventCamera,
  });
  pruneEventTrail();
}

function drawEventTrail(ctx, canvasWidth, canvasHeight) {
  const now = performance.now();
  pruneEventTrail(now);
  for (const entry of viewer.eventTrail) {
    const ageRatio = viewer.trailMs <= 0 ? 1 : (now - entry.receivedAt) / viewer.trailMs;
    if (ageRatio > 1) continue;
    drawEventPixels(ctx, canvasWidth, canvasHeight, entry.eventCamera, colorForEventAge(ageRatio));
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
  if (isNewSession || isResetState) {
    viewer.eventTrail = [];
    viewer.lastTrailKey = null;
  }
  rememberEventCamera(state);
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
    renderSnn();
    return;
  }

  PongCore.drawState(ctx, canvas.clientWidth, canvas.clientHeight, viewer.latestState);
  drawEventTrail(ctx, canvas.clientWidth, canvas.clientHeight);
  scoreEl.textContent = `${viewer.latestState.score.left}:${viewer.latestState.score.right}`;
  viewer.lastRenderedFrameSeq = viewer.latestState.frameSeq;
  renderSnn();
}

function setBar(element, value) {
  if (!element) return;
  const percent = Math.max(0, Math.min(100, Number(value || 0) * 100));
  element.style.width = `${percent.toFixed(1)}%`;
}

function setSpeedControl(value) {
  const numeric = Math.max(0.25, Math.min(4, Number(value || 1)));
  viewer.simulationSpeed = numeric;
  if (simSpeedInput && Math.abs(Number(simSpeedInput.value) - numeric) > 0.001) {
    simSpeedInput.value = String(numeric);
  }
  if (simSpeedValue) {
    simSpeedValue.textContent = `${numeric.toFixed(2)}x`;
  }
}

function renderSnn() {
  const status = viewer.snnStatus;
  const activity = status?.activity;
  const bars = activity?.outputBars || [0, 0, 0];
  setBar(outputBars.up, bars[0]);
  setBar(outputBars.down, bars[1]);
  setBar(outputBars.stay, bars[2]);
  if (winnerEl) winnerEl.textContent = activity?.winner || "stay put";
  if (eventCountEl) eventCountEl.textContent = `events ${activity?.eventCount || 0}`;
  const spikes = activity?.spikes || {};
  if (spikeCountsEl) {
    spikeCountsEl.textContent = `spikes ${spikes.hidden1 || 0} / ${spikes.hidden2 || 0} / ${spikes.hidden3 || 0}`;
  }
  const stdp = activity?.stdp || {};
  if (stdpCountsEl) stdpCountsEl.textContent = `stdp +${stdp.potentiated || 0} / -${stdp.depressed || 0}`;
  const reward = activity?.reward || status?.reward || {};
  if (rewardReadoutEl) {
    rewardReadoutEl.textContent = `reward ${formatSigned(reward.value || 0, 3)}`;
  }
  const eligibility = activity?.eligibility || status?.eligibility || {};
  const eligibilityActive = Object.values(eligibility).reduce((total, group) => total + Number(group?.active || 0), 0);
  const maxEligibility = Object.values(eligibility).reduce(
    (maxValue, group) => Math.max(maxValue, Number(group?.maxAbs || 0)),
    0
  );
  if (eligibilityReadoutEl) {
    eligibilityReadoutEl.textContent = `eligibility ${eligibilityActive} traces / max ${maxEligibility.toFixed(3)}`;
  }
  if (hitMissReadoutEl) {
    hitMissReadoutEl.textContent = `hits ${reward.recentHits || 0} / misses ${reward.recentMisses || 0}`;
  }
  const learning = activity?.learning || status?.learning || {};
  if (learningReadoutEl) {
    learningReadoutEl.textContent = `learn step ${learning.step || 0} / updates ${learning.weightUpdates || 0}`;
  }
  const device = status?.architecture?.device;
  if (deviceEl) {
    const active = device?.active || "cpu";
    const cuda = device?.cudaAvailable ? `cuda ${device.cudaName || ""}`.trim() : "cuda unavailable";
    deviceEl.textContent = `backend: ${active} (${cuda})`;
  }
  if (trainStateEl) {
    trainStateEl.textContent = status?.training && !status?.paused ? "training" : "paused";
  }
  drawNetworkDesign();
}

function formatSigned(value, digits) {
  const numeric = Number(value || 0);
  const sign = numeric > 0 ? "+" : "";
  return `${sign}${numeric.toFixed(digits)}`;
}

function drawActivityGrid(ctx, activeIndices, grid, x, y, width, height, color) {
  if (!grid || !activeIndices?.length) return;
  const cellW = width / grid.width;
  const cellH = height / grid.height;
  ctx.fillStyle = color;
  for (const index of activeIndices) {
    const col = index % grid.width;
    const row = Math.floor(index / grid.width);
    ctx.fillRect(x + col * cellW, y + row * cellH, Math.max(1.5, cellW), Math.max(1.5, cellH));
  }
}

function drawNetworkDesign() {
  if (!networkCanvas || !networkCtx) return;
  const width = networkCanvas.clientWidth;
  const height = networkCanvas.clientHeight;
  const status = viewer.snnStatus;
  const architecture = status?.architecture;
  const activity = status?.activity || {};
  networkCtx.clearRect(0, 0, width, height);
  networkCtx.fillStyle = "#fffdf8";
  networkCtx.fillRect(0, 0, width, height);
  networkCtx.strokeStyle = "#111";
  networkCtx.lineWidth = 2;
  networkCtx.font = "12px Arial";
  networkCtx.fillStyle = "#111";
  networkCtx.fillText("sparse event-camera SNN", 10, 18);

  if (!architecture) {
    networkCtx.fillText("waiting for backend", 10, 42);
    return;
  }

  const layers = [
    { label: "input", width: architecture.input.width, height: architecture.input.height, active: activity.activeInputSample || [] },
    ...(architecture.layers || []).map((layer) => ({
      label: layer.name,
      width: layer.width,
      height: layer.height,
      active:
        layer.name === "hidden1"
          ? activity.activeHidden1
          : layer.name === "hidden2"
            ? activity.activeHidden2
            : layer.name === "hidden3"
              ? activity.activeHidden3
              : [],
      layer,
    })),
  ];

  const top = 34;
  const panelHeight = height - top - 16;
  const step = width / Math.max(1, layers.length);
  const boxes = [];
  for (let index = 0; index < layers.length; index += 1) {
    const layer = layers[index];
    const boxW = Math.min(92, step * 0.68);
    const boxH = Math.min(panelHeight * 0.72, Math.max(28, boxW * (layer.height / Math.max(1, layer.width))));
    const x = step * index + (step - boxW) / 2;
    const y = top + (panelHeight - boxH) / 2;
    boxes.push({ x, y, width: boxW, height: boxH, layer });
  }

  networkCtx.strokeStyle = "rgba(17,17,17,0.42)";
  networkCtx.lineWidth = 1;
  for (let index = 0; index < boxes.length - 1; index += 1) {
    const from = boxes[index];
    const to = boxes[index + 1];
    const lines = index === 0 ? 7 : 5;
    for (let line = 0; line < lines; line += 1) {
      const fromY = from.y + ((line + 1) / (lines + 1)) * from.height;
      const toY = to.y + ((line + 1) / (lines + 1)) * to.height;
      networkCtx.beginPath();
      networkCtx.moveTo(from.x + from.width, fromY);
      networkCtx.lineTo(to.x, toY);
      networkCtx.stroke();
    }
  }

  for (const box of boxes) {
    networkCtx.fillStyle = "#fff";
    networkCtx.fillRect(box.x, box.y, box.width, box.height);
    networkCtx.strokeStyle = "#111";
    networkCtx.lineWidth = 2;
    networkCtx.strokeRect(box.x, box.y, box.width, box.height);
    drawActivityGrid(networkCtx, box.layer.active, box.layer, box.x, box.y, box.width, box.height, "#e31b2f");
    networkCtx.fillStyle = "#111";
    networkCtx.font = "12px Arial";
    networkCtx.textAlign = "center";
    networkCtx.fillText(box.layer.label, box.x + box.width / 2, box.y + box.height + 14);
    if (box.layer.layer?.receptiveField) {
      networkCtx.font = "10px Arial";
      networkCtx.fillText(box.layer.layer.receptiveField, box.x + box.width / 2, box.y + box.height + 27);
    }
  }
  networkCtx.textAlign = "left";
}

async function pollSession() {
  try {
    const response = await fetch("/api/session", { cache: "no-store" });
    if (response.ok) {
      const session = await response.json();
      viewer.pollMs = session.settings?.pollMs || viewer.pollMs;
      viewer.backendPollMs = Math.min(viewer.backendPollMs, session.settings?.statePushMs || viewer.backendPollMs);
      if (session.settings?.simulationSpeed !== undefined) {
        setSpeedControl(session.settings.simulationSpeed);
      }
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

async function pollBackendStateOnce() {
  if (viewer.backendPollInFlight) return;
  viewer.backendPollInFlight = true;
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (response.ok) {
      acceptState(await response.json(), "backend");
    }
  } finally {
    viewer.backendPollInFlight = false;
  }
}

function maybePollBackendState(now = performance.now()) {
  if (now - viewer.lastBackendPollAt < viewer.backendPollMs) return;
  viewer.lastBackendPollAt = now;
  pollBackendStateOnce();
}

async function resetGame() {
  const response = await fetch("/api/snn/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ resetScore: true, resetWeights: true }),
    cache: "no-store",
  });
  if (response.ok) {
    acceptState(await response.json(), "backend");
    await pollSnnStatusOnce();
    render();
  }
}

async function postSnn(path, body = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  if (!response.ok) return null;
  return response.json();
}

async function postSimulationSpeed(speed) {
  const response = await fetch("/api/sim-speed", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ speed }),
    cache: "no-store",
  });
  if (!response.ok) return;
  const payload = await response.json();
  if (payload?.settings?.simulationSpeed !== undefined) {
    setSpeedControl(payload.settings.simulationSpeed);
  }
}

function scheduleSpeedPost() {
  const speed = Number(simSpeedInput?.value || 1);
  setSpeedControl(speed);
  if (viewer.speedPostTimer) {
    window.clearTimeout(viewer.speedPostTimer);
  }
  viewer.speedPostTimer = window.setTimeout(() => {
    viewer.speedPostTimer = null;
    postSimulationSpeed(viewer.simulationSpeed).catch(() => {
      // Session polling will restore the last accepted backend value.
    });
  }, 80);
}

function nextObservedTick() {
  const tick = Number(viewer.latestState?.tick ?? viewer.latestState?.authoritativeTick ?? 0);
  return Number.isFinite(tick) && tick >= 0 ? Math.floor(tick) + 1 : undefined;
}

async function startSnn() {
  const payload = await postSnn("/api/snn/start", { tick: nextObservedTick() });
  if (payload?.snn) viewer.snnStatus = payload.snn;
}

async function pauseSnn() {
  const payload = await postSnn("/api/snn/pause", { tick: nextObservedTick() });
  if (payload?.snn) viewer.snnStatus = payload.snn;
}

async function saveSnn() {
  const payload = await postSnn("/api/snn/save", { name: "pong-snn" });
  if (payload?.snn) viewer.snnStatus = payload.snn;
  if (payload?.saves) updateSaves(payload.saves);
}

async function loadSnn() {
  if (!savesSelect?.value) return;
  const payload = await postSnn("/api/snn/load", { name: savesSelect.value });
  if (payload?.snn) viewer.snnStatus = payload.snn;
  if (payload?.saves) updateSaves(payload.saves);
}

function updateSaves(saves) {
  if (!savesSelect) return;
  const current = savesSelect.value;
  savesSelect.textContent = "";
  const empty = document.createElement("option");
  empty.value = "";
  empty.textContent = saves?.length ? "select saved network" : "no saved networks";
  savesSelect.appendChild(empty);
  for (const save of saves || []) {
    const option = document.createElement("option");
    option.value = save.name;
    option.textContent = save.name;
    savesSelect.appendChild(option);
  }
  if (current && [...savesSelect.options].some((option) => option.value === current)) {
    savesSelect.value = current;
  }
}

async function pollSaves() {
  try {
    const response = await fetch("/api/snn/saves", { cache: "no-store" });
    if (response.ok) {
      const payload = await response.json();
      updateSaves(payload.saves || []);
    }
  } catch {
    // Saves are optional; status polling will keep the rest of the UI alive.
  }
}

async function pollSnnStatusOnce() {
  const response = await fetch("/api/snn/status", { cache: "no-store" });
  if (response.ok) {
    viewer.snnStatus = await response.json();
  }
}

async function pollSnnStatus() {
  try {
    await pollSnnStatusOnce();
  } catch {
    // The game mirror still works before the Python SNN backend is available.
  } finally {
    window.setTimeout(pollSnnStatus, viewer.snnPollMs);
  }
}

function frame() {
  maybePollBackendState();
  render();
  requestAnimationFrame(frame);
}

resetButton?.addEventListener("click", resetGame);
startButton?.addEventListener("click", startSnn);
pauseButton?.addEventListener("click", pauseSnn);
saveButton?.addEventListener("click", saveSnn);
loadButton?.addEventListener("click", loadSnn);
if (trailInput) {
  trailInput.addEventListener("input", () => {
    viewer.trailMs = Number(trailInput.value);
    if (trailValue) {
      trailValue.textContent = `${viewer.trailMs}ms`;
    }
    pruneEventTrail();
  });
}
if (simSpeedInput) {
  simSpeedInput.addEventListener("input", scheduleSpeedPost);
}
if (viewer.stateChannel) {
  viewer.stateChannel.addEventListener("message", (event) => {
    acceptState(event.data, "game-channel");
  });
}
window.addEventListener("resize", resize);

window.__pongViewer = viewer;
resize();
pollSession();
pollBackendStateOnce();
pollSnnStatus();
pollSaves();
requestAnimationFrame(frame);
