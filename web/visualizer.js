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
const scoreRewardReadoutEl = document.getElementById("scoreRewardReadout");
const learningReadoutEl = document.getElementById("learningReadout");
const deviceEl = document.getElementById("device");
const trainStateEl = document.getElementById("trainState");
const worldStreamEl = document.getElementById("worldStream");
const eventStreamEl = document.getElementById("eventStream");
const snnStreamEl = document.getElementById("snnStream");
const actionStreamEl = document.getElementById("actionStream");

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
  streams: null,
  latestAction: null,
  worldRunner: null,
  hiddenCloud: null,
  predictionView: null,
};

function resize() {
  const dpr = window.devicePixelRatio || 1;
  if (viewer.worldRunner) {
    viewer.worldRunner.resize(canvas.clientWidth, canvas.clientHeight);
  } else {
    canvas.width = Math.floor(canvas.clientWidth * dpr);
    canvas.height = Math.floor(canvas.clientHeight * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
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
    viewer.hiddenCloud = null;
    viewer.predictionView = null;
  }
  rememberEventCamera(state);
  window.__pongViewerState = state;
}

function render() {
  if (!viewer.latestState) {
    if (!viewer.worldRunner) {
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
    }
    scoreEl.textContent = "0:0";
    renderSnn();
    return;
  }

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
  if (scoreRewardReadoutEl) {
    scoreRewardReadoutEl.textContent =
      `scores ${reward.recentRightScores || 0} / opp ${reward.recentOpponentScores || 0}`;
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
  renderStreams();
  drawNetworkDesign();
}

function renderStreams() {
  const state = viewer.latestState;
  const eventCamera = state?.eventCamera;
  const streams = viewer.streams || {};
  const snnRuntime = viewer.snnStatus?.runtime || streams.snn || {};
  const action = viewer.latestAction;
  if (worldStreamEl) {
    const source = state?.source || streams.world?.source || "waiting";
    const frameSeq = state?.frameSeq ?? streams.world?.frameSeq ?? 0;
    worldStreamEl.textContent = `world: ${source} frame ${frameSeq}`;
  }
  if (eventStreamEl) {
    const count = eventCamera?.count ?? streams.eventCamera?.count ?? 0;
    const frameSeq = eventCamera?.frameSeq ?? streams.eventCamera?.frameSeq ?? 0;
    eventStreamEl.textContent = `events: ${count} changes frame ${frameSeq}`;
  }
  if (snnStreamEl) {
    const samples = snnRuntime.samples ?? 0;
    const lastFrameSeq = snnRuntime.lastFrameSeq ?? 0;
    snnStreamEl.textContent = `snn: ${samples} samples through frame ${lastFrameSeq}`;
  }
  if (actionStreamEl) {
    const direction = action?.direction ?? viewer.snnStatus?.lastCommand ?? 0;
    const label = direction < 0 ? "up" : direction > 0 ? "down" : "hold";
    const seq = action?.seq ?? 0;
    actionStreamEl.textContent = `action: ${label} seq ${seq}`;
  }
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

function stableNoise(index) {
  const value = Math.sin(index * 12.9898 + 78.233) * 43758.5453;
  return value - Math.floor(value);
}

function ensureFloatGrid(existing, width, height) {
  const size = width * height;
  if (existing && existing.width === width && existing.height === height && existing.activity?.length === size) {
    return existing;
  }
  return {
    width,
    height,
    activity: new Float32Array(size),
    memory: new Float32Array(size),
    lastAt: performance.now(),
  };
}

function decayGrid(grid, now, fastHalfLifeMs, memoryHalfLifeMs) {
  const elapsed = Math.max(0, Math.min(500, now - (grid.lastAt || now)));
  const fastDecay = Math.pow(0.5, elapsed / fastHalfLifeMs);
  const memoryDecay = Math.pow(0.5, elapsed / memoryHalfLifeMs);
  for (let index = 0; index < grid.activity.length; index += 1) {
    grid.activity[index] *= fastDecay;
    grid.memory[index] *= memoryDecay;
  }
  grid.lastAt = now;
}

function addGridValue(values, width, height, x, y, amount, radius = 1.2) {
  const minX = Math.max(0, Math.floor(x - radius * 2));
  const maxX = Math.min(width - 1, Math.ceil(x + radius * 2));
  const minY = Math.max(0, Math.floor(y - radius * 2));
  const maxY = Math.min(height - 1, Math.ceil(y + radius * 2));
  const denom = Math.max(0.001, radius * radius);
  for (let row = minY; row <= maxY; row += 1) {
    for (let col = minX; col <= maxX; col += 1) {
      const distanceSq = (col - x) * (col - x) + (row - y) * (row - y);
      const weight = Math.exp(-distanceSq / denom);
      const index = row * width + col;
      values[index] = Math.min(1, values[index] + amount * weight);
    }
  }
}

function activeInputIndices(architecture, activity) {
  const input = architecture?.input;
  if (!input?.width || !input?.height) return activity?.activeInputSample || [];
  const eventCamera = viewer.latestState?.eventCamera;
  if (eventCamera?.pixels?.length) {
    const cells = new Set();
    for (const pixel of eventCamera.pixels) {
      const sourceX = pixel % eventCamera.width;
      const sourceY = Math.floor(pixel / eventCamera.width);
      const cellX = Math.max(0, Math.min(input.width - 1, Math.floor((sourceX / eventCamera.width) * input.width)));
      const cellY = Math.max(0, Math.min(input.height - 1, Math.floor((sourceY / eventCamera.height) * input.height)));
      cells.add(cellY * input.width + cellX);
      if (cells.size >= 900) break;
    }
    return [...cells];
  }
  return activity?.activeInputSample || [];
}

function predictionIndices(activity, status) {
  const current = activity?.activeHidden3;
  if (Array.isArray(current) && current.length) return current;
  const sample = activity?.prediction?.sample || status?.prediction?.sample;
  return Array.isArray(sample) ? sample : [];
}

function updateEstimatedHiddenCloud(architecture, activity, inputIndices, predictionSample) {
  const hiddenLayer = (architecture?.layers || []).find((layer) => layer.name === "hidden1");
  const width = Math.max(8, Number(hiddenLayer?.width || 24));
  const height = Math.max(6, Number(hiddenLayer?.height || 15));
  const input = architecture?.input || { width: 64, height: 36 };
  const now = performance.now();
  viewer.hiddenCloud = ensureFloatGrid(viewer.hiddenCloud, width, height);
  decayGrid(viewer.hiddenCloud, now, 220, 9500);

  const learning = activity?.learning || viewer.snnStatus?.learning || {};
  const learnedGain = Math.min(1, Math.log10(10 + Number(learning.step || 0)) / 5);
  for (const index of inputIndices.slice(0, 700)) {
    const inputX = index % input.width;
    const inputY = Math.floor(index / input.width);
    const hiddenX = ((inputX + 0.5) / input.width) * width - 0.5;
    const hiddenY = ((inputY + 0.5) / input.height) * height - 0.5;
    addGridValue(viewer.hiddenCloud.activity, width, height, hiddenX, hiddenY, 0.36, 1.25);
    addGridValue(viewer.hiddenCloud.memory, width, height, hiddenX, hiddenY, 0.006 + learnedGain * 0.01, 1.75);
  }

  for (const index of predictionSample.slice(0, 220)) {
    const inputX = index % input.width;
    const inputY = Math.floor(index / input.width);
    const hiddenX = ((inputX + 0.5) / input.width) * width - 0.5;
    const hiddenY = ((inputY + 0.5) / input.height) * height - 0.5;
    addGridValue(viewer.hiddenCloud.memory, width, height, hiddenX, hiddenY, 0.0035 + learnedGain * 0.003, 2.3);
  }

  return viewer.hiddenCloud;
}

function updatePredictionView(architecture, predictionSample) {
  const input = architecture?.input || { width: 64, height: 36 };
  const width = Math.max(1, Number(input.width || 64));
  const height = Math.max(1, Number(input.height || 36));
  const now = performance.now();
  viewer.predictionView = ensureFloatGrid(viewer.predictionView, width, height);
  decayGrid(viewer.predictionView, now, 260, 14000);
  for (const index of predictionSample.slice(0, 900)) {
    if (index < 0 || index >= width * height) continue;
    viewer.predictionView.activity[index] = Math.min(1, viewer.predictionView.activity[index] + 0.72);
    viewer.predictionView.memory[index] = Math.min(1, viewer.predictionView.memory[index] + 0.018);
  }
  return viewer.predictionView;
}

function drawHeatGrid(ctx, gridState, x, y, width, height, palette) {
  if (!gridState) return;
  const cellW = width / gridState.width;
  const cellH = height / gridState.height;
  for (let index = 0; index < gridState.activity.length; index += 1) {
    const value = Math.max(gridState.activity[index], gridState.memory[index] * 0.7);
    if (value < 0.015) continue;
    const col = index % gridState.width;
    const row = Math.floor(index / gridState.width);
    const alpha = Math.min(0.88, 0.1 + value * 0.78);
    ctx.fillStyle = palette(alpha, value, index);
    ctx.fillRect(x + col * cellW, y + row * cellH, Math.max(1.2, cellW), Math.max(1.2, cellH));
  }
}

function drawEstimatedCloud(ctx, cloud, x, y, width, height) {
  if (!cloud) return;
  const cellW = width / cloud.width;
  const cellH = height / cloud.height;
  for (let index = 0; index < cloud.activity.length; index += 1) {
    const value = Math.max(cloud.activity[index], cloud.memory[index] * 0.78);
    if (value < 0.018) continue;
    const col = index % cloud.width;
    const row = Math.floor(index / cloud.width);
    const jitterX = (stableNoise(index) - 0.5) * cellW * 0.45;
    const jitterY = (stableNoise(index + 99) - 0.5) * cellH * 0.45;
    const size = Math.max(2, Math.min(cellW, cellH) * (0.42 + value * 0.55));
    const alpha = Math.min(0.9, 0.12 + value * 0.82);
    ctx.fillStyle = `rgba(31, 111, 235, ${alpha.toFixed(3)})`;
    ctx.fillRect(
      x + (col + 0.5) * cellW + jitterX - size / 2,
      y + (row + 0.5) * cellH + jitterY - size / 2,
      size,
      size
    );
  }
}

function drawNetworkPanelFrame(ctx, box) {
  ctx.fillStyle = "#fff";
  ctx.fillRect(box.x, box.y, box.width, box.height);
  ctx.strokeStyle = "#111";
  ctx.lineWidth = 2;
  ctx.strokeRect(box.x, box.y, box.width, box.height);
}

function drawNetworkLabel(ctx, text, box, offset = 14) {
  ctx.fillStyle = "#111";
  ctx.font = "12px Arial";
  ctx.textAlign = "center";
  ctx.fillText(text, box.x + box.width / 2, box.y + box.height + offset);
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

  const input = architecture.input || { width: 64, height: 36 };
  const inputIndices = activeInputIndices(architecture, activity);
  const predictionSample = predictionIndices(activity, status);
  const hiddenCloud = updateEstimatedHiddenCloud(architecture, activity, inputIndices, predictionSample);
  const predictionView = updatePredictionView(architecture, predictionSample);

  const top = 36;
  const labelSpace = 34;
  const panelHeight = height - top - labelSpace - 8;
  const gap = Math.max(18, width * 0.035);
  const inputAspect = input.height / Math.max(1, input.width);
  const maxGridW = Math.max(80, (width - gap * 4) * 0.26);
  const gridW = Math.min(180, maxGridW);
  const gridH = Math.min(panelHeight * 0.74, Math.max(42, gridW * inputAspect));
  const cloudW = Math.min(210, Math.max(112, (width - gap * 4) * 0.28));
  const cloudH = Math.min(panelHeight * 0.82, Math.max(70, cloudW * 0.7));
  const totalW = gridW + cloudW + gridW + gap * 2;
  const startX = Math.max(10, (width - totalW) / 2);
  const centerY = top + panelHeight / 2;
  const boxes = {
    input: { x: startX, y: centerY - gridH / 2, width: gridW, height: gridH },
    cloud: { x: startX + gridW + gap, y: centerY - cloudH / 2, width: cloudW, height: cloudH },
    prediction: { x: startX + gridW + gap + cloudW + gap, y: centerY - gridH / 2, width: gridW, height: gridH },
  };

  networkCtx.strokeStyle = "rgba(17,17,17,0.42)";
  networkCtx.lineWidth = 1;
  for (const pair of [[boxes.input, boxes.cloud], [boxes.cloud, boxes.prediction]]) {
    const [from, to] = pair;
    const lines = 9;
    for (let line = 0; line < lines; line += 1) {
      const fromY = from.y + ((line + 1) / (lines + 1)) * from.height;
      const toY = to.y + ((line + 1) / (lines + 1)) * to.height;
      networkCtx.beginPath();
      networkCtx.moveTo(from.x + from.width, fromY);
      networkCtx.lineTo(to.x, toY);
      networkCtx.stroke();
    }
  }

  drawNetworkPanelFrame(networkCtx, boxes.input);
  drawActivityGrid(networkCtx, inputIndices, input, boxes.input.x, boxes.input.y, boxes.input.width, boxes.input.height, "#e31b2f");
  drawNetworkLabel(networkCtx, "input events", boxes.input);

  drawNetworkPanelFrame(networkCtx, boxes.cloud);
  networkCtx.fillStyle = "rgba(31, 111, 235, 0.055)";
  networkCtx.fillRect(boxes.cloud.x, boxes.cloud.y, boxes.cloud.width, boxes.cloud.height);
  drawEstimatedCloud(networkCtx, hiddenCloud, boxes.cloud.x + 6, boxes.cloud.y + 6, boxes.cloud.width - 12, boxes.cloud.height - 12);
  drawNetworkLabel(networkCtx, "estimated hidden cloud", boxes.cloud);
  networkCtx.font = "10px Arial";
  networkCtx.fillStyle = "#111";
  networkCtx.textAlign = "center";
  networkCtx.fillText("input-shaped trace memory", boxes.cloud.x + boxes.cloud.width / 2, boxes.cloud.y + boxes.cloud.height + 27);

  drawNetworkPanelFrame(networkCtx, boxes.prediction);
  drawHeatGrid(
    networkCtx,
    predictionView,
    boxes.prediction.x,
    boxes.prediction.y,
    boxes.prediction.width,
    boxes.prediction.height,
    (alpha, value) => {
      const green = Math.round(95 + value * 110);
      return `rgba(22, ${green}, 118, ${alpha.toFixed(3)})`;
    }
  );
  drawNetworkLabel(networkCtx, "predicted next events", boxes.prediction);
  if (status?.prediction) {
    const stats = status.prediction;
    networkCtx.font = "10px Arial";
    networkCtx.fillStyle = "#111";
    networkCtx.textAlign = "center";
    networkCtx.fillText(
      `hits ${stats.hits || 0} / misses ${stats.misses || 0} / mae ${Number(stats.meanAbsError || 0).toFixed(3)}`,
      boxes.prediction.x + boxes.prediction.width / 2,
      boxes.prediction.y + boxes.prediction.height + 27
    );
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
      viewer.streams = session.streams || viewer.streams;
      viewer.latestAction = session.latestAction || viewer.latestAction;
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
  if (viewer.worldRunner) {
    await viewer.worldRunner.pause();
  }
  const response = await fetch("/api/snn/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ resetScore: true, resetWeights: true }),
    cache: "no-store",
  });
  if (response.ok) {
    acceptState(await response.json(), "backend");
    if (viewer.worldRunner) {
      const session = await fetch("/api/session", { cache: "no-store" }).then((item) => item.json());
      viewer.worldRunner.resetFromSession(session);
    }
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
  if (viewer.worldRunner) {
    await viewer.worldRunner.start();
  }
  const payload = await postSnn("/api/snn/start", { tick: nextObservedTick() });
  if (payload?.snn) viewer.snnStatus = payload.snn;
  renderSnn();
}

async function pauseSnn() {
  if (viewer.worldRunner) {
    await viewer.worldRunner.pause();
  }
  const payload = await postSnn("/api/snn/pause", { tick: nextObservedTick() });
  if (payload?.snn) viewer.snnStatus = payload.snn;
  renderSnn();
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

if (window.PongWorldRunner && window.EventCamera) {
  viewer.worldRunner = PongWorldRunner.create({
    canvas,
    onFrame: (state) => acceptState(state, "world-runner"),
  });
  viewer.worldRunner.begin();
}

window.__pongViewer = viewer;
resize();
pollSession();
pollBackendStateOnce();
pollSnnStatus();
pollSaves();
requestAnimationFrame(frame);
