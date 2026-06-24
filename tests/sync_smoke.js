const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const root = path.resolve(__dirname, "..");
const coreSource = fs.readFileSync(path.join(root, "web", "pong-core.js"), "utf8");
const eventCameraSource = fs.readFileSync(path.join(root, "web", "event-camera.js"), "utf8");
const worldRunnerSource = fs.readFileSync(path.join(root, "web", "world-runner.js"), "utf8");
const gameSource = fs.readFileSync(path.join(root, "web", "game.js"), "utf8");
const visualizerSource = fs.readFileSync(path.join(root, "web", "visualizer.js"), "utf8");

const channelSubscribers = new Map();

class FakeBroadcastChannel {
  constructor(name) {
    this.name = name;
    this.listeners = new Set();
    if (!channelSubscribers.has(name)) {
      channelSubscribers.set(name, new Set());
    }
    channelSubscribers.get(name).add(this);
  }

  addEventListener(type, listener) {
    if (type === "message") {
      this.listeners.add(listener);
    }
  }

  postMessage(data) {
    for (const channel of channelSubscribers.get(this.name) || []) {
      if (channel === this) continue;
      for (const listener of channel.listeners) {
        listener({ data });
      }
    }
  }
}

function fakeContext() {
  const listeners = new Map();
  const drawCalls = [];
  const postedObservations = [];
  const pixels = new Uint8ClampedArray(800 * 450 * 4);
  let fillStyle = "";
  const fillColor = () => {
    if (fillStyle === "#000") return [0, 0, 0, 255];
    if (fillStyle === "#2d2d2d") return [45, 45, 45, 255];
    if (fillStyle === "#f5f5f5") return [245, 245, 245, 255];
    if (fillStyle === "#fffdf8" || fillStyle === "#fff" || fillStyle === "white") return [255, 255, 255, 255];
    return [0, 0, 0, 255];
  };
  const fillPixelRect = (x, y, width, height) => {
    const [r, g, b, a] = fillColor();
    const left = Math.max(0, Math.min(800, Math.floor(x)));
    const right = Math.max(0, Math.min(800, Math.ceil(x + width)));
    const top = Math.max(0, Math.min(450, Math.floor(y)));
    const bottom = Math.max(0, Math.min(450, Math.ceil(y + height)));
    for (let row = top; row < bottom; row += 1) {
      for (let col = left; col < right; col += 1) {
        const offset = (row * 800 + col) * 4;
        pixels[offset] = r;
        pixels[offset + 1] = g;
        pixels[offset + 2] = b;
        pixels[offset + 3] = a;
      }
    }
  };
  const canvas = {
    clientWidth: 800,
    clientHeight: 450,
    width: 800,
    height: 450,
    getContext() {
      return {
        clearRect() {},
        fillRect(...args) {
          drawCalls.push({ fillStyle, args });
          fillPixelRect(...args);
        },
        getImageData() {
          return { data: new Uint8ClampedArray(pixels) };
        },
        strokeRect() {},
        beginPath() {},
        moveTo() {},
        lineTo() {},
        stroke() {},
        fillText() {},
        setTransform() {},
        set fillStyle(value) {
          fillStyle = value;
        },
        set strokeStyle(value) {},
        set lineWidth(value) {},
        set font(value) {},
        set textAlign(value) {},
      };
    },
  };
  const elements = {
    game: canvas,
    preview: canvas,
    networkDesign: canvas,
    score: { textContent: "" },
    winner: { textContent: "" },
    eventCount: { textContent: "" },
    spikeCounts: { textContent: "" },
    stdpCounts: { textContent: "" },
    rewardReadout: { textContent: "" },
    eligibilityReadout: { textContent: "" },
    scoreRewardReadout: { textContent: "" },
    learningReadout: { textContent: "" },
    device: { textContent: "" },
    trainState: { textContent: "" },
    worldStream: { textContent: "" },
    eventStream: { textContent: "" },
    snnStream: { textContent: "" },
    actionStream: { textContent: "" },
    reset: { addEventListener() {} },
    start: { addEventListener() {} },
    pause: { addEventListener() {} },
    saveNetwork: { addEventListener() {} },
    loadNetwork: { addEventListener() {} },
    savedNetworks: { value: "", textContent: "", appendChild() {}, options: [] },
    trailMs: { value: "120", addEventListener() {} },
    trailValue: { textContent: "" },
    simSpeed: { value: "1", addEventListener() {} },
    simSpeedValue: { textContent: "" },
    barUp: { style: {} },
    barDown: { style: {} },
    barStay: { style: {} },
  };

  const session = {
    sessionId: "stream-smoke",
    resetToken: 3,
    seed: 123,
    tick: 0,
    mode: "api",
    running: false,
    settings: {
      width: 800,
      height: 450,
      paddleWidth: 10,
      paddleHeight: 80,
      ballSize: 10,
      fixedDt: 1 / 60,
      pollMs: 120,
      eventPollMs: 8,
      statePushMs: 16,
      simulationSpeed: 1,
    },
  };

  const context = {
    console,
    BroadcastChannel: FakeBroadcastChannel,
    KeyboardEvent: class KeyboardEvent {
      constructor(type, options) {
        this.type = type;
        Object.assign(this, options);
      }
      preventDefault() {}
    },
    Math,
    document: {
      createElement() {
        return { value: "", textContent: "" };
      },
      getElementById(id) {
        return elements[id];
      },
    },
    fetch: async (url, options = {}) => {
      const href = String(url);
      let body = {};
      if (href.includes("/api/session")) {
        body = session;
      } else if (href.includes("/api/events")) {
        body = { ...session, events: [] };
      } else if (href.includes("/api/start")) {
        session.running = true;
        body = { ...session, event: { seq: 1, type: "start", tick: 0 } };
      } else if (href.includes("/api/event")) {
        session.running = false;
        body = { ...session, event: { seq: 2, type: "pause", tick: 0 } };
      } else if (href.includes("/api/world/state")) {
        postedObservations.push(JSON.parse(options.body));
        body = { accepted: true };
      } else if (href.includes("/api/state")) {
        body = postedObservations.at(-1) || { ...session, source: "waiting-for-world" };
      } else if (href.includes("/api/snn/start")) {
        body = { ...session, snn: { training: true, paused: false, activity: { outputBars: [0, 0, 1] } } };
      } else if (href.includes("/api/snn/pause")) {
        body = { ...session, snn: { training: true, paused: true, activity: { outputBars: [0, 0, 1] } } };
      } else if (href.includes("/api/snn/reset")) {
        session.running = false;
        session.resetToken += 1;
        body = {
          ...session,
          source: "waiting-for-world",
          tick: 0,
          frameSeq: 0,
          score: { left: 0, right: 0 },
          ball: { x: 400, y: 225, vx: 0, vy: 0 },
          paddles: { leftY: 185, rightY: 185 },
          snn: { training: false, paused: true, activity: { outputBars: [0, 0, 1] } },
        };
      } else if (href.includes("/api/snn/status")) {
        body = {
          training: true,
          paused: false,
          activity: {
            outputBars: [0.2, 0.7, 0.1],
            winner: "move down",
            spikes: {},
            stdp: {},
            reward: { value: 0, recentRightScores: 0, recentOpponentScores: 0 },
            eligibility: {},
            learning: { step: 0, weightUpdates: 0 },
          },
          architecture: { device: { active: "cpu", cudaAvailable: false }, input: { width: 800, height: 450 }, layers: [] },
        };
      } else if (href.includes("/api/snn/saves")) {
        body = { saves: [] };
      }
      return {
        ok: true,
        json: async () => body,
      };
    },
    performance: { now: () => 42 },
    requestAnimationFrame() {},
    setTimeout() {},
    clearTimeout() {},
  };

  context.window = {
    __drawCalls: drawCalls,
    __postedObservations: postedObservations,
    devicePixelRatio: 1,
    innerWidth: 800,
    innerHeight: 450,
    BroadcastChannel: FakeBroadcastChannel,
    addEventListener(type, listener) {
      if (!listeners.has(type)) {
        listeners.set(type, []);
      }
      listeners.get(type).push(listener);
    },
    dispatchEvent(event) {
      for (const listener of listeners.get(event.type) || []) {
        listener(event);
      }
    },
    setTimeout: context.setTimeout,
    clearTimeout: context.clearTimeout,
    requestAnimationFrame: context.requestAnimationFrame,
  };
  context.globalThis = context;

  vm.createContext(context);
  vm.runInContext(coreSource, context, { filename: "pong-core.js" });
  vm.runInContext(eventCameraSource, context, { filename: "event-camera.js" });
  vm.runInContext(worldRunnerSource, context, { filename: "world-runner.js" });
  context.PongCore = context.window.PongCore;
  context.EventCamera = context.window.EventCamera;
  context.PongWorldRunner = context.window.PongWorldRunner;
  return context;
}

const visualizer = fakeContext();
vm.runInContext(visualizerSource, visualizer, { filename: "visualizer.js" });

const game = fakeContext();
vm.runInContext(gameSource, game, { filename: "game.js" });
vm.runInContext(
  `
  window.__pongClient.resetFromSession({
    sessionId: "stream-smoke",
    resetToken: 3,
    seed: 123,
    settings: {
      width: 800,
      height: 450,
      paddleWidth: 10,
      paddleHeight: 80,
      ballSize: 10,
      fixedDt: 1 / 60,
      pollMs: 120,
      eventPollMs: 8,
      statePushMs: 16,
      simulationSpeed: 1
    }
  });
  window.__pongClient.queueEvents([{ seq: 1, type: "start", tick: 0 }]);
  window.__pongClient.frame(0);
  window.__pongClient.frame(17);
  window.__pongClient.frame(34);
  `,
  game
);

const latestGameFrame = game.window.__pongLatestFrame;
const latestVisualizerFrame = visualizer.window.__pongViewerState;
const posted = game.window.__postedObservations.at(-1);
const asJson = (value) => JSON.stringify(value);

assert(latestGameFrame, "game should produce its own world frame");
assert(posted, "game should publish observations to the backend stream");
assert(latestVisualizerFrame, "visualizer should receive game-produced frames over the passive channel");
assert.strictEqual(latestVisualizerFrame.source, "game-channel");
assert.strictEqual(latestGameFrame.source, "game-renderer");
assert.strictEqual(latestGameFrame.eventCamera.source, "event-camera");
assert.strictEqual(latestVisualizerFrame.frameSeq, latestGameFrame.frameSeq);
assert.strictEqual(latestVisualizerFrame.tick, latestGameFrame.tick);
assert.strictEqual(asJson(latestVisualizerFrame.score), asJson(latestGameFrame.score));
assert.strictEqual(asJson(latestVisualizerFrame.ball), asJson(latestGameFrame.ball));
assert.strictEqual(asJson(latestVisualizerFrame.paddles), asJson(latestGameFrame.paddles));
assert.strictEqual(asJson(latestVisualizerFrame.eventCamera), asJson(latestGameFrame.eventCamera));
assert(posted.frameSeq <= latestGameFrame.frameSeq, "observation posting should not block newer local render frames");
assert.strictEqual(posted.source, "game-renderer");
assert.strictEqual(posted.eventCamera.source, "event-camera");

visualizer.window.__drawCalls.length = 0;
vm.runInContext("for (let i = 0; i < 2; i += 1) render();", visualizer);
assert(
  visualizer.window.__drawCalls.some((call) => call.fillStyle === "rgba(255, 0, 0, 1.000)"),
  "visualizer should draw game event-camera pixels as an opaque red overlay"
);

const controlPromise = vm.runInContext(
  `
  window.__controlPromise = (async () => {
    await startSnn();
    window.__pongViewer.worldRunner.frame(51);
    window.__pongViewer.worldRunner.frame(68);
    window.__startButtonState = {
      running: window.__pongViewer.worldRunner.sim.running,
      tick: window.__pongViewer.worldRunner.sim.tick,
      frameSeq: window.__pongViewerState?.frameSeq,
      eventCount: window.__pongViewerState?.eventCamera?.count,
      trainState: document.getElementById("trainState").textContent
    };
    await pauseSnn();
    window.__pongViewer.worldRunner.frame(85);
    window.__pauseButtonState = {
      running: window.__pongViewer.worldRunner.sim.running,
      tick: window.__pongViewer.worldRunner.sim.tick
    };
    await resetGame();
    window.__resetButtonState = {
      running: window.__pongViewer.worldRunner.sim.running,
      tick: window.__pongViewer.worldRunner.sim.tick,
      resetToken: window.__pongViewer.worldRunner.resetToken
    };
  })()
  `,
  visualizer
);
const resolvedControlPromise = controlPromise || visualizer.window.__controlPromise;

resolvedControlPromise
  .then(() => {
    assert.strictEqual(visualizer.window.__startButtonState.running, true, "visualizer start should start the world runner");
    assert(visualizer.window.__startButtonState.tick > 0, "visualizer start should advance world ticks");
    assert(
      visualizer.window.__startButtonState.frameSeq >= 0,
      "visualizer start should produce a world/event-camera frame"
    );
    assert.strictEqual(visualizer.window.__pauseButtonState.running, false, "visualizer pause should pause the world runner");
    assert.strictEqual(visualizer.window.__resetButtonState.running, false, "visualizer reset should leave the world paused");
    assert.strictEqual(visualizer.window.__resetButtonState.tick, 0, "visualizer reset should reset world tick");

    console.log(
      JSON.stringify(
        {
          ok: true,
          frameSeq: latestVisualizerFrame.frameSeq,
          tick: latestVisualizerFrame.tick,
          source: latestVisualizerFrame.source,
          resetToken: latestVisualizerFrame.resetToken,
          startButton: visualizer.window.__startButtonState,
          pauseButton: visualizer.window.__pauseButtonState,
          resetButton: visualizer.window.__resetButtonState,
        },
        null,
        2
      )
    );
  })
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
