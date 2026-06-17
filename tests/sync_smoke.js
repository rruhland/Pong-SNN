const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const root = path.resolve(__dirname, "..");
const coreSource = fs.readFileSync(path.join(root, "web", "pong-core.js"), "utf8");
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

function fakeContext(backendFrame) {
  const listeners = new Map();
  const drawCalls = [];
  let fillStyle = "";
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
    hitMissReadout: { textContent: "" },
    learningReadout: { textContent: "" },
    device: { textContent: "" },
    trainState: { textContent: "" },
    reset: { addEventListener() {} },
    start: { addEventListener() {} },
    pause: { addEventListener() {} },
    saveNetwork: { addEventListener() {} },
    loadNetwork: { addEventListener() {} },
    savedNetworks: { value: "", textContent: "", appendChild() {}, options: [] },
    trailMs: { value: "120", addEventListener() {} },
    trailValue: { textContent: "" },
    barUp: { style: {} },
    barDown: { style: {} },
    barStay: { style: {} },
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
    document: {
      createElement() {
        return { value: "", textContent: "" };
      },
      getElementById(id) {
        return elements[id];
      },
    },
    fetch: async (url) => {
      const href = String(url);
      let body = {};
      if (href.includes("/api/state")) {
        body = backendFrame;
      } else if (href.includes("/api/session")) {
        body = {
          sessionId: backendFrame.sessionId,
          resetToken: backendFrame.resetToken,
          seed: backendFrame.seed,
          tick: backendFrame.tick,
          mode: "api",
          running: backendFrame.running,
          settings: { pollMs: 16, statePushMs: 16 },
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
            reward: { value: 0, recentHits: 0, recentMisses: 0 },
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
    performance: { now: () => 1 },
    requestAnimationFrame() {},
    setTimeout() {},
  };

  context.window = {
    __drawCalls: drawCalls,
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
    requestAnimationFrame: context.requestAnimationFrame,
  };
  context.globalThis = context;

  vm.createContext(context);
  vm.runInContext(coreSource, context, { filename: "pong-core.js" });
  context.PongCore = context.window.PongCore;
  return context;
}

const backendFrame = {
  sessionId: "backend-smoke",
  resetToken: 3,
  seed: 123,
  tick: 42,
  authoritativeTick: 42,
  frameSeq: 42,
  running: true,
  settings: { width: 800, height: 450, paddleWidth: 10, paddleHeight: 80, ballSize: 10, fixedDt: 1 / 60 },
  score: { left: 2, right: 1 },
  ball: { x: 500, y: 180, vx: 200, vy: -30 },
  paddles: { leftY: 140, rightY: 220 },
  eventCamera: {
    width: 800,
    height: 450,
    tick: 42,
    frameSeq: 42,
    resetToken: 3,
    source: "backend",
    pixels: [1000, 1001, 1002],
    count: 3,
  },
  source: "backend-sim",
};

const visualizer = fakeContext(backendFrame);
vm.runInContext(visualizerSource, visualizer, { filename: "visualizer.js" });

const game = fakeContext(backendFrame);
vm.runInContext(gameSource, game, { filename: "game.js" });

vm.runInContext("acceptState(" + JSON.stringify(backendFrame) + "); render();", game);

const latestGameFrame = game.window.__pongLatestFrame;
const latestVisualizerFrame = visualizer.window.__pongViewerState;
const asJson = (value) => JSON.stringify(value);

assert(latestGameFrame, "game should accept backend frame");
assert(latestVisualizerFrame, "visualizer should receive game-broadcast backend frame");
assert.strictEqual(latestVisualizerFrame.source, "game-channel");
assert.strictEqual(latestVisualizerFrame.frameSeq, backendFrame.frameSeq);
assert.strictEqual(latestVisualizerFrame.tick, backendFrame.tick);
assert.strictEqual(asJson(latestVisualizerFrame.score), asJson(backendFrame.score));
assert.strictEqual(asJson(latestVisualizerFrame.ball), asJson(backendFrame.ball));
assert.strictEqual(asJson(latestVisualizerFrame.paddles), asJson(backendFrame.paddles));
assert.strictEqual(asJson(latestVisualizerFrame.eventCamera), asJson(backendFrame.eventCamera));

visualizer.window.__drawCalls.length = 0;
vm.runInContext("for (let i = 0; i < 2; i += 1) render();", visualizer);
assert(
  visualizer.window.__drawCalls.some((call) => call.fillStyle === "rgba(255, 0, 0, 1.000)"),
  "visualizer should draw backend event camera pixels as an opaque red overlay"
);

console.log(
  JSON.stringify(
    {
      ok: true,
      frameSeq: latestVisualizerFrame.frameSeq,
      tick: latestVisualizerFrame.tick,
      source: latestVisualizerFrame.source,
      resetToken: latestVisualizerFrame.resetToken,
    },
    null,
    2
  )
);
