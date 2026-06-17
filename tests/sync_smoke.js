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

function fakeContext() {
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
        setTransform() {},
        set fillStyle(value) {
          fillStyle = value;
        },
      };
    },
  };
  const elements = {
    game: canvas,
    preview: canvas,
    score: { textContent: "" },
    reset: { addEventListener() {} },
    trailMs: { value: "120", addEventListener() {} },
    trailValue: { textContent: "" },
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
      getElementById(id) {
        return elements[id];
      },
    },
    fetch: async (url) => {
      const body = url.includes("/api/session")
        ? {
            sessionId: "smoke-session",
            resetToken: 0,
            seed: 123,
            tick: 0,
            mode: "human",
            settings: { pollMs: 16, statePushMs: 16 },
          }
        : { events: [] };
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

const visualizer = fakeContext();
vm.runInContext(visualizerSource, visualizer, { filename: "visualizer.js" });

const game = fakeContext();
vm.runInContext(gameSource, game, { filename: "game.js" });

vm.runInContext(
  `
  PongCore.start(client.sim);
  for (let i = 0; i < 12; i += 1) {
    PongCore.step(client.sim, []);
    client.frameSeq += 1;
    broadcastState(frameState(100 + i));
  }
`,
  game
);

const latestGameFrame = game.window.__pongLatestFrame;
const latestVisualizerFrame = visualizer.window.__pongViewerState;
const asJson = (value) => JSON.stringify(value);

assert(latestVisualizerFrame, "visualizer should accept game frame");
assert.strictEqual(latestVisualizerFrame.source, "game-channel");
assert.strictEqual(latestVisualizerFrame.frameSeq, latestGameFrame.frameSeq);
assert.strictEqual(latestVisualizerFrame.tick, latestGameFrame.tick);
assert.strictEqual(asJson(latestVisualizerFrame.score), asJson(latestGameFrame.score));
assert.strictEqual(asJson(latestVisualizerFrame.ball), asJson(latestGameFrame.ball));
assert.strictEqual(asJson(latestVisualizerFrame.paddles), asJson(latestGameFrame.paddles));
assert(latestGameFrame.eventCamera, "game frame should include event camera output");
assert(Array.isArray(latestGameFrame.eventCamera.pixels), "event camera pixels should be an array");
assert(latestGameFrame.eventCamera.pixels.length > 0, "moving game objects should emit event pixels");
assert.strictEqual(asJson(latestVisualizerFrame.eventCamera), asJson(latestGameFrame.eventCamera));

visualizer.window.__drawCalls.length = 0;
vm.runInContext("for (let i = 0; i < 10; i += 1) render();", visualizer);
assert.strictEqual(visualizer.window.__pongViewerState.frameSeq, latestGameFrame.frameSeq);
assert(
  visualizer.window.__drawCalls.some((call) => call.fillStyle === "rgba(255, 0, 0, 1.000)"),
  "visualizer should draw fresh event camera pixels as an opaque red overlay"
);

vm.runInContext(
  `
  const resetState = {
    ...frameState(200),
    frameSeq: 0,
    tick: 0,
    resetToken: 1,
    sessionId: "smoke-session-reset"
  };
  broadcastState(resetState);
`,
  game
);
assert.strictEqual(visualizer.window.__pongViewerState.resetToken, 1);
assert.strictEqual(visualizer.window.__pongViewerState.frameSeq, 0);

console.log(
  JSON.stringify(
    {
      ok: true,
      frameSeq: visualizer.window.__pongViewerState.frameSeq,
      tick: visualizer.window.__pongViewerState.tick,
      source: visualizer.window.__pongViewerState.source,
      resetToken: visualizer.window.__pongViewerState.resetToken,
    },
    null,
    2
  )
);
