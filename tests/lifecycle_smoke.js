const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const root = path.resolve(__dirname, "..");
const coreSource = fs.readFileSync(path.join(root, "web", "pong-core.js"), "utf8");

const context = { window: {} };
context.globalThis = context;
vm.createContext(context);
vm.runInContext(coreSource, context, { filename: "pong-core.js" });

const PongCore = context.window.PongCore;
const sim = PongCore.createSimulation({ seed: 123 });
PongCore.start(sim);
for (let index = 0; index < 20; index += 1) {
  PongCore.step(sim, []);
}

const pausedBall = { ...sim.ball };
const pausedTick = sim.tick;
PongCore.step(sim, [{ type: "pause", tick: sim.tick }]);
assert.strictEqual(sim.running, false, "pause event should stop the simulation");

for (let index = 0; index < 5; index += 1) {
  PongCore.step(sim, []);
}
assert.strictEqual(sim.ball.x, pausedBall.x, "paused ball x should remain fixed");
assert.strictEqual(sim.ball.y, pausedBall.y, "paused ball y should remain fixed");

PongCore.step(sim, [{ type: "start", tick: sim.tick }]);
assert.strictEqual(sim.running, true, "start after pause should resume the simulation");
assert.strictEqual(sim.ball.vx, pausedBall.vx, "resume should preserve ball vx");
assert.strictEqual(sim.ball.vy, pausedBall.vy, "resume should preserve ball vy");
assert(Math.abs(sim.ball.x - (pausedBall.x + pausedBall.vx * sim.settings.fixedDt)) < 1e-9, "resume should continue ball x");
assert(Math.abs(sim.ball.y - (pausedBall.y + pausedBall.vy * sim.settings.fixedDt)) < 1e-9, "resume should continue ball y");
assert.notStrictEqual(sim.ball.x, sim.settings.width / 2, "resume should not re-center ball x");
assert.notStrictEqual(sim.ball.y, sim.settings.height / 2, "resume should not re-center ball y");
assert.notStrictEqual(sim.tick, pausedTick, "ticks should continue advancing while paused");

console.log(JSON.stringify({ ok: true, pausedBall, resumedTick: sim.tick }));
