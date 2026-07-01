# Pong-SNN

A local Pong world with a decoupled event-camera/SNN control loop.

The browser game owns Pong physics, rendering, and event-camera capture. The
Python server is a stream/control service: it receives world observations,
steps the SNN on its own worker thread when it can, and publishes actuator
commands back as input events. The visualizer passively watches the same
streams.

```text
Game renderer
  -> renders Pong and advances world state at 60 FPS
  -> publishes world observations to /api/world/state

Event camera
  -> passively observes rendered pixels from a canvas/video source
  -> timestamps visual changes and emits the event-window contract

SNN worker
  <- samples the latest event-camera frame when ready
  -> emits up/down/hold actuator events

Actuator bridge
  -> game polls /api/events and holds the latest input direction

Visualizer
  <- game frames over BroadcastChannel when open beside the game
  <- latest backend-observed world frame as fallback
  <- SNN status/activity from /api/snn/status
```

The SNN does not run inside the render loop. The live event-camera path is not
Pong-state-derived either: `web/event-camera.js` reads rendered pixels and sends
them to `web/event-camera-worker.js` for visual differencing. If the camera,
backend, or SNN falls behind, the game keeps rendering and the SNN learns from
the newest event window it can process.

## Run

```powershell
python server.py
```

If Python is not on PATH in this Codex environment, the bundled runtime works:

```powershell
C:\Users\rruhl\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe server.py
```

Open the game and visualizer:

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/visualizer
```

The game page is the world. The event camera is a separate observer attached to
the rendered canvas. The visualizer is a controller and observer, but it does
not generate Pong physics or train from its own canvas. Press `start` in the
visualizer to start SNN training and send a start event to the game.

The event camera boundary is intentionally portable. An adapter for an online
Pong page should only need to provide a visual source plus metadata:

```js
const camera = EventCamera.create({ width: 800, height: 450 });
const eventCamera = await camera.observeCanvas(canvasOrCopiedFrame, {
  tick,
  frameSeq,
  resetToken,
  renderedAt,
  source: "event-camera",
});
```

The emitted `eventCamera.pixels` format is the same one consumed by the SNN.
Only the actuator bridge would need to map `-1`, `0`, and `1` onto the target
game's controls.

## Backend SNN

The SNN backend now uses a compressed, recurrent predictive architecture:

- Input: the `800 x 450` event-camera pixels are compressed into a `64 x 36`
  event grid, with one input neuron per changed cell. Each active cell fans out
  to about two dozen nearby hidden-cloud targets instead of a tiny fixed stencil.
- Motor context: the previous `move up`, `move down`, and `stay put` actions are
  fed back as three decaying traces.
- Hidden cloud: about `5,000` recurrent neurons, with roughly `80%` excitatory
  and `20%` inhibitory cells. Initial recurrent connectivity is sparse and
  distance-biased, with stronger nearby edges and weaker long-range edges.
- Outputs: a sparse three-neuron motor readout for `move up`, `move down`, and
  `stay put`, plus a prediction head that predicts the next `64 x 36` event
  grid from mostly local edges with a few medium and long-range targets.

Training uses STDP eligibility traces with separate reward and self-supervised
prediction signals. Active pre plus active post increases each synapse group's
fast, medium, and slow eligibility traces; active pre without post decreases
them slightly. The traces decay independently at roughly 500 ms, 2 s, and 5 s
scales, then combine as `fast + 0.35 * medium + 0.08 * slow`. Prediction error
trains the visual input, recurrent world-model, and prediction-head pathways.
Pong reward trains motor-context and sparse motor-output pathways, plus a small
motor-adjacent subset of recurrent synapses.

Reward is assembled from named, tunable components in `RewardFunction`, but the
signal is intentionally sparse and game-level. Up/down movement receives a small
cost, doing nothing has no direct cost or reward, opponent score receives a large
negative reward, right-paddle score receives a larger positive reward, and each
surviving frame receives a small `log1p(seconds_since_restart)` reward. The
reward path does not use paddle-ball distance, output confidence, or other
engineered Pong-state shaping terms.

SNN controls:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/snn/start
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/snn/pause
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/snn/reset -ContentType "application/json" -Body '{"resetWeights":true,"resetScore":true}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/snn/save -ContentType "application/json" -Body '{"name":"pong-snn"}'
Invoke-RestMethod http://127.0.0.1:8000/api/snn/status
```

Saved networks are JSON files under `network_saves/`.

## Controls

- `ArrowUp` or `W`: move the right paddle up manually
- `ArrowDown` or `S`: move the right paddle down manually

Manual input is posted as input events. SNN actions use the same event stream,
so the game only needs an actuator bridge that applies the latest event and
holds the last direction until a new one arrives.

## Stream API

Publish a world observation:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/world/state -ContentType "application/json" -Body '{"tick":1,"frameSeq":1,"running":true,"score":{"left":0,"right":0},"ball":{"x":400,"y":220,"vx":200,"vy":0},"paddles":{"leftY":180,"rightY":180},"eventCamera":{"width":800,"height":450,"tick":1,"frameSeq":1,"pixels":[1000,1001]}}'
```

Read the latest backend-observed world frame:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/state
```

Read session and stream metadata:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/session
```

Read event/action streams:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/api/events?sinceSeq=0"
Invoke-RestMethod "http://127.0.0.1:8000/api/inputs?sinceSeq=0"
Invoke-RestMethod http://127.0.0.1:8000/api/actions/latest
```

Adjust browser world speed without changing the render/SNN boundary:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/sim-speed -ContentType "application/json" -Body '{"speed":1.75}'
```

Reset the session:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/reset -ContentType "application/json" -Body '{"resetScore":true}'
```

## Smoke Tests

```powershell
node tests/lifecycle_smoke.js
node tests/sync_smoke.js
C:\Users\rruhl\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe tests\snn_smoke.py
```

`sync_smoke.js` verifies that the game produces frames locally and the
visualizer observes them passively. `snn_smoke.py` verifies that the SNN does
not advance from an empty backend, then learns after world observations are
posted.
