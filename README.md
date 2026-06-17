# Pong-SNN

A minimal local Pong surface for later SNN/event-camera experiments.

The browser shows only the Pong game. Score, control mode, and computer inputs
are handled through the local backend API.

## Run

```powershell
python server.py
```

If Python is not on PATH, run the equivalent dependency-free Node server:

```powershell
node server.js
```

Open http://127.0.0.1:8000 and press any key to start.

Open the separate visualizer in another window or side-by-side browser tab:

```text
http://127.0.0.1:8000/visualizer
```

The game page is the source of truth. It runs the Pong simulation, publishes the
exact rendered frame over a browser `BroadcastChannel`, and checkpoints the same
frame to `/api/state` at frame-rate cadence. The visualizer is passive and
mirror-only: it never advances its own Pong simulation, so a virtual event camera
should attach to the game page while the visualizer shows that same live game
state in real time.

Each published game frame now includes `eventCamera`, a game-side virtual event
camera payload. `eventCamera.pixels` is a list of changed logical game-pixel
indices, encoded as `y * eventCamera.width + x`. The visualizer draws those
events as opaque red pixels over the mirrored Pong view; it does not sample its
own canvas to generate events.

## Controls

The right paddle defaults to human keyboard control:

- `ArrowUp` or `W`: move up
- `ArrowDown` or `S`: move down

The left paddle is always an auto-player.

## Backend API

Switch the right paddle to computer input:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/control-mode -ContentType "application/json" -Body '{"mode":"api"}'
```

Drive the right paddle:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/input -ContentType "application/json" -Body '{"tick":1234,"direction":-1}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/input -ContentType "application/json" -Body '{"tick":1235,"direction":0}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/input -ContentType "application/json" -Body '{"tick":1236,"direction":1}'
```

Use `-1` for up, `0` for stop, and `1` for down. The `tick` field schedules
the input for a specific fixed simulation step. For temporary backward
compatibility, omitting `tick` schedules the input on the next authoritative
backend tick.

Return to keyboard control:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/control-mode -ContentType "application/json" -Body '{"mode":"human"}'
```

Read backend state:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/state
```

Read session metadata:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/session
```

Read tick-indexed inputs after a tick:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/api/inputs?sinceTick=1200"
```

Start is also a tick-indexed event:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/start -ContentType "application/json" -Body '{"tick":120}'
```

Read the full session event stream:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/api/events?sinceSeq=0"
```

Reset the game and score:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/reset -ContentType "application/json" -Body '{"resetScore":true}'
```

## Sync smoke test

```powershell
node tests/sync_smoke.js
```

The smoke test runs the game and visualizer scripts in separate browser-like
contexts with a shared frame channel. It fails if the visualizer mutates or
reconstructs state instead of preserving the exact frame published by the game.
