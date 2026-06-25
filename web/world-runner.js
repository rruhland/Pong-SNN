(function () {
  const FRAME_MS = 1000 / 60;
  const MAX_STEPS_PER_FRAME = 5;

  function createWorldRunner(options) {
    const canvas = options.canvas;
    const ctx = canvas.getContext("2d");
    const stateChannel = "BroadcastChannel" in window ? new BroadcastChannel("pong-snn-state") : null;
    const runner = {
      canvas,
      ctx,
      sim: null,
      eventCamera: null,
      sessionId: null,
      resetToken: null,
      seed: null,
      keys: new Set(),
      pendingEvents: [],
      lastEventSeq: 0,
      lastPublishedDirection: 0,
      pollMs: 120,
      eventPollMs: 8,
      postMs: 16,
      simulationSpeed: 1,
      frameSeq: 0,
      frameHistory: [],
      pendingWorldFrames: new Map(),
      postingObservation: false,
      lastPostAt: -Infinity,
      lastFrameAt: null,
      accumulatorMs: 0,
      started: false,
      stateChannel,
      onFrame: typeof options.onFrame === "function" ? options.onFrame : null,
    };

    function normalizeSessionSettings(settings) {
      return { ...PongCore.DEFAULT_SETTINGS, ...(settings || {}) };
    }

    function createLocalSimulation(seed = 1, settings = PongCore.DEFAULT_SETTINGS) {
      runner.seed = seed;
      runner.sim = PongCore.createSimulation({ seed, settings });
      if (!runner.eventCamera) {
        runner.eventCamera = EventCamera.create({
          width: settings.width,
          height: settings.height,
          source: "event-camera",
          onEventCamera: acceptEventCamera,
        });
      } else {
        runner.eventCamera.width = settings.width;
        runner.eventCamera.height = settings.height;
        runner.eventCamera.reset();
      }
      runner.pendingEvents = [];
      runner.lastEventSeq = 0;
      runner.lastPublishedDirection = 0;
      runner.frameSeq = 0;
      runner.frameHistory = [];
      runner.pendingWorldFrames.clear();
      runner.accumulatorMs = 0;
      runner.lastFrameAt = null;
    }

    function mergeSettings(settings) {
      if (!runner.sim) return;
      runner.sim.settings = normalizeSessionSettings(settings);
      runner.sim.left.speed = 240;
      runner.sim.right.speed = 360;
      runner.sim.left.x = 24;
      runner.sim.right.x = runner.sim.settings.width - 34;
      runner.simulationSpeed = Number(settings?.simulationSpeed ?? runner.simulationSpeed ?? 1);
    }

    function resetFromSession(session) {
      const settings = normalizeSessionSettings(session.settings);
      runner.sessionId = session.sessionId;
      runner.resetToken = session.resetToken;
      createLocalSimulation(Number(session.seed || 1), settings);
      if (session.running) {
        PongCore.start(runner.sim);
      }
      runner.pollMs = Number(session.settings?.pollMs || runner.pollMs);
      runner.eventPollMs = Number(session.settings?.eventPollMs || runner.eventPollMs);
      runner.postMs = Number(session.settings?.statePushMs || runner.postMs);
      runner.simulationSpeed = Number(session.settings?.simulationSpeed || 1);
      render();
    }

    function normalizeEvent(rawEvent) {
      if (!rawEvent || typeof rawEvent !== "object") return null;
      const tick = Number(rawEvent.tick ?? 0);
      const type = rawEvent.type || "input";
      if (!Number.isFinite(tick) || !["start", "pause", "input"].includes(type)) return null;
      return {
        ...rawEvent,
        type,
        tick: Math.max(0, Math.floor(tick)),
        seq: Number(rawEvent.seq || 0),
        direction: type === "input" ? Math.max(-1, Math.min(1, Number(rawEvent.direction || 0))) : undefined,
      };
    }

    function queueEvents(events) {
      for (const rawEvent of events || []) {
        const event = normalizeEvent(rawEvent);
        if (!event) continue;
        if (event.seq > 0) {
          runner.lastEventSeq = Math.max(runner.lastEventSeq, event.seq);
        }
        runner.pendingEvents.push(event);
      }
      runner.pendingEvents.sort((a, b) => (a.tick - b.tick) || ((a.seq || 0) - (b.seq || 0)));
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

    function localEvent(type, body = {}) {
      return {
        localId: `${type}:${performance.now()}:${Math.random()}`,
        type,
        tick: Math.floor(runner.sim?.tick || 0),
        ...body,
      };
    }

    function currentKeyboardDirection() {
      const up = runner.keys.has("ArrowUp") || runner.keys.has("KeyW");
      const down = runner.keys.has("ArrowDown") || runner.keys.has("KeyS");
      if (up && !down) return -1;
      if (down && !up) return 1;
      return 0;
    }

    async function pollSession() {
      try {
        const response = await fetch("/api/session", { cache: "no-store" });
        if (response.ok) {
          const session = await response.json();
          const isReset =
            !runner.sim ||
            runner.sessionId !== session.sessionId ||
            runner.resetToken !== session.resetToken;
          if (isReset) {
            resetFromSession(session);
          } else {
            mergeSettings(session.settings);
            runner.pollMs = Number(session.settings?.pollMs || runner.pollMs);
            runner.eventPollMs = Number(session.settings?.eventPollMs || runner.eventPollMs);
            runner.postMs = Number(session.settings?.statePushMs || runner.postMs);
          }
        }
      } catch {
        // The world keeps running locally if the stream server is briefly unavailable.
      } finally {
        window.setTimeout(pollSession, runner.pollMs);
      }
    }

    async function pollEvents() {
      try {
        const response = await fetch(`/api/events?sinceSeq=${runner.lastEventSeq}`, { cache: "no-store" });
        if (response.ok) {
          const payload = await response.json();
          queueEvents(payload.events || []);
        }
      } catch {
        // Hold the last actuator direction until the event stream returns.
      } finally {
        window.setTimeout(pollEvents, runner.eventPollMs);
      }
    }

    async function start() {
      if (!runner.sim) return;
      runner.started = true;
      queueEvents([localEvent("start")]);
      await postJson("/api/start", { tick: Math.floor(runner.sim.tick) });
    }

    async function pause() {
      if (!runner.sim) return;
      queueEvents([localEvent("pause")]);
      await postJson("/api/event", { type: "pause", tick: Math.floor(runner.sim.tick) });
    }

    async function publishInput(direction, source = "human") {
      if (!runner.sim || direction === runner.lastPublishedDirection) return;
      runner.lastPublishedDirection = direction;
      queueEvents([localEvent("input", { direction, source })]);
      await postJson("/api/input", {
        tick: Math.floor(runner.sim.tick),
        direction,
        source,
      });
    }

    function handleInputChange() {
      publishInput(currentKeyboardDirection(), "human");
    }

    function makeWorldFrame() {
      const snapshot = PongCore.snapshot(runner.sim);
      const renderedAt = performance.now();
      return {
        ...snapshot,
        sessionId: runner.sessionId,
        resetToken: runner.resetToken,
        seed: runner.seed,
        authoritativeTick: snapshot.tick,
        frameSeq: runner.frameSeq,
        renderedAt,
        source: "game-renderer",
      };
    }

    function acceptEventCamera(eventCamera) {
      const worldFrame = runner.pendingWorldFrames.get(eventCamera.frameSeq);
      if (!worldFrame) return;
      runner.pendingWorldFrames.delete(eventCamera.frameSeq);
      const state = { ...worldFrame, eventCamera };
      broadcastState(state);
      publishObservation(state);

      const minFrameSeq = eventCamera.frameSeq - 12;
      for (const frameSeq of runner.pendingWorldFrames.keys()) {
        if (frameSeq < minFrameSeq) {
          runner.pendingWorldFrames.delete(frameSeq);
        }
      }
    }

    function observeWorldFrame(state) {
      runner.pendingWorldFrames.set(state.frameSeq, state);
      runner.eventCamera.observeCanvas(canvas, {
        width: state.settings.width,
        height: state.settings.height,
        tick: state.tick,
        frameSeq: state.frameSeq,
        resetToken: state.resetToken,
        renderedAt: state.renderedAt,
        source: "event-camera",
      });
    }

    function broadcastState(state) {
      runner.frameHistory.push(state);
      if (runner.frameHistory.length > 180) {
        runner.frameHistory.shift();
      }
      if (runner.stateChannel) {
        runner.stateChannel.postMessage(state);
      }
      window.__pongLatestFrame = state;
      window.__pongFrameHistory = runner.frameHistory;
      if (runner.onFrame) {
        runner.onFrame(state);
      }
    }

    function publishObservation(state, now = performance.now()) {
      if (runner.postingObservation || now - runner.lastPostAt < runner.postMs) return;
      runner.postingObservation = true;
      runner.lastPostAt = now;
      postJson("/api/world/state", state).finally(() => {
        runner.postingObservation = false;
      });
    }

    function stepWorld() {
      if (!runner.sim) {
        createLocalSimulation(1, PongCore.DEFAULT_SETTINGS);
      }
      const due = [];
      const future = [];
      for (const event of runner.pendingEvents) {
        if (event.tick <= runner.sim.tick) {
          due.push(event);
        } else {
          future.push(event);
        }
      }
      runner.pendingEvents = future;
      PongCore.step(runner.sim, due, { simulationSpeed: runner.simulationSpeed });
    }

    function render() {
      if (!runner.sim) {
        createLocalSimulation(1, PongCore.DEFAULT_SETTINGS);
      }
      const width = canvas.clientWidth || window.innerWidth;
      const height = canvas.clientHeight || window.innerHeight;
      PongCore.draw(runner.ctx, width, height, runner.sim);
    }

    function frame(now) {
      if (runner.lastFrameAt === null) {
        runner.lastFrameAt = now;
      }
      runner.accumulatorMs += Math.min(250, now - runner.lastFrameAt);
      runner.lastFrameAt = now;

      let steps = 0;
      while (runner.accumulatorMs >= FRAME_MS && steps < MAX_STEPS_PER_FRAME) {
        stepWorld();
        runner.accumulatorMs -= FRAME_MS;
        steps += 1;
      }

      if (steps > 0) {
        render();
        const state = makeWorldFrame();
        observeWorldFrame(state);
        runner.frameSeq += 1;
      }

      requestAnimationFrame(frame);
    }

    function resize(width, height) {
      const dpr = window.devicePixelRatio || 1;
      const cssWidth = Math.max(1, Math.floor(width || canvas.clientWidth || window.innerWidth));
      const cssHeight = Math.max(1, Math.floor(height || canvas.clientHeight || window.innerHeight));
      canvas.width = Math.floor(cssWidth * dpr);
      canvas.height = Math.floor(cssHeight * dpr);
      runner.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      render();
    }

    function keydown(event) {
      runner.keys.add(event.code);
      start();
      handleInputChange();
      if (["ArrowUp", "ArrowDown", "Space"].includes(event.code)) {
        event.preventDefault();
      }
    }

    function keyup(event) {
      runner.keys.delete(event.code);
      handleInputChange();
      if (["ArrowUp", "ArrowDown", "Space"].includes(event.code)) {
        event.preventDefault();
      }
    }

    runner.resize = resize;
    runner.resetFromSession = resetFromSession;
    runner.queueEvents = queueEvents;
    runner.start = start;
    runner.pause = pause;
    runner.publishInput = publishInput;
    runner.render = render;
    runner.frame = frame;
    runner.attachKeyboard = () => {
      window.addEventListener("keydown", keydown);
      window.addEventListener("keyup", keyup);
    };
    runner.begin = () => {
      createLocalSimulation(1, PongCore.DEFAULT_SETTINGS);
      resize(options.width, options.height);
      pollSession();
      pollEvents();
      requestAnimationFrame(frame);
    };

    return runner;
  }

  window.PongWorldRunner = { create: createWorldRunner };
})();
