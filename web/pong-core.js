(function () {
  const DEFAULT_SETTINGS = {
    width: 800,
    height: 450,
    paddleWidth: 10,
    paddleHeight: 80,
    ballSize: 10,
    fixedDt: 1 / 60,
  };

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function mulberry32(seed) {
    let value = seed >>> 0;
    return function () {
      value += 0x6d2b79f5;
      let mixed = value;
      mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
      mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
      return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
    };
  }

  function createSimulation(options = {}) {
    const settings = { ...DEFAULT_SETTINGS, ...(options.settings || {}) };
    const seed = Number.isFinite(options.seed) ? options.seed : 1;
    return {
      settings,
      seed,
      rng: mulberry32(seed),
      tick: 0,
      running: false,
      inputDirection: 0,
      appliedInputSeq: 0,
      appliedEventSeq: 0,
      appliedInputIds: new Set(),
      appliedEventIds: new Set(),
      score: { left: 0, right: 0 },
      left: { x: 24, y: (settings.height - settings.paddleHeight) / 2, speed: 320, vy: 0 },
      right: {
        x: settings.width - 34,
        y: (settings.height - settings.paddleHeight) / 2,
        speed: 360,
        vy: 0,
      },
      ball: {
        x: settings.width / 2,
        y: settings.height / 2,
        vx: 0,
        vy: 0,
        speed: 275,
      },
    };
  }

  function resetSimulation(sim, seed) {
    const fresh = createSimulation({ seed, settings: sim.settings });
    Object.assign(sim, fresh);
  }

  function resetBall(sim, direction) {
    const angle = sim.rng() * 0.7 - 0.35;
    sim.ball.x = sim.settings.width / 2;
    sim.ball.y = sim.settings.height / 2;
    sim.ball.vx = Math.cos(angle) * sim.ball.speed * direction;
    sim.ball.vy = Math.sin(angle) * sim.ball.speed;
  }

  function start(sim) {
    if (!sim.running) {
      sim.running = true;
      resetBall(sim, sim.rng() < 0.5 ? -1 : 1);
    }
  }

  function applyTimelineEvents(sim, events) {
    const ordered = [...events].sort((a, b) => (a.tick - b.tick) || ((a.seq || 0) - (b.seq || 0)));
    for (const event of ordered) {
      const seq = event.seq || 0;
      const type = event.type || "input";
      const eventId = seq > 0 ? `seq:${seq}` : `local:${event.localId || `${type}:${event.tick}`}`;
      if (sim.appliedEventIds.has(eventId) || event.tick > sim.tick) {
        continue;
      }
      if (type === "start") {
        start(sim);
      } else if (type === "input") {
        sim.inputDirection = clamp(Number(event.direction) || 0, -1, 1);
        sim.appliedInputIds.add(eventId);
      }
      sim.appliedEventIds.add(eventId);
      if (seq > 0) {
        sim.appliedEventSeq = Math.max(sim.appliedEventSeq, seq);
        if (type === "input") {
          sim.appliedInputSeq = Math.max(sim.appliedInputSeq, seq);
        }
      }
    }
  }

  function updatePaddles(sim, dt) {
    const settings = sim.settings;
    const target = sim.ball.y - settings.paddleHeight / 2;
    const leftStartY = sim.left.y;
    const leftDelta = clamp(target - sim.left.y, -sim.left.speed * dt, sim.left.speed * dt);
    sim.left.y = clamp(sim.left.y + leftDelta, 0, settings.height - settings.paddleHeight);
    sim.left.vy = dt > 0 ? (sim.left.y - leftStartY) / dt : 0;

    const rightStartY = sim.right.y;
    sim.right.y = clamp(
      sim.right.y + sim.inputDirection * sim.right.speed * dt,
      0,
      settings.height - settings.paddleHeight
    );
    sim.right.vy = dt > 0 ? (sim.right.y - rightStartY) / dt : 0;
  }

  function paddleHit(sim, paddle) {
    const ball = sim.ball;
    const settings = sim.settings;
    return (
      ball.x < paddle.x + settings.paddleWidth &&
      ball.x + settings.ballSize > paddle.x &&
      ball.y < paddle.y + settings.paddleHeight &&
      ball.y + settings.ballSize > paddle.y
    );
  }

  function bounceFromPaddle(sim, paddle, direction) {
    const settings = sim.settings;
    const ballCenter = sim.ball.y + settings.ballSize / 2;
    const paddleCenter = paddle.y + settings.paddleHeight / 2;
    const normalized = clamp((ballCenter - paddleCenter) / (settings.paddleHeight / 2), -1, 1);
    const incomingVy = sim.ball.vy;

    sim.ball.speed = Math.min(sim.ball.speed + 8, 520);

    const inheritedAngle = incomingVy * 0.85;
    const paddleMomentum = paddle.vy * 0.45;
    const contactInfluence = normalized * sim.ball.speed * 0.18;
    const maxVertical = sim.ball.speed * 0.88;

    sim.ball.vy = clamp(inheritedAngle + paddleMomentum + contactInfluence, -maxVertical, maxVertical);
    sim.ball.vx = direction * Math.sqrt(sim.ball.speed ** 2 - sim.ball.vy ** 2);
  }

  function updateBall(sim, dt) {
    const settings = sim.settings;
    sim.ball.x += sim.ball.vx * dt;
    sim.ball.y += sim.ball.vy * dt;

    if (sim.ball.y <= 0) {
      sim.ball.y = 0;
      sim.ball.vy = Math.abs(sim.ball.vy);
    } else if (sim.ball.y + settings.ballSize >= settings.height) {
      sim.ball.y = settings.height - settings.ballSize;
      sim.ball.vy = -Math.abs(sim.ball.vy);
    }

    if (sim.ball.vx < 0 && paddleHit(sim, sim.left)) {
      sim.ball.x = sim.left.x + settings.paddleWidth;
      bounceFromPaddle(sim, sim.left, 1);
    } else if (sim.ball.vx > 0 && paddleHit(sim, sim.right)) {
      sim.ball.x = sim.right.x - settings.ballSize;
      bounceFromPaddle(sim, sim.right, -1);
    }

    if (sim.ball.x + settings.ballSize < 0) {
      sim.score.right += 1;
      sim.ball.speed = 275;
      resetBall(sim, 1);
    } else if (sim.ball.x > settings.width) {
      sim.score.left += 1;
      sim.ball.speed = 275;
      resetBall(sim, -1);
    }
  }

  function step(sim, events = []) {
    applyTimelineEvents(sim, events);
    updatePaddles(sim, sim.settings.fixedDt);
    if (sim.running) {
      updateBall(sim, sim.settings.fixedDt);
    }
    sim.tick += 1;
  }

  function draw(ctx, canvasWidth, canvasHeight, sim) {
    drawState(ctx, canvasWidth, canvasHeight, {
      settings: sim.settings,
      score: sim.score,
      ball: sim.ball,
      paddles: { leftY: sim.left.y, rightY: sim.right.y },
    });
  }

  function drawState(ctx, canvasWidth, canvasHeight, state) {
    const stateSettings = state.settings || {};
    const settings = { ...DEFAULT_SETTINGS, ...stateSettings };
    const scaleX = canvasWidth / settings.width;
    const scaleY = canvasHeight / settings.height;
    const rect = (x, y, width, height) => {
      ctx.fillRect(x * scaleX, y * scaleY, width * scaleX, height * scaleY);
    };

    ctx.clearRect(0, 0, canvasWidth, canvasHeight);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    ctx.fillStyle = "#2d2d2d";
    for (let y = 0; y < settings.height; y += 24) {
      rect(settings.width / 2 - 1, y, 2, 12);
    }

    ctx.fillStyle = "#f5f5f5";
    rect(24, state.paddles.leftY, settings.paddleWidth, settings.paddleHeight);
    rect(settings.width - 34, state.paddles.rightY, settings.paddleWidth, settings.paddleHeight);
    rect(state.ball.x, state.ball.y, settings.ballSize, settings.ballSize);
  }

  function rasterizeState(state) {
    const stateSettings = state.settings || {};
    const settings = { ...DEFAULT_SETTINGS, ...stateSettings };
    const width = Math.max(1, Math.floor(settings.width));
    const height = Math.max(1, Math.floor(settings.height));
    const mask = new Uint8Array(width * height);
    const paddles = state.paddles || {};
    const ball = state.ball || {};
    const markRect = (x, y, rectWidth, rectHeight) => {
      const left = clamp(Math.floor(x), 0, width);
      const right = clamp(Math.ceil(x + rectWidth), 0, width);
      const top = clamp(Math.floor(y), 0, height);
      const bottom = clamp(Math.ceil(y + rectHeight), 0, height);
      if (left >= right || top >= bottom) return;
      for (let row = top; row < bottom; row += 1) {
        mask.fill(1, row * width + left, row * width + right);
      }
    };

    for (let y = 0; y < settings.height; y += 24) {
      markRect(settings.width / 2 - 1, y, 2, 12);
    }
    markRect(24, Number(paddles.leftY ?? (settings.height - settings.paddleHeight) / 2), settings.paddleWidth, settings.paddleHeight);
    markRect(
      settings.width - 34,
      Number(paddles.rightY ?? (settings.height - settings.paddleHeight) / 2),
      settings.paddleWidth,
      settings.paddleHeight
    );
    markRect(
      Number(ball.x ?? settings.width / 2),
      Number(ball.y ?? settings.height / 2),
      settings.ballSize,
      settings.ballSize
    );

    return { width, height, mask };
  }

  function createEventCamera() {
    return {
      width: 0,
      height: 0,
      resetToken: undefined,
      previousMask: null,
    };
  }

  function captureEventFrame(camera, state, metadata = {}) {
    const raster = rasterizeState(state);
    const resetToken = metadata.resetToken;
    const pixels = [];
    const changedBaseline =
      !camera.previousMask ||
      camera.width !== raster.width ||
      camera.height !== raster.height ||
      camera.resetToken !== resetToken;

    if (!changedBaseline) {
      for (let index = 0; index < raster.mask.length; index += 1) {
        if (raster.mask[index] !== camera.previousMask[index]) {
          pixels.push(index);
        }
      }
    }

    camera.width = raster.width;
    camera.height = raster.height;
    camera.resetToken = resetToken;
    camera.previousMask = raster.mask;

    return {
      width: raster.width,
      height: raster.height,
      tick: Number(state.tick ?? metadata.tick ?? 0),
      frameSeq: Number(metadata.frameSeq ?? state.frameSeq ?? state.tick ?? 0),
      resetToken,
      renderedAt: metadata.renderedAt,
      source: "game",
      pixels,
      count: pixels.length,
    };
  }

  function snapshot(sim) {
    return {
      tick: sim.tick,
      running: sim.running,
      settings: {
        width: sim.settings.width,
        height: sim.settings.height,
        paddleWidth: sim.settings.paddleWidth,
        paddleHeight: sim.settings.paddleHeight,
        ballSize: sim.settings.ballSize,
        fixedDt: sim.settings.fixedDt,
      },
      score: { ...sim.score },
      ball: { x: sim.ball.x, y: sim.ball.y, vx: sim.ball.vx, vy: sim.ball.vy },
      paddles: { leftY: sim.left.y, rightY: sim.right.y },
      appliedInputSeq: sim.appliedInputSeq,
      appliedEventSeq: sim.appliedEventSeq,
    };
  }

  window.PongCore = {
    DEFAULT_SETTINGS,
    createSimulation,
    resetSimulation,
    start,
    step,
    draw,
    drawState,
    rasterizeState,
    createEventCamera,
    captureEventFrame,
    snapshot,
  };
})();
