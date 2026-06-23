(function () {
  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function captureInlineState(state, frame) {
    const sourceWidth = Math.max(1, Math.floor(frame.sourceWidth));
    const sourceHeight = Math.max(1, Math.floor(frame.sourceHeight));
    const targetWidth = Math.max(1, Math.floor(frame.width));
    const targetHeight = Math.max(1, Math.floor(frame.height));
    const threshold = Number(frame.threshold ?? 24);
    const data = frame.data;
    const mask = new Uint8Array(targetWidth * targetHeight);

    for (let y = 0; y < targetHeight; y += 1) {
      const sourceY = clamp(Math.floor((y + 0.5) * sourceHeight / targetHeight), 0, sourceHeight - 1);
      for (let x = 0; x < targetWidth; x += 1) {
        const sourceX = clamp(Math.floor((x + 0.5) * sourceWidth / targetWidth), 0, sourceWidth - 1);
        const offset = (sourceY * sourceWidth + sourceX) * 4;
        const brightness = data[offset] * 0.299 + data[offset + 1] * 0.587 + data[offset + 2] * 0.114;
        mask[y * targetWidth + x] = brightness >= threshold ? 1 : 0;
      }
    }

    const changedBaseline =
      !state.previousMask ||
      state.width !== targetWidth ||
      state.height !== targetHeight ||
      state.resetToken !== frame.resetToken;
    const pixels = [];
    if (!changedBaseline) {
      for (let index = 0; index < mask.length; index += 1) {
        if (mask[index] !== state.previousMask[index]) {
          pixels.push(index);
        }
      }
    }

    state.previousMask = mask;
    state.width = targetWidth;
    state.height = targetHeight;
    state.resetToken = frame.resetToken;

    return {
      width: targetWidth,
      height: targetHeight,
      tick: Number(frame.tick || 0),
      frameSeq: Number(frame.frameSeq || 0),
      resetToken: frame.resetToken,
      renderedAt: Number(frame.renderedAt || 0),
      observedAt: Number(frame.observedAt || 0),
      source: frame.source || "event-camera",
      pixels,
      count: pixels.length,
    };
  }

  function createEventCamera(options = {}) {
    const camera = {
      width: Math.max(1, Math.floor(Number(options.width || 800))),
      height: Math.max(1, Math.floor(Number(options.height || 450))),
      threshold: Number(options.threshold ?? 24),
      latest: null,
      pending: new Map(),
      inlineState: {},
      worker: null,
      onEventCamera: typeof options.onEventCamera === "function" ? options.onEventCamera : null,
    };

    if ("Worker" in window) {
      try {
        camera.worker = new Worker(options.workerUrl || "/web/event-camera-worker.js");
        camera.worker.onmessage = (event) => {
          const message = event.data || {};
          if (message.type !== "events" || !message.eventCamera) return;
          camera.latest = message.eventCamera;
          const pending = camera.pending.get(message.eventCamera.frameSeq);
          camera.pending.delete(message.eventCamera.frameSeq);
          if (pending) pending(message.eventCamera);
          if (camera.onEventCamera) camera.onEventCamera(message.eventCamera);
        };
      } catch {
        camera.worker = null;
      }
    }

    camera.reset = () => {
      camera.latest = null;
      camera.inlineState = {};
      camera.pending.clear();
      if (camera.worker) {
        camera.worker.postMessage({ type: "reset" });
      }
    };

    camera.observeCanvas = (canvas, metadata = {}) => {
      const sourceWidth = Math.max(1, Math.floor(canvas.width || canvas.clientWidth || camera.width));
      const sourceHeight = Math.max(1, Math.floor(canvas.height || canvas.clientHeight || camera.height));
      const context = canvas.getContext("2d", { willReadFrequently: true }) || canvas.getContext("2d");
      const image = context.getImageData(0, 0, sourceWidth, sourceHeight);
      const observedAt = performance.now();
      const frame = {
        width: Math.max(1, Math.floor(Number(metadata.width || camera.width))),
        height: Math.max(1, Math.floor(Number(metadata.height || camera.height))),
        sourceWidth,
        sourceHeight,
        threshold: Number(metadata.threshold ?? camera.threshold),
        tick: Number(metadata.tick || 0),
        frameSeq: Number(metadata.frameSeq || 0),
        resetToken: metadata.resetToken,
        renderedAt: Number(metadata.renderedAt || 0),
        observedAt,
        source: metadata.source || "event-camera",
        data: image.data,
      };

      if (!camera.worker) {
        const eventCamera = captureInlineState(camera.inlineState, frame);
        camera.latest = eventCamera;
        if (camera.onEventCamera) camera.onEventCamera(eventCamera);
        return Promise.resolve(eventCamera);
      }

      return new Promise((resolve) => {
        camera.pending.set(frame.frameSeq, resolve);
        camera.worker.postMessage({ type: "frame", frame }, [frame.data.buffer]);
      });
    };

    return camera;
  }

  window.EventCamera = { create: createEventCamera };
})();
