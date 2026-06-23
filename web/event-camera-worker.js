let previousMask = null;
let previousWidth = 0;
let previousHeight = 0;
let previousResetToken;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function sampleMask(frame) {
  const sourceWidth = Math.max(1, Math.floor(frame.sourceWidth));
  const sourceHeight = Math.max(1, Math.floor(frame.sourceHeight));
  const targetWidth = Math.max(1, Math.floor(frame.width));
  const targetHeight = Math.max(1, Math.floor(frame.height));
  const data = frame.data;
  const threshold = Number(frame.threshold ?? 24);
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

  return { width: targetWidth, height: targetHeight, mask };
}

function capture(frame) {
  const raster = sampleMask(frame);
  const resetToken = frame.resetToken;
  const changedBaseline =
    !previousMask ||
    previousWidth !== raster.width ||
    previousHeight !== raster.height ||
    previousResetToken !== resetToken;
  const pixels = [];

  if (!changedBaseline) {
    for (let index = 0; index < raster.mask.length; index += 1) {
      if (raster.mask[index] !== previousMask[index]) {
        pixels.push(index);
      }
    }
  }

  previousMask = raster.mask;
  previousWidth = raster.width;
  previousHeight = raster.height;
  previousResetToken = resetToken;

  return {
    width: raster.width,
    height: raster.height,
    tick: Number(frame.tick || 0),
    frameSeq: Number(frame.frameSeq || 0),
    resetToken,
    renderedAt: Number(frame.renderedAt || 0),
    observedAt: Number(frame.observedAt || 0),
    source: frame.source || "event-camera",
    pixels,
    count: pixels.length,
  };
}

self.onmessage = (event) => {
  const message = event.data || {};
  if (message.type === "reset") {
    previousMask = null;
    previousWidth = 0;
    previousHeight = 0;
    previousResetToken = undefined;
    return;
  }
  if (message.type !== "frame") return;
  self.postMessage({ type: "events", eventCamera: capture(message.frame) });
};
