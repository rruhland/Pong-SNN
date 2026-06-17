import json
import math
import random
import time
from pathlib import Path


OUTPUTS = (
    {"name": "move up", "direction": -1},
    {"name": "move down", "direction": 1},
    {"name": "stay put", "direction": 0},
)


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def _grid_index(x, y, width):
    return y * width + x


def _neighborhood(cx, cy, width, height, radius):
    coords = []
    for y in range(max(0, cy - radius), min(height, cy + radius + 1)):
        for x in range(max(0, cx - radius), min(width, cx + radius + 1)):
            coords.append(_grid_index(x, y, width))
    return coords


class PongSNN:
    """Sparse, backend-owned SNN scaffold for Pong event-camera input.

    The implementation is intentionally small and dependency-free. It keeps a
    GPU-ready device descriptor and tensor-friendly dense weight arrays, while
    the first pass runs as sparse Python updates so the API works without CUDA.
    """

    version = 1

    def __init__(self, width=800, height=450, seed=None, save_dir=None):
        self.width = int(width)
        self.height = int(height)
        self.seed = int(seed if seed is not None else time.time() * 1000) & 0xFFFFFFFF
        self.rng = random.Random(self.seed)
        self.save_dir = Path(save_dir or "network_saves")
        self.training = False
        self.paused = True
        self.tick = 0
        self.train_steps = 0
        self.last_command = 0
        self.last_command_tick = -999
        self.last_save = None
        self.loaded_from = None
        self.device = self._detect_device()

        self.h1_cell = 10
        self.h1_w = math.ceil(self.width / self.h1_cell)
        self.h1_h = math.ceil(self.height / self.h1_cell)
        self.h2_w = math.ceil(self.h1_w / 2)
        self.h2_h = math.ceil(self.h1_h / 2)
        self.h3_w = math.ceil(self.h2_w / 2)
        self.h3_h = math.ceil(self.h2_h / 2)

        self.h1_size = self.h1_w * self.h1_h
        self.h2_size = self.h2_w * self.h2_h
        self.h3_size = self.h3_w * self.h3_h
        self.input_size = self.width * self.height

        self.input_weights = [self._rand_weight() for _ in range(self.input_size)]
        self.h1_h2_pre = self._build_projection(self.h1_w, self.h1_h, self.h2_w, self.h2_h, radius=1)
        self.h2_h3_pre = self._build_projection(self.h2_w, self.h2_h, self.h3_w, self.h3_h, radius=2)
        self.h1_h2_weights = self._build_weights(self.h1_h2_pre)
        self.h2_h3_weights = self._build_weights(self.h2_h3_pre)
        self.h3_out_pre = [list(range(self.h3_size)) for _ in OUTPUTS]
        self.h3_out_weights = self._build_output_weights()

        self.activity = self._empty_activity()

    def _detect_device(self):
        info = {"backend": "python", "requested": "auto", "active": "cpu", "cudaAvailable": False}
        try:
            import torch  # noqa: F401

            info["torchAvailable"] = True
            info["cudaAvailable"] = bool(torch.cuda.is_available())
            if info["cudaAvailable"]:
                info["active"] = "cuda"
                info["cudaName"] = torch.cuda.get_device_name(0)
        except Exception as exc:
            info["torchAvailable"] = False
            info["note"] = f"torch unavailable: {exc.__class__.__name__}"
        return info

    def _rand_weight(self):
        return 0.34 + self.rng.random() * 0.32

    def _build_projection(self, pre_w, pre_h, post_w, post_h, radius):
        projection = []
        for py in range(post_h):
            for px in range(post_w):
                cx = min(pre_w - 1, round((px + 0.5) * pre_w / post_w - 0.5))
                cy = min(pre_h - 1, round((py + 0.5) * pre_h / post_h - 0.5))
                projection.append(_neighborhood(cx, cy, pre_w, pre_h, radius))
        return projection

    def _build_weights(self, projection):
        return [[self._rand_weight() for _ in pres] for pres in projection]

    def _build_output_weights(self):
        weights = []
        for output_index, output in enumerate(OUTPUTS):
            output_weights = []
            for pre in self.h3_out_pre[output_index]:
                y = pre // self.h3_w
                vertical = y / max(1, self.h3_h - 1)
                if output["direction"] == -1:
                    bias = 0.14 * (1.0 - vertical)
                elif output["direction"] == 1:
                    bias = 0.14 * vertical
                else:
                    bias = 0.04
                output_weights.append(_clamp(self._rand_weight() + bias, 0.05, 0.95))
            weights.append(output_weights)
        return weights

    def _empty_activity(self):
        return {
            "tick": 0,
            "eventCount": 0,
            "spikes": {"input": 0, "hidden1": 0, "hidden2": 0, "hidden3": 0, "output": 0},
            "outputDrive": [0.0, 0.0, 0.0],
            "outputBars": [0.0, 0.0, 0.0],
            "winner": "stay put",
            "direction": 0,
            "activeInputSample": [],
            "activeHidden1": [],
            "activeHidden2": [],
            "activeHidden3": [],
            "stdp": {"potentiated": 0, "depressed": 0},
        }

    def architecture(self):
        input_connections = self.input_size
        h1_h2_connections = sum(len(pres) for pres in self.h1_h2_pre)
        h2_h3_connections = sum(len(pres) for pres in self.h2_h3_pre)
        output_connections = sum(len(pres) for pres in self.h3_out_pre)
        total_possible_first = self.input_size * self.h1_size
        total_connections = input_connections + h1_h2_connections + h2_h3_connections + output_connections
        return {
            "version": self.version,
            "device": self.device,
            "input": {
                "name": "event camera pixels",
                "width": self.width,
                "height": self.height,
                "neurons": self.input_size,
                "activeRule": "only changed event-camera pixels spike",
            },
            "layers": [
                {
                    "name": "hidden1",
                    "width": self.h1_w,
                    "height": self.h1_h,
                    "neurons": self.h1_size,
                    "receptiveField": "10x10 input-pixel chunks",
                    "connections": input_connections,
                },
                {
                    "name": "hidden2",
                    "width": self.h2_w,
                    "height": self.h2_h,
                    "neurons": self.h2_size,
                    "receptiveField": "3x3 hidden1 neighborhoods",
                    "connections": h1_h2_connections,
                },
                {
                    "name": "hidden3",
                    "width": self.h3_w,
                    "height": self.h3_h,
                    "neurons": self.h3_size,
                    "receptiveField": "5x5 hidden2 neighborhoods",
                    "connections": h2_h3_connections,
                },
                {
                    "name": "output",
                    "width": 3,
                    "height": 1,
                    "neurons": 3,
                    "receptiveField": "sparse readout from hidden3",
                    "connections": output_connections,
                    "labels": [item["name"] for item in OUTPUTS],
                },
            ],
            "sparsity": {
                "firstLayerConnections": input_connections,
                "firstLayerAllToAllWouldBe": total_possible_first,
                "firstLayerDensity": input_connections / total_possible_first,
                "totalConnections": total_connections,
            },
            "stdp": {
                "aPlus": 0.012,
                "aMinus": 0.004,
                "rule": "active pre plus active post potentiates; active pre without post depresses slightly",
            },
        }

    def _input_to_h1(self, pixels):
        drives = [0.0] * self.h1_size
        active_pixels = []
        max_index = self.input_size
        for raw_pixel in pixels:
            try:
                pixel = int(raw_pixel)
            except (TypeError, ValueError):
                continue
            if pixel < 0 or pixel >= max_index:
                continue
            active_pixels.append(pixel)
            y = pixel // self.width
            x = pixel - y * self.width
            hx = min(self.h1_w - 1, x // self.h1_cell)
            hy = min(self.h1_h - 1, y // self.h1_cell)
            drives[_grid_index(hx, hy, self.h1_w)] += self.input_weights[pixel]
        return active_pixels, drives

    def _spikes_from_drives(self, drives, threshold, max_spikes):
        indexed = [(index, drive) for index, drive in enumerate(drives) if drive > 0.0]
        if not indexed:
            return []
        spikes = [index for index, drive in indexed if drive >= threshold]
        if not spikes:
            strongest = max(indexed, key=lambda item: item[1])
            spikes = [strongest[0]]
        if len(spikes) > max_spikes:
            strongest = sorted(((index, drives[index]) for index in spikes), key=lambda item: item[1], reverse=True)
            spikes = [index for index, _ in strongest[:max_spikes]]
        return spikes

    def _project(self, pre_spikes, projection, weights):
        active = set(pre_spikes)
        drives = []
        for pres, post_weights in zip(projection, weights):
            drive = 0.0
            for offset, pre in enumerate(pres):
                if pre in active:
                    drive += post_weights[offset]
            drives.append(drive)
        return drives

    def _output(self, h3_spikes):
        active = set(h3_spikes)
        drives = []
        for pres, weights in zip(self.h3_out_pre, self.h3_out_weights):
            drive = 0.0
            for offset, pre in enumerate(pres):
                if pre in active:
                    drive += weights[offset]
            drives.append(drive)
        if max(drives, default=0.0) <= 0.0:
            winner_index = 2
        else:
            winner_index = max(range(len(drives)), key=lambda index: drives[index])
        max_drive = max(1.0, max(drives, default=0.0))
        return drives, [drive / max_drive for drive in drives], winner_index

    def _stdp_input(self, active_pixels, h1_spikes):
        h1_active = set(h1_spikes)
        changed = {"potentiated": 0, "depressed": 0}
        for pixel in active_pixels:
            y = pixel // self.width
            x = pixel - y * self.width
            hx = min(self.h1_w - 1, x // self.h1_cell)
            hy = min(self.h1_h - 1, y // self.h1_cell)
            post = _grid_index(hx, hy, self.h1_w)
            weight = self.input_weights[pixel]
            if post in h1_active:
                self.input_weights[pixel] = _clamp(weight + 0.012 * (1.0 - weight))
                changed["potentiated"] += 1
            else:
                self.input_weights[pixel] = _clamp(weight - 0.004 * weight)
                changed["depressed"] += 1
        return changed

    def _stdp_projection(self, pre_spikes, post_spikes, projection, weights):
        pre_active = set(pre_spikes)
        post_active = set(post_spikes)
        changed = {"potentiated": 0, "depressed": 0}
        if not pre_active:
            return changed
        for post, (pres, post_weights) in enumerate(zip(projection, weights)):
            active_post = post in post_active
            for offset, pre in enumerate(pres):
                if pre not in pre_active:
                    continue
                weight = post_weights[offset]
                if active_post:
                    post_weights[offset] = _clamp(weight + 0.012 * (1.0 - weight))
                    changed["potentiated"] += 1
                else:
                    post_weights[offset] = _clamp(weight - 0.004 * weight)
                    changed["depressed"] += 1
        return changed

    def _stdp_output(self, h3_spikes, winner_index):
        active = set(h3_spikes)
        changed = {"potentiated": 0, "depressed": 0}
        if not active:
            return changed
        for output_index, (pres, weights) in enumerate(zip(self.h3_out_pre, self.h3_out_weights)):
            for offset, pre in enumerate(pres):
                if pre not in active:
                    continue
                weight = weights[offset]
                if output_index == winner_index:
                    weights[offset] = _clamp(weight + 0.010 * (1.0 - weight))
                    changed["potentiated"] += 1
                else:
                    weights[offset] = _clamp(weight - 0.003 * weight)
                    changed["depressed"] += 1
        return changed

    def step(self, event_camera, tick):
        if not event_camera:
            return None
        pixels = event_camera.get("pixels") if isinstance(event_camera, dict) else []
        if not isinstance(pixels, list):
            pixels = []

        self.tick = int(tick or self.tick)
        active_pixels, h1_drive = self._input_to_h1(pixels)
        h1_spikes = self._spikes_from_drives(h1_drive, threshold=0.58, max_spikes=256)
        h2_drive = self._project(h1_spikes, self.h1_h2_pre, self.h1_h2_weights)
        h2_spikes = self._spikes_from_drives(h2_drive, threshold=0.82, max_spikes=96)
        h3_drive = self._project(h2_spikes, self.h2_h3_pre, self.h2_h3_weights)
        h3_spikes = self._spikes_from_drives(h3_drive, threshold=0.92, max_spikes=36)
        output_drive, output_bars, winner_index = self._output(h3_spikes)
        winner = OUTPUTS[winner_index]

        changed = {"potentiated": 0, "depressed": 0}
        if self.training and not self.paused:
            for update in (
                self._stdp_input(active_pixels, h1_spikes),
                self._stdp_projection(h1_spikes, h2_spikes, self.h1_h2_pre, self.h1_h2_weights),
                self._stdp_projection(h2_spikes, h3_spikes, self.h2_h3_pre, self.h2_h3_weights),
                self._stdp_output(h3_spikes, winner_index),
            ):
                changed["potentiated"] += update["potentiated"]
                changed["depressed"] += update["depressed"]
            self.train_steps += 1

        direction = int(winner["direction"])
        self.activity = {
            "tick": self.tick,
            "eventCount": len(active_pixels),
            "spikes": {
                "input": len(active_pixels),
                "hidden1": len(h1_spikes),
                "hidden2": len(h2_spikes),
                "hidden3": len(h3_spikes),
                "output": 1 if max(output_drive, default=0.0) > 0.0 else 0,
            },
            "outputDrive": [round(value, 4) for value in output_drive],
            "outputBars": [round(value, 4) for value in output_bars],
            "winner": winner["name"],
            "direction": direction,
            "activeInputSample": active_pixels[:900],
            "activeHidden1": h1_spikes[:256],
            "activeHidden2": h2_spikes[:192],
            "activeHidden3": h3_spikes[:128],
            "stdp": changed,
        }
        return direction

    def should_emit_command(self, direction, tick):
        if direction != self.last_command or tick - self.last_command_tick >= 8:
            self.last_command = direction
            self.last_command_tick = tick
            return True
        return False

    def status(self):
        return {
            "training": self.training,
            "paused": self.paused,
            "tick": self.tick,
            "trainSteps": self.train_steps,
            "lastCommand": self.last_command,
            "lastSave": self.last_save,
            "loadedFrom": self.loaded_from,
            "architecture": self.architecture(),
            "activity": self.activity,
            "outputs": OUTPUTS,
        }

    def start(self):
        self.training = True
        self.paused = False

    def pause(self):
        self.paused = True

    def reset(self, reset_weights=True):
        seed = self.seed if not reset_weights else int(time.time() * 1000) & 0xFFFFFFFF
        save_dir = self.save_dir
        self.__init__(self.width, self.height, seed=seed, save_dir=save_dir)

    def to_dict(self):
        return {
            "version": self.version,
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "trainSteps": self.train_steps,
            "inputWeights": self.input_weights,
            "h1h2Weights": self.h1_h2_weights,
            "h2h3Weights": self.h2_h3_weights,
            "h3outWeights": self.h3_out_weights,
            "savedAt": time.time(),
        }

    def load_dict(self, payload):
        if int(payload.get("width", self.width)) != self.width or int(payload.get("height", self.height)) != self.height:
            raise ValueError("saved network dimensions do not match this Pong event camera")
        self.seed = int(payload.get("seed", self.seed))
        self.train_steps = int(payload.get("trainSteps", 0))
        self.input_weights = [float(value) for value in payload["inputWeights"]]
        self.h1_h2_weights = [[float(value) for value in row] for row in payload["h1h2Weights"]]
        self.h2_h3_weights = [[float(value) for value in row] for row in payload["h2h3Weights"]]
        self.h3_out_weights = [[float(value) for value in row] for row in payload["h3outWeights"]]
        self.activity = self._empty_activity()

    def save(self, name=None):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in (name or "network")).strip("-")
        if not safe_name:
            safe_name = "network"
        path = self.save_dir / f"{safe_name}-{int(time.time())}.json"
        path.write_text(json.dumps(self.to_dict(), separators=(",", ":")), encoding="utf-8")
        self.last_save = path.name
        return path.name

    def load(self, name):
        path = (self.save_dir / name).resolve()
        root = self.save_dir.resolve()
        if root != path.parent:
            raise ValueError("load name must refer to a saved network file")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.load_dict(payload)
        self.loaded_from = path.name
        return path.name

    def list_saves(self):
        if not self.save_dir.exists():
            return []
        saves = []
        for path in sorted(self.save_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            saves.append({"name": path.name, "bytes": path.stat().st_size, "modified": path.stat().st_mtime})
        return saves
