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

REWARD_PARAMS = {
    "total": {"min": -3.5, "max": 4.0},
    "movement": {"value": -0.04},
    "survival": {"scale": 0.02, "max": 0.048, "tickRate": 60.0},
    "opponentScore": {"value": -5.0},
    "rightScore": {"value": 10},
}

ELIGIBILITY_TRACES = (
    {"name": "fast", "decay": 0.92, "scale": 1.0, "horizonMs": 500},
    {"name": "medium", "decay": 0.98, "scale": 0.35, "horizonMs": 2000},
    {"name": "slow", "decay": 0.99, "scale": 0.08, "horizonMs": 5000},
)


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def _grid_index(x, y, width):
    return y * width + x


def _sigmoid(value):
    if value < -30:
        return 0.0
    if value > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))


class RewardFunction:
    """Named reward components with central, inspectable tuning parameters."""

    def __init__(self, params=None):
        self.params = self._merge_params(REWARD_PARAMS, params or {})
        self.episode_ticks = 0
        self.last_reset_token = None

    def _merge_params(self, defaults, overrides):
        merged = {}
        for key, value in defaults.items():
            base = dict(value)
            base.update(overrides.get(key, {}))
            merged[key] = base
        for key, value in overrides.items():
            if key not in merged:
                merged[key] = dict(value)
        return merged

    def reset(self):
        self.episode_ticks = 0
        self.last_reset_token = None

    def config(self):
        return {key: dict(value) for key, value in self.params.items()}

    def empty_components(self):
        return {key: 0.0 for key in self.params if key != "total"}

    def total(self, components):
        raw = sum(float(value) for value in components.values())
        params = self.params["total"]
        return _clamp(raw, params["min"], params["max"])

    def components(self, current, previous, winner_index):
        components = self.empty_components()
        direction = int(OUTPUTS[self._valid_winner_index(winner_index)]["direction"])
        if direction != 0:
            components["movement"] = self.params["movement"]["value"]

        last_event = None
        if current is None:
            return components, last_event, self.metrics()

        reset_token = current.get("resetToken")
        reset_seen = self.last_reset_token is None or reset_token != self.last_reset_token
        if reset_seen:
            self.episode_ticks = 0
            self.last_reset_token = reset_token

        if previous is None or reset_seen:
            return components, last_event, self.metrics()

        elapsed_ticks = max(1, int(current["tick"] - previous["tick"]))
        self.episode_ticks += elapsed_ticks
        seconds_alive = self.episode_ticks / max(1.0, float(self.params["survival"]["tickRate"]))
        survival = math.log1p(seconds_alive) * self.params["survival"]["scale"]
        components["survival"] = min(self.params["survival"]["max"], survival)

        if current["leftScore"] > previous["leftScore"]:
            components["opponentScore"] = self.params["opponentScore"]["value"]
            last_event = "opponent-score"
        if current["rightScore"] > previous["rightScore"]:
            components["rightScore"] = self.params["rightScore"]["value"]
            last_event = "right-score"

        metrics = self.metrics()
        if components["opponentScore"] != 0.0 or components["rightScore"] != 0.0:
            self.episode_ticks = 0

        return components, last_event, metrics

    def metrics(self):
        seconds_alive = self.episode_ticks / max(1.0, float(self.params["survival"]["tickRate"]))
        return {
            "episodeTicks": int(self.episode_ticks),
            "episodeSeconds": round(seconds_alive, 3),
        }

    def _valid_winner_index(self, winner_index):
        try:
            winner_index = int(winner_index)
        except (TypeError, ValueError):
            return len(OUTPUTS) - 1
        if winner_index < 0 or winner_index >= len(OUTPUTS):
            return len(OUTPUTS) - 1
        return winner_index


class SynapseGroup:
    def __init__(self, name, pre_size, post_size, edges, traces):
        self.name = name
        self.pre_size = int(pre_size)
        self.post_size = int(post_size)
        self.pre = [int(edge[0]) for edge in edges]
        self.post = [int(edge[1]) for edge in edges]
        self.weight = [float(edge[2]) for edge in edges]
        self.low = [float(edge[3]) for edge in edges]
        self.high = [float(edge[4]) for edge in edges]
        self.traces = tuple(dict(trace) for trace in traces)
        self.eligibility = {
            trace["name"]: [0.0 for _ in self.weight]
            for trace in self.traces
        }
        self.active_edges = set()
        self.by_pre = [[] for _ in range(self.pre_size)]
        for index, pre in enumerate(self.pre):
            if 0 <= pre < self.pre_size:
                self.by_pre[pre].append(index)

    def __len__(self):
        return len(self.weight)

    def decay(self):
        if not self.active_edges:
            return
        still_active = set()
        for trace in self.traces:
            values = self.eligibility[trace["name"]]
            decay = trace["decay"]
            for index in self.active_edges:
                value = values[index]
                if value == 0.0:
                    continue
                decayed = value * decay
                values[index] = 0.0 if abs(decayed) < 0.00001 else decayed
                if values[index] != 0.0:
                    still_active.add(index)
        self.active_edges = still_active

    def add_eligibility(self, edge_index, delta):
        is_active = False
        for trace in self.traces:
            values = self.eligibility[trace["name"]]
            values[edge_index] = max(-1.0, min(1.0, values[edge_index] + delta))
            if abs(values[edge_index]) > 0.00001:
                is_active = True
        if is_active:
            self.active_edges.add(edge_index)
        else:
            self.active_edges.discard(edge_index)

    def effective(self, edge_index):
        value = 0.0
        for trace in self.traces:
            value += trace["scale"] * self.eligibility[trace["name"]][edge_index]
        return value

    def effective_values(self):
        return [self.effective(index) for index in self.active_edges]

    def apply_modulator(self, learning_rate, modulator, stats):
        if modulator == 0.0:
            return
        for index in list(self.active_edges):
            before = self.weight[index]
            trace = self.effective(index)
            if abs(trace) <= 0.00001:
                continue
            delta = learning_rate * modulator * trace
            after = max(self.low[index], min(self.high[index], before + delta))
            if after == before:
                continue
            self.weight[index] = after
            stats["weightUpdates"] += 1
            stats["meanAbsDelta"] += abs(after - before)
            if after == self.low[index] or after == self.high[index]:
                stats["clamped"] += 1
            if after > before:
                stats["potentiated"] += 1
            else:
                stats["depressed"] += 1

    def apply_post_errors(self, learning_rate, post_errors, stats):
        if not post_errors:
            return
        for index in list(self.active_edges):
            post = self.post[index]
            error = post_errors.get(post)
            if error is None:
                continue
            trace = self.effective(index)
            if abs(trace) <= 0.00001:
                continue
            before = self.weight[index]
            delta = learning_rate * error * trace
            after = max(self.low[index], min(self.high[index], before + delta))
            if after == before:
                continue
            self.weight[index] = after
            stats["predictionWeightUpdates"] += 1
            stats["weightUpdates"] += 1
            stats["meanAbsDelta"] += abs(after - before)
            if after == self.low[index] or after == self.high[index]:
                stats["clamped"] += 1
            if after > before:
                stats["potentiated"] += 1
            else:
                stats["depressed"] += 1

    def stats(self):
        active = 0
        positive = 0
        negative = 0
        total_abs = 0.0
        max_abs = 0.0
        for index in self.active_edges:
            value = self.effective(index)
            magnitude = abs(value)
            if magnitude <= 0.00001:
                continue
            active += 1
            total_abs += magnitude
            max_abs = max(max_abs, magnitude)
            if value > 0:
                positive += 1
            else:
                negative += 1
        return {
            "name": self.name,
            "connections": len(self.weight),
            "active": active,
            "positive": positive,
            "negative": negative,
            "meanAbs": round(total_abs / max(1, len(self.weight)), 6),
            "maxAbs": round(max_abs, 6),
            "traces": {
                trace["name"]: self._trace_stats(trace["name"])
                for trace in self.traces
            },
        }

    def _trace_stats(self, name):
        values = self.eligibility[name]
        active = 0
        positive = 0
        negative = 0
        total_abs = 0.0
        max_abs = 0.0
        for index in self.active_edges:
            value = values[index]
            magnitude = abs(value)
            if magnitude <= 0.00001:
                continue
            active += 1
            total_abs += magnitude
            max_abs = max(max_abs, magnitude)
            if value > 0:
                positive += 1
            else:
                negative += 1
        return {
            "name": name,
            "connections": len(values),
            "active": active,
            "positive": positive,
            "negative": negative,
            "meanAbs": round(total_abs / max(1, len(values)), 6),
            "maxAbs": round(max_abs, 6),
        }

    def to_dict(self):
        return {"weights": self.weight}

    def load_weights(self, payload):
        weights = payload.get("weights", [])
        if len(weights) != len(self.weight):
            raise ValueError(f"saved {self.name} weights do not match this architecture")
        self.weight = [max(self.low[index], min(self.high[index], float(value))) for index, value in enumerate(weights)]


class PongSNN:
    """Compressed event-camera SNN with a recurrent predictive hidden cloud."""

    version = 2

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

        self.input_w = 64
        self.input_h = 36
        self.input_size = self.input_w * self.input_h
        self.motor_size = len(OUTPUTS)
        self.hidden_size = 5000
        self.hidden_w = 100
        self.hidden_h = 50
        self.excitatory_count = int(self.hidden_size * 0.8)
        self.inhibitory_count = self.hidden_size - self.excitatory_count

        self.learning_rate = 0.0045
        self.prediction_learning_rate = 0.018
        self.predictive_modulation_scale = 0.18
        self.status_summary_interval = 10
        self.eligibility_plus = 0.14
        self.eligibility_minus = 0.03
        self.eligibility_traces = tuple(dict(trace) for trace in ELIGIBILITY_TRACES)
        self.eligibility_decay = self.eligibility_traces[0]["decay"]
        self.motor_trace_decay = 0.86

        self.hidden_potential = [0.0 for _ in range(self.hidden_size)]
        self.last_hidden_spikes = []
        self.previous_action_traces = [0.0, 0.0, 1.0]
        self.last_prediction = {}
        self.last_prediction_edges = 0
        self.last_pong_snapshot = None
        self.reward_function = RewardFunction()
        self.recent_right_scores = 0
        self.recent_opponent_scores = 0
        self.reward_state = self._empty_reward_state()
        self.learning_stats = self._empty_learning_stats()
        self.prediction_state = self._empty_prediction_state()

        self.neuron_is_excitatory = self._build_neuron_types()
        self.input_hidden = SynapseGroup(
            "inputHidden",
            self.input_size,
            self.hidden_size,
            self._build_input_hidden_edges(),
            self.eligibility_traces,
        )
        self.motor_hidden = SynapseGroup(
            "motorHidden",
            self.motor_size,
            self.hidden_size,
            self._build_motor_hidden_edges(),
            self.eligibility_traces,
        )
        self.recurrent = SynapseGroup(
            "recurrentCloud",
            self.hidden_size,
            self.hidden_size,
            self._build_recurrent_edges(),
            self.eligibility_traces,
        )
        self.hidden_output = SynapseGroup(
            "hiddenOutput",
            self.hidden_size,
            self.motor_size,
            self._build_output_edges(),
            self.eligibility_traces,
        )
        self.hidden_prediction = SynapseGroup(
            "predictionHead",
            self.hidden_size,
            self.input_size,
            self._build_prediction_edges(),
            self.eligibility_traces,
        )
        self.eligibility_stats = self._empty_eligibility_stats()
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

    def _build_neuron_types(self):
        types = [True] * self.excitatory_count + [False] * self.inhibitory_count
        self.rng.shuffle(types)
        return types

    def _hidden_xy(self, index):
        return index % self.hidden_w, index // self.hidden_w

    def _hidden_index(self, x, y):
        return _grid_index(x % self.hidden_w, y % self.hidden_h, self.hidden_w)

    def _input_to_hidden_center(self, input_index):
        x = input_index % self.input_w
        y = input_index // self.input_w
        hx = int((x + 0.5) * self.hidden_w / self.input_w)
        hy = int((y + 0.5) * self.hidden_h / self.input_h)
        return min(self.hidden_w - 1, hx), min(self.hidden_h - 1, hy)

    def _signed_bounds(self, pre):
        return (0.0, 1.0) if self.neuron_is_excitatory[pre] else (-1.0, 0.0)

    def _signed_weight(self, pre, low_abs, high_abs):
        magnitude = low_abs + self.rng.random() * (high_abs - low_abs)
        return magnitude if self.neuron_is_excitatory[pre] else -magnitude

    def _build_input_hidden_edges(self):
        edges = []
        offsets = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
        for pre in range(self.input_size):
            cx, cy = self._input_to_hidden_center(pre)
            for dx, dy in offsets:
                hx = max(0, min(self.hidden_w - 1, cx + dx))
                hy = max(0, min(self.hidden_h - 1, cy + dy))
                post = self._hidden_index(hx, hy)
                weight = 0.34 + self.rng.random() * 0.38
                edges.append((pre, post, weight, 0.0, 1.0))
        return edges

    def _build_motor_hidden_edges(self):
        edges = []
        per_action = 420
        for pre in range(self.motor_size):
            target_y = (pre + 0.5) * self.hidden_h / self.motor_size
            for _ in range(per_action):
                hx = self.rng.randrange(self.hidden_w)
                hy = int(max(0, min(self.hidden_h - 1, self.rng.gauss(target_y, self.hidden_h * 0.18))))
                post = self._hidden_index(hx, hy)
                weight = 0.08 + self.rng.random() * 0.26
                edges.append((pre, post, weight, 0.0, 1.0))
        return edges

    def _build_recurrent_edges(self):
        edges = []
        nearby = 18
        long_range = 6
        for pre in range(self.hidden_size):
            px, py = self._hidden_xy(pre)
            low, high = self._signed_bounds(pre)
            for _ in range(nearby):
                dx = int(round(self.rng.gauss(0, 3.2)))
                dy = int(round(self.rng.gauss(0, 2.1)))
                post = self._hidden_index(px + dx, py + dy)
                if post == pre:
                    post = self._hidden_index(px + 1, py)
                weight = self._signed_weight(pre, 0.22, 0.72)
                edges.append((pre, post, weight, low, high))
            for _ in range(long_range):
                post = self.rng.randrange(self.hidden_size)
                if post == pre:
                    post = (post + 1) % self.hidden_size
                weight = self._signed_weight(pre, 0.025, 0.16)
                edges.append((pre, post, weight, low, high))
        return edges

    def _build_output_edges(self):
        edges = []
        for pre in range(self.hidden_size):
            _, hy = self._hidden_xy(pre)
            vertical = hy / max(1, self.hidden_h - 1)
            low, high = self._signed_bounds(pre)
            for post, output in enumerate(OUTPUTS):
                if output["direction"] == -1:
                    bias = 0.055 * (1.0 - vertical)
                elif output["direction"] == 1:
                    bias = 0.055 * vertical
                else:
                    bias = 0.03
                magnitude = 0.01 + self.rng.random() * 0.08 + bias
                weight = magnitude if self.neuron_is_excitatory[pre] else -magnitude
                edges.append((pre, post, weight, low, high))
        return edges

    def _build_prediction_edges(self):
        edges = []
        offsets = ((0, 0), (-1, 0), (1, 0), (0, -1))
        for pre in range(self.hidden_size):
            hx, hy = self._hidden_xy(pre)
            gx = int((hx + 0.5) * self.input_w / self.hidden_w)
            gy = int((hy + 0.5) * self.input_h / self.hidden_h)
            low, high = self._signed_bounds(pre)
            for dx, dy in offsets:
                px = max(0, min(self.input_w - 1, gx + dx))
                py = max(0, min(self.input_h - 1, gy + dy))
                post = _grid_index(px, py, self.input_w)
                weight = self._signed_weight(pre, 0.04, 0.22)
                edges.append((pre, post, weight, low, high))
        return edges

    def _empty_activity(self):
        return {
            "tick": 0,
            "eventCount": 0,
            "compressedEventCount": 0,
            "spikes": {"input": 0, "hidden1": 0, "hidden2": 0, "hidden3": 0, "output": 0},
            "outputDrive": [0.0, 0.0, 0.0],
            "outputBars": [0.0, 0.0, 0.0],
            "winner": "stay put",
            "direction": 0,
            "activeInputSample": [],
            "activeHidden1": [],
            "activeHidden2": [],
            "activeHidden3": [],
            "motorTraces": list(self.previous_action_traces),
            "prediction": self._empty_prediction_state(),
            "stdp": {
                "potentiated": 0,
                "depressed": 0,
                "eligibilityIncreased": 0,
                "eligibilityDecreased": 0,
            },
        }

    def _empty_reward_state(self):
        return {
            "value": 0.0,
            "components": self.reward_function.empty_components(),
            "metrics": self.reward_function.metrics(),
            "lastEvent": None,
            "recentRightScores": self.recent_right_scores,
            "recentOpponentScores": self.recent_opponent_scores,
        }

    def _empty_prediction_state(self):
        return {
            "gridWidth": self.input_w,
            "gridHeight": self.input_h,
            "actualActive": 0,
            "predictedActive": 0,
            "hits": 0,
            "misses": 0,
            "falsePositives": 0,
            "meanAbsError": 0.0,
            "modulatorySignal": 0.0,
            "sample": [],
        }

    def _empty_eligibility_stats(self):
        return {
            "inputH1": self.input_hidden.stats(),
            "motorContext": self.motor_hidden.stats(),
            "recurrentCloud": self.recurrent.stats(),
            "predictionHead": self.hidden_prediction.stats(),
            "h3Output": self.hidden_output.stats(),
        }

    def _empty_learning_stats(self):
        return {
            "step": self.train_steps,
            "learningRate": self.learning_rate,
            "predictionLearningRate": self.prediction_learning_rate,
            "rewardApplied": 0.0,
            "predictionApplied": 0.0,
            "combinedModulator": 0.0,
            "weightUpdates": 0,
            "predictionWeightUpdates": 0,
            "potentiated": 0,
            "depressed": 0,
            "clamped": 0,
            "meanAbsDelta": 0.0,
            "eligibilityIncreased": 0,
            "eligibilityDecreased": 0,
        }

    def architecture(self):
        total_connections = (
            len(self.input_hidden)
            + len(self.motor_hidden)
            + len(self.recurrent)
            + len(self.hidden_output)
            + len(self.hidden_prediction)
        )
        return {
            "version": self.version,
            "device": self.device,
            "input": {
                "name": "compressed event camera grid",
                "sourceWidth": self.width,
                "sourceHeight": self.height,
                "width": self.input_w,
                "height": self.input_h,
                "neurons": self.input_size,
                "activeRule": "any event-camera pixel in a 64x36 cell spikes that cell",
            },
            "motorContext": {
                "neurons": self.motor_size,
                "labels": [item["name"] for item in OUTPUTS],
                "traceDecay": self.motor_trace_decay,
                "activeRule": "previous action is fed back through decaying traces",
            },
            "layers": [
                {
                    "name": "hidden1",
                    "role": "recurrent hidden cloud",
                    "width": self.hidden_w,
                    "height": self.hidden_h,
                    "neurons": self.hidden_size,
                    "excitatory": self.excitatory_count,
                    "inhibitory": self.inhibitory_count,
                    "receptiveField": "sparse distance-biased recurrence",
                    "connections": len(self.recurrent),
                },
                {
                    "name": "prediction",
                    "role": "next event grid prediction head",
                    "width": self.input_w,
                    "height": self.input_h,
                    "neurons": self.input_size,
                    "receptiveField": "local hidden-cloud readout",
                    "connections": len(self.hidden_prediction),
                },
                {
                    "name": "output",
                    "role": "motor output",
                    "width": 3,
                    "height": 1,
                    "neurons": 3,
                    "receptiveField": "hidden-cloud readout",
                    "connections": len(self.hidden_output),
                    "labels": [item["name"] for item in OUTPUTS],
                },
            ],
            "sparsity": {
                "inputHiddenConnections": len(self.input_hidden),
                "motorContextConnections": len(self.motor_hidden),
                "recurrentConnections": len(self.recurrent),
                "predictionConnections": len(self.hidden_prediction),
                "outputConnections": len(self.hidden_output),
                "totalConnections": total_connections,
            },
            "stdp": {
                "mode": "reward and prediction-error modulated eligibility traces",
                "learningRate": self.learning_rate,
                "predictionLearningRate": self.prediction_learning_rate,
                "eligibilityDecay": self.eligibility_decay,
                "eligibilityTraces": self._eligibility_trace_config(),
                "effectiveEligibility": "fast + 0.35 * medium + 0.08 * slow",
                "eligibilityPlus": self.eligibility_plus,
                "eligibilityMinus": self.eligibility_minus,
                "rule": "local pre/post coactivity updates eligibility; reward and predictive error gate weight changes",
                "rewardComponents": self.reward_function.config(),
            },
        }

    def _eligibility_trace_config(self):
        return [dict(trace) for trace in self.eligibility_traces]

    def _compress_events(self, pixels, camera_width, camera_height):
        active_pixels = []
        active_cells = set()
        max_index = camera_width * camera_height
        for raw_pixel in pixels:
            try:
                pixel = int(raw_pixel)
            except (TypeError, ValueError):
                continue
            if pixel < 0 or pixel >= max_index:
                continue
            active_pixels.append(pixel)
            y = pixel // camera_width
            x = pixel - y * camera_width
            cx = min(self.input_w - 1, int(x * self.input_w / max(1, camera_width)))
            cy = min(self.input_h - 1, int(y * self.input_h / max(1, camera_height)))
            active_cells.add(_grid_index(cx, cy, self.input_w))
        return active_pixels, sorted(active_cells)

    def _add_group_drive(self, drive, group, active_pres, scale=1.0):
        for pre in active_pres:
            if pre < 0 or pre >= group.pre_size:
                continue
            for edge_index in group.by_pre[pre]:
                drive[group.post[edge_index]] += group.weight[edge_index] * scale

    def _add_motor_drive(self, drive):
        for pre, trace in enumerate(self.previous_action_traces):
            if trace <= 0.005:
                continue
            self._add_group_drive(drive, self.motor_hidden, (pre,), trace)

    def _hidden_spikes(self, drive):
        candidates = []
        for index, incoming in enumerate(drive):
            value = self.hidden_potential[index] * 0.80 + incoming
            self.hidden_potential[index] = value
            if value >= 0.42:
                candidates.append((index, value))
        if not candidates:
            strongest = sorted(((index, value) for index, value in enumerate(drive) if value > 0.0), key=lambda item: item[1], reverse=True)
            candidates = strongest[:16]
        if len(candidates) > 220:
            candidates = sorted(candidates, key=lambda item: item[1], reverse=True)[:220]
        spikes = [index for index, _ in candidates]
        for index in spikes:
            self.hidden_potential[index] *= 0.18
        return spikes

    def _output(self, hidden_spikes):
        drives = [0.0, 0.0, 0.035]
        for pre in hidden_spikes:
            for edge_index in self.hidden_output.by_pre[pre]:
                drives[self.hidden_output.post[edge_index]] += self.hidden_output.weight[edge_index]
        if not hidden_spikes:
            winner_index = 2
        else:
            winner_index = max(range(len(drives)), key=lambda index: drives[index])
        low = min(drives)
        high = max(drives)
        if high - low <= 0.00001:
            bars = [0.0, 0.0, 1.0]
        else:
            bars = [(value - low) / (high - low) for value in drives]
        return drives, bars, winner_index

    def _predict_next(self, hidden_spikes):
        scores = {}
        edge_count = 0
        for pre in hidden_spikes:
            for edge_index in self.hidden_prediction.by_pre[pre]:
                post = self.hidden_prediction.post[edge_index]
                scores[post] = scores.get(post, 0.0) + self.hidden_prediction.weight[edge_index]
                edge_count += 1
        prediction = {}
        for post, score in scores.items():
            probability = _sigmoid(score - 0.28)
            if probability >= 0.08:
                prediction[post] = probability
        self.last_prediction_edges = edge_count
        return prediction

    def _prediction_error(self, active_cells):
        actual = set(active_cells)
        predicted = {cell for cell, probability in self.last_prediction.items() if probability >= 0.32}
        union = actual | set(self.last_prediction)
        if not union:
            self.prediction_state = self._empty_prediction_state()
            return 0.0, {}

        post_errors = {}
        total_abs = 0.0
        hits = 0
        misses = 0
        false_positives = 0
        for cell in union:
            target = 1.0 if cell in actual else 0.0
            probability = self.last_prediction.get(cell, 0.0)
            error = target - probability
            total_abs += abs(error)
            if abs(error) >= 0.02:
                post_errors[cell] = error
            if cell in actual and cell in predicted:
                hits += 1
            elif cell in actual:
                misses += 1
            elif cell in predicted:
                false_positives += 1

        mean_abs = total_abs / max(1, len(union))
        quality = (hits - misses - false_positives) / max(1, len(actual) + len(predicted))
        modulatory = max(-1.0, min(1.0, quality))
        self.prediction_state = {
            "gridWidth": self.input_w,
            "gridHeight": self.input_h,
            "actualActive": len(actual),
            "predictedActive": len(predicted),
            "hits": hits,
            "misses": misses,
            "falsePositives": false_positives,
            "meanAbsError": round(mean_abs, 5),
            "modulatorySignal": round(modulatory, 5),
            "sample": sorted(predicted)[:900],
        }
        return modulatory, post_errors

    def _decay_eligibilities(self):
        self.input_hidden.decay()
        self.motor_hidden.decay()
        self.recurrent.decay()
        self.hidden_output.decay()
        self.hidden_prediction.decay()

    def _update_pre_post_eligibility(self, group, active_pres, active_posts, plus=None, minus=None, pre_scales=None):
        active_post_set = set(active_posts)
        changed = {"increased": 0, "decreased": 0}
        plus = self.eligibility_plus if plus is None else plus
        minus = self.eligibility_minus if minus is None else minus
        for pre in active_pres:
            if pre < 0 or pre >= group.pre_size:
                continue
            pre_scale = 1.0 if pre_scales is None else pre_scales.get(pre, 0.0)
            if pre_scale <= 0.0:
                continue
            for edge_index in group.by_pre[pre]:
                if group.post[edge_index] in active_post_set:
                    group.add_eligibility(edge_index, plus * pre_scale)
                    changed["increased"] += 1
                else:
                    group.add_eligibility(edge_index, -minus * pre_scale)
                    changed["decreased"] += 1
        return changed

    def _update_prediction_eligibility(self, hidden_spikes):
        changed = {"increased": 0, "decreased": 0}
        for pre in hidden_spikes:
            for edge_index in self.hidden_prediction.by_pre[pre]:
                self.hidden_prediction.add_eligibility(edge_index, self.eligibility_plus)
                changed["increased"] += 1
        return changed

    def _summarize_eligibility(self):
        return {
            "inputH1": self.input_hidden.stats(),
            "motorContext": self.motor_hidden.stats(),
            "recurrentCloud": self.recurrent.stats(),
            "predictionHead": self.hidden_prediction.stats(),
            "h3Output": self.hidden_output.stats(),
        }

    def _apply_learning(self, reward, predictive_signal, prediction_errors, local_changes):
        stats = self._empty_learning_stats()
        stats["step"] = self.train_steps + 1
        stats["rewardApplied"] = round(reward, 5)
        stats["predictionApplied"] = round(predictive_signal, 5)
        combined = reward + self.predictive_modulation_scale * predictive_signal
        stats["combinedModulator"] = round(combined, 5)
        stats["eligibilityIncreased"] = local_changes["increased"]
        stats["eligibilityDecreased"] = local_changes["decreased"]

        self.hidden_prediction.apply_post_errors(self.prediction_learning_rate, prediction_errors, stats)
        for group in (self.input_hidden, self.motor_hidden, self.recurrent, self.hidden_output):
            group.apply_modulator(self.learning_rate, combined, stats)

        if stats["weightUpdates"] > 0:
            stats["meanAbsDelta"] = round(stats["meanAbsDelta"] / stats["weightUpdates"], 8)
        else:
            stats["meanAbsDelta"] = 0.0
        return stats

    def _snapshot_pong(self, game_state):
        if not isinstance(game_state, dict):
            return None
        score = game_state.get("score") if isinstance(game_state.get("score"), dict) else {}
        try:
            tick = int(game_state.get("tick", game_state.get("authoritativeTick", self.tick)) or 0)
            reset_token = game_state.get("resetToken")
            left_score = int(score.get("left", 0))
            right_score = int(score.get("right", 0))
        except (TypeError, ValueError):
            return None
        return {
            "tick": tick,
            "resetToken": reset_token,
            "leftScore": left_score,
            "rightScore": right_score,
        }

    def _reward_from_pong(self, game_state, output_bars=None, winner_index=None):
        current = self._snapshot_pong(game_state)
        previous = self.last_pong_snapshot
        self.last_pong_snapshot = current

        components, last_event, metrics = self.reward_function.components(current, previous, winner_index)
        if components["rightScore"] != 0.0:
            self.recent_right_scores += 1
        if components["opponentScore"] != 0.0:
            self.recent_opponent_scores += 1

        total = self.reward_function.total(components)
        self.reward_state = {
            "value": round(total, 5),
            "components": {key: round(value, 5) for key, value in components.items()},
            "metrics": metrics,
            "lastEvent": last_event,
            "recentRightScores": self.recent_right_scores,
            "recentOpponentScores": self.recent_opponent_scores,
        }
        return total

    def _update_motor_traces(self, winner_index):
        self.previous_action_traces = [value * self.motor_trace_decay for value in self.previous_action_traces]
        self.previous_action_traces[winner_index] = 1.0

    def step(self, event_camera, tick, game_state=None):
        if not event_camera:
            return None
        pixels = event_camera.get("pixels") if isinstance(event_camera, dict) else []
        if not isinstance(pixels, list):
            pixels = []

        self.tick = int(tick or self.tick)
        camera_width = max(1, int(event_camera.get("width", self.width) or self.width))
        camera_height = max(1, int(event_camera.get("height", self.height) or self.height))
        active_pixels, active_cells = self._compress_events(pixels, camera_width, camera_height)

        drive = [0.0 for _ in range(self.hidden_size)]
        self._add_group_drive(drive, self.input_hidden, active_cells)
        self._add_motor_drive(drive)
        self._add_group_drive(drive, self.recurrent, self.last_hidden_spikes)
        hidden_spikes = self._hidden_spikes(drive)
        output_drive, output_bars, winner_index = self._output(hidden_spikes)
        winner = OUTPUTS[winner_index]

        changed = {
            "potentiated": 0,
            "depressed": 0,
            "eligibilityIncreased": 0,
            "eligibilityDecreased": 0,
        }
        if self.training and not self.paused:
            reward = self._reward_from_pong(game_state, output_bars=output_bars, winner_index=winner_index)
            self._decay_eligibilities()
            predictive_signal, prediction_errors = self._prediction_error(active_cells)
            local_changes = {"increased": 0, "decreased": 0}
            motor_scales = {
                index: trace
                for index, trace in enumerate(self.previous_action_traces)
                if trace > 0.005
            }
            for update in (
                self._update_pre_post_eligibility(self.input_hidden, active_cells, hidden_spikes),
                self._update_pre_post_eligibility(self.motor_hidden, range(self.motor_size), hidden_spikes, pre_scales=motor_scales),
                self._update_pre_post_eligibility(self.recurrent, self.last_hidden_spikes, hidden_spikes),
                self._update_pre_post_eligibility(self.hidden_output, hidden_spikes, (winner_index,)),
                self._update_prediction_eligibility(hidden_spikes),
            ):
                local_changes["increased"] += update["increased"]
                local_changes["decreased"] += update["decreased"]
            self.learning_stats = self._apply_learning(reward, predictive_signal, prediction_errors, local_changes)
            if self.train_steps % self.status_summary_interval == 0:
                self.eligibility_stats = self._summarize_eligibility()
            changed["potentiated"] = self.learning_stats["potentiated"]
            changed["depressed"] = self.learning_stats["depressed"]
            changed["eligibilityIncreased"] = local_changes["increased"]
            changed["eligibilityDecreased"] = local_changes["decreased"]
            self.train_steps += 1
        else:
            self._prediction_error(active_cells)

        prediction = self._predict_next(hidden_spikes)
        self.last_prediction = prediction
        direction = int(winner["direction"])
        self._update_motor_traces(winner_index)
        self.last_hidden_spikes = hidden_spikes

        hidden_sample = hidden_spikes[:900]
        self.activity = {
            "tick": self.tick,
            "eventCount": len(active_pixels),
            "compressedEventCount": len(active_cells),
            "spikes": {
                "input": len(active_cells),
                "hidden1": len(hidden_spikes),
                "hidden2": len(self.last_hidden_spikes),
                "hidden3": self.last_prediction_edges,
                "output": 1 if hidden_spikes else 0,
            },
            "outputDrive": [round(value, 4) for value in output_drive],
            "outputBars": [round(value, 4) for value in output_bars],
            "winner": winner["name"],
            "direction": direction,
            "activeInputSample": active_cells[:900],
            "activeHidden1": hidden_sample,
            "activeHidden2": hidden_sample[:320],
            "activeHidden3": sorted(prediction, key=prediction.get, reverse=True)[:256],
            "motorTraces": [round(value, 4) for value in self.previous_action_traces],
            "prediction": self.prediction_state,
            "stdp": changed,
            "reward": self.reward_state,
            "eligibility": self.eligibility_stats,
            "learning": self.learning_stats,
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
            "reward": self.reward_state,
            "prediction": self.prediction_state,
            "eligibility": self.eligibility_stats,
            "learning": self.learning_stats,
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
            "inputGrid": {"width": self.input_w, "height": self.input_h},
            "hidden": {
                "size": self.hidden_size,
                "width": self.hidden_w,
                "height": self.hidden_h,
            },
            "previousActionTraces": self.previous_action_traces,
            "groups": {
                "inputHidden": self.input_hidden.to_dict(),
                "motorHidden": self.motor_hidden.to_dict(),
                "recurrentCloud": self.recurrent.to_dict(),
                "hiddenOutput": self.hidden_output.to_dict(),
                "predictionHead": self.hidden_prediction.to_dict(),
            },
            "savedAt": time.time(),
        }

    def load_dict(self, payload):
        if int(payload.get("version", 0)) != self.version:
            raise ValueError("saved network version does not match the recurrent predictive backend")
        if int(payload.get("width", self.width)) != self.width or int(payload.get("height", self.height)) != self.height:
            raise ValueError("saved network dimensions do not match this Pong event camera")
        hidden = payload.get("hidden", {})
        grid = payload.get("inputGrid", {})
        if int(hidden.get("size", self.hidden_size)) != self.hidden_size:
            raise ValueError("saved hidden cloud size does not match this backend")
        if int(grid.get("width", self.input_w)) != self.input_w or int(grid.get("height", self.input_h)) != self.input_h:
            raise ValueError("saved input grid does not match this backend")

        self.seed = int(payload.get("seed", self.seed))
        self.train_steps = int(payload.get("trainSteps", 0))
        traces = payload.get("previousActionTraces", self.previous_action_traces)
        self.previous_action_traces = [float(value) for value in traces[: self.motor_size]]
        while len(self.previous_action_traces) < self.motor_size:
            self.previous_action_traces.append(0.0)

        groups = payload.get("groups", {})
        self.input_hidden.load_weights(groups["inputHidden"])
        self.motor_hidden.load_weights(groups["motorHidden"])
        self.recurrent.load_weights(groups["recurrentCloud"])
        self.hidden_output.load_weights(groups["hiddenOutput"])
        self.hidden_prediction.load_weights(groups["predictionHead"])

        self.last_hidden_spikes = []
        self.last_prediction = {}
        self.last_pong_snapshot = None
        self.reward_function.reset()
        self.recent_right_scores = 0
        self.recent_opponent_scores = 0
        self.reward_state = self._empty_reward_state()
        self.prediction_state = self._empty_prediction_state()
        self.eligibility_stats = self._empty_eligibility_stats()
        self.learning_stats = self._empty_learning_stats()
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
