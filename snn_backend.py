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

# Edit these values to change the backend architecture shape without touching the
# builder logic. Defaults mirror the pre-config version of the recurrent SNN.
SNN_ARCHITECTURE = {
    "input_grid_width": 64,
    "input_grid_height": 36,
    "hidden_neurons": 256,
    "hidden_grid_width": 32,
    "hidden_grid_height": None,
    "excitatory_fraction": 0.8,
    "input_hidden_targets_per_cell": 6,
    "input_hidden_sigma_x": 2.0,
    "input_hidden_sigma_y": 1.4,
    "input_hidden_attempt_multiplier": 8,
    "motor_hidden_targets_per_action": 64,
    "motor_hidden_sigma_y_fraction": 0.1,
    "recurrent_local_edges_per_neuron": 3,
    "recurrent_long_range_edges_per_neuron": 1,
    "recurrent_local_sigma_x": 2.1,
    "recurrent_local_sigma_y": 1.4,
    "output_targets_per_hidden": 1,
    "prediction_local_targets": 2,
    "prediction_medium_targets": 1,
    "prediction_long_targets": 0,
    "prediction_local_sigma_x": 1.2,
    "prediction_local_sigma_y": 0.8,
    "prediction_medium_sigma_x": 3.0,
    "prediction_medium_sigma_y": 2.0,
    "structured_hidden_spikes": 48,
    "hidden_spike_cap": 64,
    "prediction_motion_horizon": 1.15,
    "prediction_persistence": 0.18,
    "prediction_min_probability": 0.08,
    "prediction_strong_threshold": 0.32,
    "prediction_max_cells": 260,
    "event_trace_decay": 0.82,
    "velocity_decay": 0.72,
    "motion_match_radius": 4,
    "controlled_edge_fraction": 0.16,
    "visual_servo_gain": 1.35,
    "visual_servo_deadband": 1.4,
    "policy_learning_rate": 0.04,
    "policy_trace_decay": 0.94,
    "policy_weight_decay": 0.9995,
    "policy_error_bins": 7,
    "policy_x_bins": 4,
    "max_input_eligibility_edges": 900,
    "max_motor_eligibility_edges": 192,
    "max_recurrent_eligibility_edges": 520,
    "max_prediction_eligibility_edges": 420,
    "max_output_eligibility_edges": 256,
}


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def _architecture_config(overrides=None):
    config = dict(SNN_ARCHITECTURE)
    if overrides:
        config.update(overrides)
    return config


def _positive_int(config, key, minimum=1):
    return max(minimum, int(config[key]))


def _non_negative_int(config, key):
    return max(0, int(config[key]))


def _positive_float(config, key):
    return max(0.000001, float(config[key]))


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

    def trim_active_edges(self, max_edges):
        if max_edges <= 0 or len(self.active_edges) <= max_edges:
            return
        ranked = sorted(self.active_edges, key=lambda index: abs(self.effective(index)), reverse=True)
        keep = set(ranked[:max_edges])
        drop = self.active_edges - keep
        for trace in self.traces:
            values = self.eligibility[trace["name"]]
            for index in drop:
                values[index] = 0.0
        self.active_edges = keep

    def apply_modulator(self, learning_rate, modulator, stats, update_key=None, edge_filter=None):
        if modulator == 0.0:
            return
        for index in list(self.active_edges):
            if edge_filter is not None and not edge_filter(index):
                continue
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
            if update_key:
                stats[update_key] += 1
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

    version = 3

    def __init__(self, width=800, height=450, seed=None, save_dir=None, architecture=None):
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

        self.architecture_config = _architecture_config(architecture)
        self.input_w = _positive_int(self.architecture_config, "input_grid_width")
        self.input_h = _positive_int(self.architecture_config, "input_grid_height")
        self.input_size = self.input_w * self.input_h
        self.motor_size = len(OUTPUTS)
        self.hidden_size = _positive_int(self.architecture_config, "hidden_neurons")
        self.hidden_w = _positive_int(self.architecture_config, "hidden_grid_width")
        hidden_grid_height = self.architecture_config.get("hidden_grid_height")
        if hidden_grid_height is None:
            self.hidden_h = max(1, math.ceil(self.hidden_size / self.hidden_w))
        else:
            self.hidden_h = _positive_int(self.architecture_config, "hidden_grid_height")
        if self.hidden_w * self.hidden_h < self.hidden_size:
            raise ValueError("hidden_grid_width * hidden_grid_height must cover hidden_neurons")
        excitatory_fraction = _clamp(float(self.architecture_config["excitatory_fraction"]))
        self.excitatory_count = int(self.hidden_size * excitatory_fraction)
        self.inhibitory_count = self.hidden_size - self.excitatory_count

        self.learning_rate = 0.006
        self.prediction_learning_rate = 0.028
        self.prediction_pathway_learning_rate = 0.0016
        self.reward_recurrent_learning_rate = 0.0007
        self.status_summary_interval = 30
        self.input_hidden_targets_per_cell = _positive_int(self.architecture_config, "input_hidden_targets_per_cell")
        self.input_hidden_sigma_x = _positive_float(self.architecture_config, "input_hidden_sigma_x")
        self.input_hidden_sigma_y = _positive_float(self.architecture_config, "input_hidden_sigma_y")
        self.input_hidden_attempt_multiplier = _positive_int(self.architecture_config, "input_hidden_attempt_multiplier")
        self.motor_hidden_targets_per_action = _non_negative_int(self.architecture_config, "motor_hidden_targets_per_action")
        self.motor_hidden_sigma_y_fraction = _positive_float(self.architecture_config, "motor_hidden_sigma_y_fraction")
        self.recurrent_local_edges_per_neuron = _non_negative_int(self.architecture_config, "recurrent_local_edges_per_neuron")
        self.recurrent_long_range_edges_per_neuron = _non_negative_int(self.architecture_config, "recurrent_long_range_edges_per_neuron")
        self.recurrent_local_sigma_x = _positive_float(self.architecture_config, "recurrent_local_sigma_x")
        self.recurrent_local_sigma_y = _positive_float(self.architecture_config, "recurrent_local_sigma_y")
        self.output_targets_per_hidden = _positive_int(self.architecture_config, "output_targets_per_hidden")
        self.prediction_local_targets = _non_negative_int(self.architecture_config, "prediction_local_targets")
        self.prediction_medium_targets = _non_negative_int(self.architecture_config, "prediction_medium_targets")
        self.prediction_long_targets = _non_negative_int(self.architecture_config, "prediction_long_targets")
        self.prediction_local_sigma_x = _positive_float(self.architecture_config, "prediction_local_sigma_x")
        self.prediction_local_sigma_y = _positive_float(self.architecture_config, "prediction_local_sigma_y")
        self.prediction_medium_sigma_x = _positive_float(self.architecture_config, "prediction_medium_sigma_x")
        self.prediction_medium_sigma_y = _positive_float(self.architecture_config, "prediction_medium_sigma_y")
        self.structured_hidden_spikes = _positive_int(self.architecture_config, "structured_hidden_spikes")
        self.hidden_spike_cap = _positive_int(self.architecture_config, "hidden_spike_cap")
        self.prediction_motion_horizon = _positive_float(self.architecture_config, "prediction_motion_horizon")
        self.prediction_persistence = _clamp(float(self.architecture_config["prediction_persistence"]))
        self.prediction_min_probability = _clamp(float(self.architecture_config["prediction_min_probability"]))
        self.prediction_strong_threshold = _clamp(float(self.architecture_config["prediction_strong_threshold"]))
        self.prediction_max_cells = _positive_int(self.architecture_config, "prediction_max_cells")
        self.event_trace_decay = _clamp(float(self.architecture_config["event_trace_decay"]))
        self.velocity_decay = _clamp(float(self.architecture_config["velocity_decay"]))
        self.motion_match_radius = _positive_int(self.architecture_config, "motion_match_radius")
        self.controlled_edge_fraction = _clamp(float(self.architecture_config["controlled_edge_fraction"]), 0.01, 0.5)
        self.visual_servo_gain = _positive_float(self.architecture_config, "visual_servo_gain")
        self.visual_servo_deadband = _positive_float(self.architecture_config, "visual_servo_deadband")
        self.policy_learning_rate = _positive_float(self.architecture_config, "policy_learning_rate")
        self.policy_trace_decay = _clamp(float(self.architecture_config["policy_trace_decay"]))
        self.policy_weight_decay = _clamp(float(self.architecture_config["policy_weight_decay"]))
        self.policy_error_bins = _positive_int(self.architecture_config, "policy_error_bins")
        self.policy_x_bins = _positive_int(self.architecture_config, "policy_x_bins")
        self.policy_feature_count = self.policy_error_bins * self.policy_x_bins
        self.max_input_eligibility_edges = _positive_int(self.architecture_config, "max_input_eligibility_edges")
        self.max_motor_eligibility_edges = _positive_int(self.architecture_config, "max_motor_eligibility_edges")
        self.max_recurrent_eligibility_edges = _positive_int(self.architecture_config, "max_recurrent_eligibility_edges")
        self.max_prediction_eligibility_edges = _positive_int(self.architecture_config, "max_prediction_eligibility_edges")
        self.max_output_eligibility_edges = _positive_int(self.architecture_config, "max_output_eligibility_edges")
        self.eligibility_plus = 0.14
        self.eligibility_minus = 0.03
        self.eligibility_traces = tuple(dict(trace) for trace in ELIGIBILITY_TRACES)
        self.eligibility_decay = self.eligibility_traces[0]["decay"]
        self.motor_trace_decay = 0.86

        self.hidden_potential = [0.0 for _ in range(self.hidden_size)]
        self.input_trace = [0.0 for _ in range(self.input_size)]
        self.cell_vx = [0.0 for _ in range(self.input_size)]
        self.cell_vy = [0.0 for _ in range(self.input_size)]
        self.previous_active_cells = []
        self.tracked_target_y = self.input_h / 2
        self.tracked_target_x = self.input_w / 2
        self.tracked_effector_y = self.input_h / 2
        self.last_event_features = self._empty_event_features()
        self.policy_weights = [[0.0 for _ in range(self.motor_size)] for _ in range(self.policy_feature_count)]
        self.policy_traces = []
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
        self.motor_context_neurons = set(self.motor_hidden.post)
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
        return _grid_index(x % self.hidden_w, y % self.hidden_h, self.hidden_w) % self.hidden_size

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
        for pre in range(self.input_size):
            cx, cy = self._input_to_hidden_center(pre)
            posts = set()
            attempts = 0
            max_attempts = self.input_hidden_targets_per_cell * self.input_hidden_attempt_multiplier
            while len(posts) < self.input_hidden_targets_per_cell and attempts < max_attempts:
                attempts += 1
                hx = int(round(self.rng.gauss(cx, self.input_hidden_sigma_x)))
                hy = int(round(self.rng.gauss(cy, self.input_hidden_sigma_y)))
                hx = max(0, min(self.hidden_w - 1, hx))
                hy = max(0, min(self.hidden_h - 1, hy))
                posts.add(self._hidden_index(hx, hy))
            posts.add(self._hidden_index(cx, cy))
            for post in sorted(posts):
                weight = 0.30 + self.rng.random() * 0.32
                edges.append((pre, post, weight, 0.0, 1.0))
        return edges

    def _build_motor_hidden_edges(self):
        edges = []
        for pre in range(self.motor_size):
            target_y = (pre + 0.5) * self.hidden_h / self.motor_size
            for _ in range(self.motor_hidden_targets_per_action):
                hx = self.rng.randrange(self.hidden_w)
                sigma_y = self.hidden_h * self.motor_hidden_sigma_y_fraction
                hy = int(max(0, min(self.hidden_h - 1, self.rng.gauss(target_y, sigma_y))))
                post = self._hidden_index(hx, hy)
                weight = 0.08 + self.rng.random() * 0.26
                edges.append((pre, post, weight, 0.0, 1.0))
        return edges

    def _build_recurrent_edges(self):
        edges = []
        for pre in range(self.hidden_size):
            px, py = self._hidden_xy(pre)
            low, high = self._signed_bounds(pre)
            for _ in range(self.recurrent_local_edges_per_neuron):
                dx = int(round(self.rng.gauss(0, self.recurrent_local_sigma_x)))
                dy = int(round(self.rng.gauss(0, self.recurrent_local_sigma_y)))
                post = self._hidden_index(px + dx, py + dy)
                if post == pre:
                    post = self._hidden_index(px + 1, py)
                weight = self._signed_weight(pre, 0.22, 0.72)
                edges.append((pre, post, weight, low, high))
            for _ in range(self.recurrent_long_range_edges_per_neuron):
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
            action_scores = [
                0.62 * (1.0 - vertical) + self.rng.random() * 0.22,
                0.62 * vertical + self.rng.random() * 0.22,
                0.36 - abs(vertical - 0.5) * 0.22 + self.rng.random() * 0.24,
            ]
            if self.output_targets_per_hidden == 1:
                targets = [max(range(self.motor_size), key=lambda index: action_scores[index])]
            else:
                targets = sorted(range(self.motor_size), key=lambda index: action_scores[index], reverse=True)[
                    : min(self.motor_size, self.output_targets_per_hidden)
                ]
            for target in targets:
                magnitude = 0.055 + self.rng.random() * 0.11
                weight = magnitude if self.neuron_is_excitatory[pre] else -magnitude
                edges.append((pre, target, weight, low, high))
        return edges

    def _build_prediction_edges(self):
        edges = []
        for pre in range(self.hidden_size):
            hx, hy = self._hidden_xy(pre)
            gx = int((hx + 0.5) * self.input_w / self.hidden_w)
            gy = int((hy + 0.5) * self.input_h / self.hidden_h)
            low, high = self._signed_bounds(pre)
            posts = set()
            self._add_prediction_targets(
                posts,
                gx,
                gy,
                self.prediction_local_targets,
                sigma_x=self.prediction_local_sigma_x,
                sigma_y=self.prediction_local_sigma_y,
            )
            self._add_prediction_targets(
                posts,
                gx,
                gy,
                self.prediction_medium_targets,
                sigma_x=self.prediction_medium_sigma_x,
                sigma_y=self.prediction_medium_sigma_y,
            )
            while len(posts) < self.prediction_local_targets + self.prediction_medium_targets + self.prediction_long_targets:
                posts.add(self.rng.randrange(self.input_size))
            for post in sorted(posts):
                weight = self._signed_weight(pre, 0.025, 0.18)
                edges.append((pre, post, weight, low, high))
        return edges

    def _add_prediction_targets(self, posts, cx, cy, count, sigma_x, sigma_y):
        attempts = 0
        target_size = len(posts) + count
        while len(posts) < target_size and attempts < count * 10:
            attempts += 1
            px = int(round(self.rng.gauss(cx, sigma_x)))
            py = int(round(self.rng.gauss(cy, sigma_y)))
            px = max(0, min(self.input_w - 1, px))
            py = max(0, min(self.input_h - 1, py))
            posts.add(_grid_index(px, py, self.input_w))

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
            "eventFeatures": self._empty_event_features(),
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

    def _empty_event_features(self):
        return {
            "targetX": self.input_w / 2,
            "targetY": self.input_h / 2,
            "targetVx": 0.0,
            "targetVy": 0.0,
            "effectorY": self.input_h / 2,
            "errorY": 0.0,
            "confidence": 0.0,
            "featureIndex": self.policy_feature_count // 2 if hasattr(self, "policy_feature_count") else 0,
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
            "predictionPathwayLearningRate": self.prediction_pathway_learning_rate,
            "rewardRecurrentLearningRate": self.reward_recurrent_learning_rate,
            "rewardApplied": 0.0,
            "predictionApplied": 0.0,
            "combinedModulator": 0.0,
            "weightUpdates": 0,
            "rewardWeightUpdates": 0,
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
                "hiddenFanoutPerCell": self.input_hidden_targets_per_cell,
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
                    "receptiveField": "structured event trace, local motion, and sparse distance-biased recurrence",
                    "connections": len(self.recurrent),
                },
                {
                    "name": "prediction",
                    "role": "next event grid prediction head",
                    "width": self.input_w,
                    "height": self.input_h,
                    "neurons": self.input_size,
                    "receptiveField": "learned local readout plus per-cell temporal motion prediction",
                    "connections": len(self.hidden_prediction),
                },
                {
                    "name": "output",
                    "role": "motor output",
                    "width": 3,
                    "height": 1,
                    "neurons": 3,
                    "receptiveField": "sparse hidden readout plus event-feature reward policy",
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
                "predictionPathwayLearningRate": self.prediction_pathway_learning_rate,
                "rewardRecurrentLearningRate": self.reward_recurrent_learning_rate,
                "eligibilityDecay": self.eligibility_decay,
                "eligibilityTraces": self._eligibility_trace_config(),
                "effectiveEligibility": "fast + 0.35 * medium + 0.08 * slow",
                "eligibilityPlus": self.eligibility_plus,
                "eligibilityMinus": self.eligibility_minus,
                "rule": "prediction error trains visual/recurrent/prediction pathways; reward trains motor-context/output pathways, a small motor-adjacent recurrent subset, and event-feature policy traces",
                "rewardComponents": self.reward_function.config(),
            },
            "eventModel": {
                "traceDecay": self.event_trace_decay,
                "velocityDecay": self.velocity_decay,
                "motionMatchRadius": self.motion_match_radius,
                "predictionMotionHorizon": self.prediction_motion_horizon,
                "predictionStrongThreshold": self.prediction_strong_threshold,
                "policyFeatures": self.policy_feature_count,
            },
            "config": dict(self.architecture_config),
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

    def _cell_xy(self, index):
        return index % self.input_w, index // self.input_w

    def _input_index(self, x, y):
        return _grid_index(
            max(0, min(self.input_w - 1, int(round(x)))),
            max(0, min(self.input_h - 1, int(round(y)))),
            self.input_w,
        )

    def _nearest_previous_cell(self, x, y, previous):
        best = None
        best_distance = (self.motion_match_radius + 1) ** 2
        for dy in range(-self.motion_match_radius, self.motion_match_radius + 1):
            row = y + dy
            if row < 0 or row >= self.input_h:
                continue
            for dx in range(-self.motion_match_radius, self.motion_match_radius + 1):
                col = x + dx
                if col < 0 or col >= self.input_w:
                    continue
                candidate = _grid_index(col, row, self.input_w)
                if candidate not in previous:
                    continue
                distance = dx * dx + dy * dy
                if distance < best_distance:
                    best = (col, row, candidate)
                    best_distance = distance
        return best

    def _policy_feature_index(self, error_y, target_x):
        span = max(1.0, self.input_h / 2)
        normalized_error = _clamp((error_y / span + 1.0) * 0.5)
        error_bin = min(self.policy_error_bins - 1, int(normalized_error * self.policy_error_bins))
        x_bin = min(self.policy_x_bins - 1, int(_clamp(target_x / max(1, self.input_w - 1)) * self.policy_x_bins))
        return x_bin * self.policy_error_bins + error_bin

    def _update_event_model(self, active_cells):
        previous = set(self.previous_active_cells)
        for index, value in enumerate(self.input_trace):
            self.input_trace[index] = value * self.event_trace_decay

        active_set = set(active_cells)
        for index in active_set:
            self.input_trace[index] = 1.0

        edge_start = int(self.input_w * (1.0 - self.controlled_edge_fraction))
        effector_weight = 0.0
        effector_y = 0.0
        target_weight = 0.0
        target_x = 0.0
        target_y = 0.0
        target_vx = 0.0
        target_vy = 0.0

        for index in active_cells:
            x, y = self._cell_xy(index)
            nearest = self._nearest_previous_cell(x, y, previous) if previous else None
            if nearest is None:
                dx = 0.0
                dy = 0.0
                previous_vx = 0.0
                previous_vy = 0.0
            else:
                dx = x - nearest[0]
                dy = y - nearest[1]
                previous_vx = self.cell_vx[nearest[2]]
                previous_vy = self.cell_vy[nearest[2]]
            self.cell_vx[index] = previous_vx * self.velocity_decay + dx * (1.0 - self.velocity_decay)
            self.cell_vy[index] = previous_vy * self.velocity_decay + dy * (1.0 - self.velocity_decay)

            if x >= edge_start:
                weight = 1.0 + self.input_trace[index]
                effector_y += y * weight
                effector_weight += weight
            else:
                motion = abs(self.cell_vx[index]) + abs(self.cell_vy[index])
                right_bias = 0.35 + 0.65 * (x / max(1, edge_start))
                weight = (0.35 + motion) * right_bias
                target_x += x * weight
                target_y += y * weight
                target_vx += self.cell_vx[index] * weight
                target_vy += self.cell_vy[index] * weight
                target_weight += weight

        if effector_weight > 0.0:
            self.tracked_effector_y = 0.78 * self.tracked_effector_y + 0.22 * (effector_y / effector_weight)
        else:
            action_direction = 0.0
            if self.previous_action_traces[0] > self.previous_action_traces[1] and self.previous_action_traces[0] > 0.1:
                action_direction = -1.0
            elif self.previous_action_traces[1] > 0.1:
                action_direction = 1.0
            self.tracked_effector_y = _clamp(self.tracked_effector_y + action_direction * 0.45, 0.0, self.input_h - 1)

        if target_weight > 0.0:
            self.tracked_target_x = 0.68 * self.tracked_target_x + 0.32 * (target_x / target_weight)
            self.tracked_target_y = 0.68 * self.tracked_target_y + 0.32 * (target_y / target_weight)
            mean_vx = target_vx / target_weight
            mean_vy = target_vy / target_weight
        else:
            mean_vx = 0.0
            mean_vy = 0.0

        predicted_y = _clamp(self.tracked_target_y + mean_vy * self.prediction_motion_horizon, 0.0, self.input_h - 1)
        error_y = predicted_y - self.tracked_effector_y
        confidence = _clamp((target_weight / 8.0) * (0.45 + 0.55 * self.tracked_target_x / max(1, self.input_w - 1)))
        feature_index = self._policy_feature_index(error_y, self.tracked_target_x)

        self.previous_active_cells = active_cells[:]
        self.last_event_features = {
            "targetX": self.tracked_target_x,
            "targetY": predicted_y,
            "targetVx": mean_vx,
            "targetVy": mean_vy,
            "effectorY": self.tracked_effector_y,
            "errorY": error_y,
            "confidence": confidence,
            "featureIndex": feature_index,
        }
        return self.last_event_features

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

    def _structured_hidden_spikes(self, active_cells, features):
        scores = {}
        for index in active_cells:
            x, y = self._cell_xy(index)
            hx = int((x + 0.5) * self.hidden_w / self.input_w)
            hy = int((y + 0.5) * self.hidden_h / self.input_h)
            hidden = self._hidden_index(hx, hy)
            scores[hidden] = max(scores.get(hidden, 0.0), 1.0)
            shifted = self._hidden_index(hx + round(self.cell_vx[index]), hy + round(self.cell_vy[index]))
            scores[shifted] = max(scores.get(shifted, 0.0), 0.78)

        for x, y, value in (
            (features["targetX"], features["targetY"], 0.92),
            (self.input_w - 1, features["effectorY"], 0.82),
        ):
            hx = int((x + 0.5) * self.hidden_w / self.input_w)
            hy = int((y + 0.5) * self.hidden_h / self.input_h)
            for dy in (-1, 0, 1):
                hidden = self._hidden_index(hx, hy + dy)
                scores[hidden] = max(scores.get(hidden, 0.0), value - abs(dy) * 0.12)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [index for index, _ in ranked[: self.structured_hidden_spikes]]

    def _hidden_spikes(self, drive, active_cells=None, features=None):
        candidates = []
        for index, incoming in enumerate(drive):
            value = self.hidden_potential[index] * 0.80 + incoming
            self.hidden_potential[index] = value
            if value >= 0.42:
                candidates.append((index, value))
        if len(candidates) > self.hidden_spike_cap:
            candidates = sorted(candidates, key=lambda item: item[1], reverse=True)[: self.hidden_spike_cap]
        spikes = [index for index, _ in candidates]
        if active_cells is not None and features is not None:
            spikes = sorted(set(spikes) | set(self._structured_hidden_spikes(active_cells, features)))
            if len(spikes) > self.hidden_spike_cap:
                spikes = sorted(spikes, key=lambda index: self.hidden_potential[index], reverse=True)[: self.hidden_spike_cap]
        for index in spikes:
            self.hidden_potential[index] *= 0.18
        return spikes

    def _policy_drives(self, features):
        drives = [0.0, 0.0, 0.08]
        confidence = float(features.get("confidence", 0.0))
        error_y = float(features.get("errorY", 0.0))
        if confidence > 0.02:
            if error_y < -self.visual_servo_deadband:
                drives[0] += min(1.0, abs(error_y) / max(1.0, self.input_h / 3)) * self.visual_servo_gain * confidence
            elif error_y > self.visual_servo_deadband:
                drives[1] += min(1.0, abs(error_y) / max(1.0, self.input_h / 3)) * self.visual_servo_gain * confidence
            else:
                drives[2] += 0.25 * confidence
        feature_index = int(features.get("featureIndex", 0))
        if 0 <= feature_index < len(self.policy_weights):
            for action, value in enumerate(self.policy_weights[feature_index]):
                drives[action] += value
        return drives

    def _output(self, hidden_spikes, features=None):
        drives = [0.0, 0.0, 0.035]
        for pre in hidden_spikes:
            for edge_index in self.hidden_output.by_pre[pre]:
                drives[self.hidden_output.post[edge_index]] += self.hidden_output.weight[edge_index]
        if features is not None:
            policy = self._policy_drives(features)
            drives = [drives[index] + policy[index] for index in range(self.motor_size)]
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

    def _motion_prediction(self, active_cells, features):
        scores = {}
        for index in active_cells:
            x, y = self._cell_xy(index)
            vx = self.cell_vx[index]
            vy = self.cell_vy[index]
            shifted = self._input_index(x + vx * self.prediction_motion_horizon, y + vy * self.prediction_motion_horizon)
            speed = min(1.0, (abs(vx) + abs(vy)) / 3.0)
            scores[shifted] = max(scores.get(shifted, 0.0), 0.36 + 0.42 * speed)
            if self.prediction_persistence > 0:
                scores[index] = max(scores.get(index, 0.0), self.prediction_persistence)

        if features.get("confidence", 0.0) > 0.05:
            target = self._input_index(
                features["targetX"] + features["targetVx"] * self.prediction_motion_horizon,
                features["targetY"],
            )
            scores[target] = max(scores.get(target, 0.0), 0.56 * features["confidence"])

        if len(scores) > self.prediction_max_cells:
            scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[: self.prediction_max_cells])
        return scores

    def _predict_next(self, hidden_spikes, active_cells=None, features=None):
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
            if probability >= self.prediction_min_probability:
                prediction[post] = probability
        if active_cells is not None and features is not None:
            for post, probability in self._motion_prediction(active_cells, features).items():
                if probability >= self.prediction_min_probability:
                    prediction[post] = max(prediction.get(post, 0.0), probability)
        self.last_prediction_edges = edge_count
        return prediction

    def _prediction_error(self, active_cells):
        actual = set(active_cells)
        predicted = {cell for cell, probability in self.last_prediction.items() if probability >= self.prediction_strong_threshold}
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

    def _trim_eligibilities(self):
        self.input_hidden.trim_active_edges(self.max_input_eligibility_edges)
        self.motor_hidden.trim_active_edges(self.max_motor_eligibility_edges)
        self.recurrent.trim_active_edges(self.max_recurrent_eligibility_edges)
        self.hidden_output.trim_active_edges(self.max_output_eligibility_edges)
        self.hidden_prediction.trim_active_edges(self.max_prediction_eligibility_edges)

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

    def _apply_learning(self, reward, predictive_signal, prediction_errors, local_changes, feature_index=None, winner_index=None):
        stats = self._empty_learning_stats()
        stats["step"] = self.train_steps + 1
        stats["rewardApplied"] = round(reward, 5)
        stats["predictionApplied"] = round(predictive_signal, 5)
        stats["combinedModulator"] = 0.0
        stats["eligibilityIncreased"] = local_changes["increased"]
        stats["eligibilityDecreased"] = local_changes["decreased"]

        self.hidden_prediction.apply_post_errors(self.prediction_learning_rate, prediction_errors, stats)
        self.input_hidden.apply_modulator(
            self.prediction_pathway_learning_rate,
            predictive_signal,
            stats,
            update_key="predictionWeightUpdates",
        )
        self.recurrent.apply_modulator(
            self.prediction_pathway_learning_rate,
            predictive_signal,
            stats,
            update_key="predictionWeightUpdates",
        )
        self.motor_hidden.apply_modulator(
            self.learning_rate,
            reward,
            stats,
            update_key="rewardWeightUpdates",
        )
        self.hidden_output.apply_modulator(
            self.learning_rate,
            reward,
            stats,
            update_key="rewardWeightUpdates",
        )
        self.recurrent.apply_modulator(
            self.reward_recurrent_learning_rate,
            reward,
            stats,
            update_key="rewardWeightUpdates",
            edge_filter=lambda index: (
                self.recurrent.pre[index] in self.motor_context_neurons
                or self.recurrent.post[index] in self.motor_context_neurons
            ),
        )
        if feature_index is not None and winner_index is not None:
            self._apply_policy_reward(reward, stats)

        if stats["weightUpdates"] > 0:
            stats["meanAbsDelta"] = round(stats["meanAbsDelta"] / stats["weightUpdates"], 8)
        else:
            stats["meanAbsDelta"] = 0.0
        return stats

    def _update_policy_traces(self, feature_index, winner_index, confidence):
        decayed = []
        for trace in self.policy_traces:
            value = trace["value"] * self.policy_trace_decay
            if abs(value) > 0.001:
                decayed.append({"feature": trace["feature"], "action": trace["action"], "value": value})
        if confidence > 0.02 and 0 <= feature_index < self.policy_feature_count:
            decayed.append({"feature": feature_index, "action": winner_index, "value": max(0.05, confidence)})
        self.policy_traces = decayed[-120:]

    def _apply_policy_reward(self, reward, stats):
        if reward == 0.0:
            return
        for row in self.policy_weights:
            for action in range(self.motor_size):
                row[action] *= self.policy_weight_decay
        for trace in self.policy_traces:
            feature = trace["feature"]
            action = trace["action"]
            if feature < 0 or feature >= self.policy_feature_count or action < 0 or action >= self.motor_size:
                continue
            before = self.policy_weights[feature][action]
            after = _clamp(before + self.policy_learning_rate * reward * trace["value"], -1.25, 1.25)
            if after == before:
                continue
            self.policy_weights[feature][action] = after
            stats["rewardWeightUpdates"] += 1
            stats["weightUpdates"] += 1
            stats["meanAbsDelta"] += abs(after - before)
            if after > before:
                stats["potentiated"] += 1
            else:
                stats["depressed"] += 1

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
        features = self._update_event_model(active_cells)

        drive = [0.0 for _ in range(self.hidden_size)]
        self._add_group_drive(drive, self.input_hidden, active_cells)
        self._add_motor_drive(drive)
        self._add_group_drive(drive, self.recurrent, self.last_hidden_spikes)
        hidden_spikes = self._hidden_spikes(drive, active_cells, features)
        output_drive, output_bars, winner_index = self._output(hidden_spikes, features)
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
            self._trim_eligibilities()
            self.learning_stats = self._apply_learning(
                reward,
                predictive_signal,
                prediction_errors,
                local_changes,
                feature_index=features["featureIndex"],
                winner_index=winner_index,
            )
            self._update_policy_traces(features["featureIndex"], winner_index, features["confidence"])
            if self.train_steps % self.status_summary_interval == 0:
                self.eligibility_stats = self._summarize_eligibility()
            changed["potentiated"] = self.learning_stats["potentiated"]
            changed["depressed"] = self.learning_stats["depressed"]
            changed["eligibilityIncreased"] = local_changes["increased"]
            changed["eligibilityDecreased"] = local_changes["decreased"]
            self.train_steps += 1
        else:
            self._prediction_error(active_cells)

        prediction = self._predict_next(hidden_spikes, active_cells, features)
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
            "eventFeatures": {
                "targetX": round(features["targetX"], 3),
                "targetY": round(features["targetY"], 3),
                "targetVx": round(features["targetVx"], 3),
                "targetVy": round(features["targetVy"], 3),
                "effectorY": round(features["effectorY"], 3),
                "errorY": round(features["errorY"], 3),
                "confidence": round(features["confidence"], 3),
                "featureIndex": features["featureIndex"],
            },
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
        architecture = dict(self.architecture_config)
        self.__init__(self.width, self.height, seed=seed, save_dir=save_dir, architecture=architecture)

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
            "architectureConfig": dict(self.architecture_config),
            "previousActionTraces": self.previous_action_traces,
            "policyWeights": self.policy_weights,
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

        policy_weights = payload.get("policyWeights")
        if isinstance(policy_weights, list):
            for feature, row in enumerate(policy_weights[: self.policy_feature_count]):
                if not isinstance(row, list):
                    continue
                for action, value in enumerate(row[: self.motor_size]):
                    self.policy_weights[feature][action] = _clamp(float(value), -1.25, 1.25)

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
