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
        self.input_eligibility = [0.0 for _ in range(self.input_size)]
        self.h1_h2_eligibility = self._build_eligibility(self.h1_h2_pre)
        self.h2_h3_eligibility = self._build_eligibility(self.h2_h3_pre)
        self.h3_out_eligibility = self._build_eligibility(self.h3_out_pre)
        self.learning_rate = 0.006
        self.eligibility_decay = 0.92
        self.eligibility_plus = 0.16
        self.eligibility_minus = 0.035
        self.last_pong_snapshot = None
        self.recent_hits = 0
        self.recent_misses = 0
        self.reward_state = self._empty_reward_state()
        self.eligibility_stats = self._empty_eligibility_stats()
        self.learning_stats = self._empty_learning_stats()

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

    def _build_eligibility(self, projection):
        return [[0.0 for _ in pres] for pres in projection]

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
            "components": {
                "alignment": 0.0,
                "hit": 0.0,
                "miss": 0.0,
                "movementPenalty": 0.0,
            },
            "rightPaddleDistance": None,
            "rightPaddleDelta": 0.0,
            "lastEvent": None,
            "recentHits": self.recent_hits,
            "recentMisses": self.recent_misses,
        }

    def _empty_eligibility_stats(self):
        return {
            "inputH1": self._empty_group_eligibility_stats("inputH1", self.input_size),
            "h1H2": self._empty_group_eligibility_stats("h1H2", sum(len(row) for row in self.h1_h2_pre)),
            "h2H3": self._empty_group_eligibility_stats("h2H3", sum(len(row) for row in self.h2_h3_pre)),
            "h3Output": self._empty_group_eligibility_stats("h3Output", sum(len(row) for row in self.h3_out_pre)),
        }

    def _empty_group_eligibility_stats(self, name, connections):
        return {
            "name": name,
            "connections": int(connections),
            "active": 0,
            "positive": 0,
            "negative": 0,
            "meanAbs": 0.0,
            "maxAbs": 0.0,
        }

    def _empty_learning_stats(self):
        return {
            "step": self.train_steps,
            "learningRate": self.learning_rate,
            "rewardApplied": 0.0,
            "weightUpdates": 0,
            "potentiated": 0,
            "depressed": 0,
            "clamped": 0,
            "meanAbsDelta": 0.0,
            "eligibilityIncreased": 0,
            "eligibilityDecreased": 0,
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
                "mode": "reward-modulated eligibility traces",
                "learningRate": self.learning_rate,
                "eligibilityDecay": self.eligibility_decay,
                "eligibilityPlus": self.eligibility_plus,
                "eligibilityMinus": self.eligibility_minus,
                "rule": "local pre/post coactivity updates eligibility; reward gates weight changes",
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

    def _clamp_eligibility(self, value):
        return max(-1.0, min(1.0, value))

    def _decay_flat_eligibility(self, eligibility):
        for index, value in enumerate(eligibility):
            if value == 0.0:
                continue
            decayed = value * self.eligibility_decay
            eligibility[index] = 0.0 if abs(decayed) < 0.00001 else decayed

    def _decay_nested_eligibility(self, eligibility):
        for row in eligibility:
            self._decay_flat_eligibility(row)

    def _decay_eligibilities(self):
        self._decay_flat_eligibility(self.input_eligibility)
        self._decay_nested_eligibility(self.h1_h2_eligibility)
        self._decay_nested_eligibility(self.h2_h3_eligibility)
        self._decay_nested_eligibility(self.h3_out_eligibility)

    def _eligibility_stats_flat(self, name, eligibility):
        active = 0
        positive = 0
        negative = 0
        total_abs = 0.0
        max_abs = 0.0
        for value in eligibility:
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
        connections = len(eligibility)
        return {
            "name": name,
            "connections": connections,
            "active": active,
            "positive": positive,
            "negative": negative,
            "meanAbs": round(total_abs / max(1, connections), 6),
            "maxAbs": round(max_abs, 6),
        }

    def _eligibility_stats_nested(self, name, eligibility):
        flat = [value for row in eligibility for value in row]
        return self._eligibility_stats_flat(name, flat)

    def _summarize_eligibility(self):
        return {
            "inputH1": self._eligibility_stats_flat("inputH1", self.input_eligibility),
            "h1H2": self._eligibility_stats_nested("h1H2", self.h1_h2_eligibility),
            "h2H3": self._eligibility_stats_nested("h2H3", self.h2_h3_eligibility),
            "h3Output": self._eligibility_stats_nested("h3Output", self.h3_out_eligibility),
        }

    def _eligibility_input(self, active_pixels, h1_spikes):
        h1_active = set(h1_spikes)
        changed = {"increased": 0, "decreased": 0}
        for pixel in active_pixels:
            y = pixel // self.width
            x = pixel - y * self.width
            hx = min(self.h1_w - 1, x // self.h1_cell)
            hy = min(self.h1_h - 1, y // self.h1_cell)
            post = _grid_index(hx, hy, self.h1_w)
            if post in h1_active:
                self.input_eligibility[pixel] = self._clamp_eligibility(
                    self.input_eligibility[pixel] + self.eligibility_plus
                )
                changed["increased"] += 1
            else:
                self.input_eligibility[pixel] = self._clamp_eligibility(
                    self.input_eligibility[pixel] - self.eligibility_minus
                )
                changed["decreased"] += 1
        return changed

    def _eligibility_projection(self, pre_spikes, post_spikes, projection, eligibility):
        pre_active = set(pre_spikes)
        post_active = set(post_spikes)
        changed = {"increased": 0, "decreased": 0}
        if not pre_active:
            return changed
        for post, (pres, post_eligibility) in enumerate(zip(projection, eligibility)):
            active_post = post in post_active
            for offset, pre in enumerate(pres):
                if pre not in pre_active:
                    continue
                if active_post:
                    post_eligibility[offset] = self._clamp_eligibility(
                        post_eligibility[offset] + self.eligibility_plus
                    )
                    changed["increased"] += 1
                else:
                    post_eligibility[offset] = self._clamp_eligibility(
                        post_eligibility[offset] - self.eligibility_minus
                    )
                    changed["decreased"] += 1
        return changed

    def _eligibility_output(self, h3_spikes, winner_index):
        active = set(h3_spikes)
        changed = {"increased": 0, "decreased": 0}
        if not active:
            return changed
        for output_index, (pres, eligibility) in enumerate(zip(self.h3_out_pre, self.h3_out_eligibility)):
            for offset, pre in enumerate(pres):
                if pre not in active:
                    continue
                if output_index == winner_index:
                    eligibility[offset] = self._clamp_eligibility(eligibility[offset] + self.eligibility_plus)
                    changed["increased"] += 1
                else:
                    eligibility[offset] = self._clamp_eligibility(eligibility[offset] - self.eligibility_minus)
                    changed["decreased"] += 1
        return changed

    def _apply_flat_reward(self, weights, eligibility, reward, stats):
        if reward == 0.0:
            return
        for index, trace in enumerate(eligibility):
            if abs(trace) <= 0.00001:
                continue
            before = weights[index]
            delta = self.learning_rate * reward * trace
            after = _clamp(before + delta)
            if after == before:
                continue
            weights[index] = after
            stats["weightUpdates"] += 1
            stats["meanAbsDelta"] += abs(after - before)
            if after == 0.0 or after == 1.0:
                stats["clamped"] += 1
            if after > before:
                stats["potentiated"] += 1
            else:
                stats["depressed"] += 1

    def _apply_nested_reward(self, weights, eligibility, reward, stats):
        for weight_row, eligibility_row in zip(weights, eligibility):
            self._apply_flat_reward(weight_row, eligibility_row, reward, stats)

    def _apply_rewarded_weights(self, reward, local_changes):
        stats = self._empty_learning_stats()
        stats["step"] = self.train_steps + 1
        stats["rewardApplied"] = round(reward, 5)
        stats["eligibilityIncreased"] = local_changes["increased"]
        stats["eligibilityDecreased"] = local_changes["decreased"]
        self._apply_flat_reward(self.input_weights, self.input_eligibility, reward, stats)
        self._apply_nested_reward(self.h1_h2_weights, self.h1_h2_eligibility, reward, stats)
        self._apply_nested_reward(self.h2_h3_weights, self.h2_h3_eligibility, reward, stats)
        self._apply_nested_reward(self.h3_out_weights, self.h3_out_eligibility, reward, stats)
        if stats["weightUpdates"] > 0:
            stats["meanAbsDelta"] = round(stats["meanAbsDelta"] / stats["weightUpdates"], 8)
        else:
            stats["meanAbsDelta"] = 0.0
        return stats

    def _snapshot_pong(self, game_state):
        if not isinstance(game_state, dict):
            return None
        settings = game_state.get("settings") if isinstance(game_state.get("settings"), dict) else {}
        ball = game_state.get("ball") if isinstance(game_state.get("ball"), dict) else {}
        paddles = game_state.get("paddles") if isinstance(game_state.get("paddles"), dict) else {}
        score = game_state.get("score") if isinstance(game_state.get("score"), dict) else {}
        try:
            height = float(settings.get("height", self.height))
            width = float(settings.get("width", self.width))
            paddle_height = float(settings.get("paddleHeight", 80))
            ball_size = float(settings.get("ballSize", 10))
            ball_x = float(ball.get("x", width / 2))
            ball_y = float(ball.get("y", height / 2))
            ball_vx = float(ball.get("vx", 0.0))
            right_y = float(paddles.get("rightY", (height - paddle_height) / 2))
            left_score = int(score.get("left", 0))
            right_score = int(score.get("right", 0))
        except (TypeError, ValueError):
            return None
        ball_center = ball_y + ball_size / 2
        right_center = right_y + paddle_height / 2
        return {
            "width": width,
            "height": height,
            "paddleHeight": paddle_height,
            "ballSize": ball_size,
            "ballX": ball_x,
            "ballY": ball_y,
            "ballVx": ball_vx,
            "rightY": right_y,
            "rightCenter": right_center,
            "ballCenter": ball_center,
            "rightDistance": abs(right_center - ball_center),
            "leftScore": left_score,
            "rightScore": right_score,
        }

    def _reward_from_pong(self, game_state):
        current = self._snapshot_pong(game_state)
        if current is None:
            self.reward_state = self._empty_reward_state()
            return 0.0

        previous = self.last_pong_snapshot
        self.last_pong_snapshot = current
        if previous is None:
            self.reward_state = {
                **self._empty_reward_state(),
                "rightPaddleDistance": round(current["rightDistance"], 3),
                "recentHits": self.recent_hits,
                "recentMisses": self.recent_misses,
            }
            return 0.0

        movement = abs(current["rightY"] - previous["rightY"])
        distance_delta = previous["rightDistance"] - current["rightDistance"]
        alignment_reward = 0.0
        if movement > 0.01:
            scale = max(1.0, current["paddleHeight"])
            alignment_reward = _clamp(distance_delta / scale, -1.0, 1.0) * 0.08

        movement_penalty = 0.0
        aligned = current["rightDistance"] <= max(current["ballSize"] * 2.0, current["paddleHeight"] * 0.16)
        if aligned and movement > 0.01:
            movement_penalty = -0.012

        hit_reward = 0.0
        miss_reward = 0.0
        last_event = None
        right_side = current["ballX"] > current["width"] * 0.65
        if previous["ballVx"] > 0.0 and current["ballVx"] < 0.0 and right_side:
            hit_reward = 1.0
            self.recent_hits += 1
            last_event = "right-paddle-hit"
        if current["leftScore"] > previous["leftScore"]:
            miss_reward = -1.0
            self.recent_misses += 1
            last_event = "right-paddle-miss"

        total = alignment_reward + movement_penalty + hit_reward + miss_reward
        self.reward_state = {
            "value": round(total, 5),
            "components": {
                "alignment": round(alignment_reward, 5),
                "hit": round(hit_reward, 5),
                "miss": round(miss_reward, 5),
                "movementPenalty": round(movement_penalty, 5),
            },
            "rightPaddleDistance": round(current["rightDistance"], 3),
            "rightPaddleDelta": round(distance_delta, 3),
            "lastEvent": last_event,
            "recentHits": self.recent_hits,
            "recentMisses": self.recent_misses,
        }
        return total

    def step(self, event_camera, tick, game_state=None):
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

        changed = {
            "potentiated": 0,
            "depressed": 0,
            "eligibilityIncreased": 0,
            "eligibilityDecreased": 0,
        }
        if self.training and not self.paused:
            reward = self._reward_from_pong(game_state)
            self._decay_eligibilities()
            local_changes = {"increased": 0, "decreased": 0}
            for update in (
                self._eligibility_input(active_pixels, h1_spikes),
                self._eligibility_projection(h1_spikes, h2_spikes, self.h1_h2_pre, self.h1_h2_eligibility),
                self._eligibility_projection(h2_spikes, h3_spikes, self.h2_h3_pre, self.h2_h3_eligibility),
                self._eligibility_output(h3_spikes, winner_index),
            ):
                local_changes["increased"] += update["increased"]
                local_changes["decreased"] += update["decreased"]
            self.eligibility_stats = self._summarize_eligibility()
            self.learning_stats = self._apply_rewarded_weights(reward, local_changes)
            changed["potentiated"] = self.learning_stats["potentiated"]
            changed["depressed"] = self.learning_stats["depressed"]
            changed["eligibilityIncreased"] = local_changes["increased"]
            changed["eligibilityDecreased"] = local_changes["decreased"]
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
