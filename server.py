from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
import argparse
import copy
import json
import mimetypes
import random
import threading
import time

from snn_backend import PongSNN


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
NETWORK_SAVE_ROOT = ROOT / "network_saves"

SETTINGS = {
    "width": 800,
    "height": 450,
    "paddleWidth": 10,
    "paddleHeight": 80,
    "ballSize": 10,
    "pollMs": 16,
    "statePushMs": 16,
    "fixedDt": 1 / 60,
}

STATE = {
    "sessionId": "local-1",
    "resetToken": 0,
    "seed": int(time.time() * 1000) & 0xFFFFFFFF,
    "mode": "api",
    "apiDirection": 0,
    "authoritativeTick": 0,
    "eventSeq": 0,
    "events": [],
    "inputSeq": 0,
    "inputEvents": [],
    "score": {"left": 0, "right": 0},
    "running": False,
    "ball": {"x": SETTINGS["width"] / 2, "y": SETTINGS["height"] / 2, "vx": 0, "vy": 0},
    "paddles": {
        "leftY": (SETTINGS["height"] - SETTINGS["paddleHeight"]) / 2,
        "rightY": (SETTINGS["height"] - SETTINGS["paddleHeight"]) / 2,
    },
    "appliedInputSeq": 0,
    "appliedEventSeq": 0,
    "frameSeq": 0,
    "renderedAt": 0,
    "eventCamera": None,
    "source": "backend",
    "updatedAt": time.time(),
}

STATE_LOCK = threading.Lock()
SNN = PongSNN(SETTINGS["width"], SETTINGS["height"], save_dir=NETWORK_SAVE_ROOT)
SIM_RNG = random.Random(STATE["seed"])
EVENT_CAMERA_PREVIOUS = None
EVENT_CAMERA_TOKEN = None
SIM_THREAD = None
SIM_THREAD_LOCK = threading.Lock()


def clamp(value, low, high):
    return max(low, min(high, value))


def parse_direction(value):
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        raise ValueError("direction must be -1, 0, or 1")
    if numeric not in (-1, 0, 1):
        raise ValueError("direction must be -1, 0, or 1")
    return numeric


def parse_tick(value, fallback):
    if value is None:
        return fallback
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        raise ValueError("tick must be a non-negative integer")
    if numeric < 0:
        raise ValueError("tick must be a non-negative integer")
    return numeric


def read_json(handler):
    length = int(handler.headers.get("Content-Length", "0"))
    if length == 0:
        return {}
    try:
        return json.loads(handler.rfile.read(length).decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("request body must be valid JSON") from exc


def session_payload():
    return {
        "sessionId": STATE["sessionId"],
        "resetToken": STATE["resetToken"],
        "seed": STATE["seed"],
        "tick": STATE["authoritativeTick"],
        "authoritativeTick": STATE["authoritativeTick"],
        "mode": STATE["mode"],
        "running": STATE["running"],
        "apiDirection": STATE["apiDirection"],
        "latestEventSeq": STATE["eventSeq"],
        "latestInputSeq": STATE["inputSeq"],
        "settings": SETTINGS,
        "snn": {
            "training": SNN.training,
            "paused": SNN.paused,
            "winner": SNN.activity["winner"],
            "direction": SNN.activity["direction"],
        },
    }


def normalize_event_camera(value):
    if not isinstance(value, dict):
        return None
    try:
        width = int(value.get("width"))
        height = int(value.get("height"))
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    max_index = width * height
    pixels = []
    for pixel in value.get("pixels", []):
        try:
            numeric = int(pixel)
        except (TypeError, ValueError):
            continue
        if 0 <= numeric < max_index:
            pixels.append(numeric)
    return {
        "width": width,
        "height": height,
        "tick": int(value.get("tick", STATE["authoritativeTick"]) or 0),
        "frameSeq": int(value.get("frameSeq", STATE["frameSeq"]) or 0),
        "resetToken": value.get("resetToken"),
        "renderedAt": float(value.get("renderedAt", STATE["renderedAt"]) or 0),
        "source": value.get("source") if isinstance(value.get("source"), str) else "game",
        "pixels": pixels,
        "count": len(pixels),
    }


def compact_events():
    min_tick = max(0, STATE["authoritativeTick"] - 1200)
    if len(STATE["events"]) > 2500:
        STATE["events"] = [event for event in STATE["events"] if event["tick"] >= min_tick]
        STATE["inputEvents"] = [event for event in STATE["events"] if event["type"] == "input"]


def add_session_event(body, default_type="input"):
    event_type = body.get("type", default_type)
    if event_type not in ("start", "input", "pause"):
        raise ValueError('type must be "start", "input", or "pause"')
    tick = parse_tick(body.get("tick"), STATE["authoritativeTick"] + 1)
    event = {
        "seq": STATE["eventSeq"] + 1,
        "type": event_type,
        "tick": tick,
        "sessionId": STATE["sessionId"],
        "receivedAt": time.time(),
    }
    if event_type == "input":
        event["direction"] = parse_direction(body.get("direction"))
        event["source"] = "human" if body.get("source") == "human" else "api"
        STATE["apiDirection"] = event["direction"]
        STATE["inputSeq"] = event["seq"]
    STATE["eventSeq"] = event["seq"]
    STATE["events"].append(event)
    STATE["events"].sort(key=lambda item: (item["tick"], item["seq"]))
    STATE["inputEvents"] = [item for item in STATE["events"] if item["type"] == "input"]
    if event_type == "start":
        STATE["running"] = True
    elif event_type == "pause":
        STATE["running"] = False
    STATE["updatedAt"] = time.time()
    compact_events()
    return event


def apply_snn_to_state():
    if not STATE.get("eventCamera"):
        return None
    if not SNN.training or SNN.paused:
        return None
    direction = SNN.step(STATE["eventCamera"], STATE["authoritativeTick"])
    if direction is None:
        return None
    if not SNN.should_emit_command(direction, STATE["authoritativeTick"]):
        return None
    return add_session_event(
        {
            "type": "input",
            "tick": STATE["authoritativeTick"] + 1,
            "direction": direction,
            "source": "api",
        },
        "input",
    )


def reset_ball(direction):
    angle = SIM_RNG.random() * 0.7 - 0.35
    speed = float(STATE["ball"].get("speed", 275))
    STATE["ball"] = {
        "x": SETTINGS["width"] / 2,
        "y": SETTINGS["height"] / 2,
        "vx": math_cos(angle) * speed * direction,
        "vy": math_sin(angle) * speed,
        "speed": speed,
    }


def math_sin(value):
    import math

    return math.sin(value)


def math_cos(value):
    import math

    return math.cos(value)


def start_simulation():
    if not STATE["running"]:
        STATE["running"] = True
    if float(STATE["ball"].get("vx", 0)) == 0 and float(STATE["ball"].get("vy", 0)) == 0:
        reset_ball(-1 if SIM_RNG.random() < 0.5 else 1)


def apply_due_events():
    for event in sorted(STATE["events"], key=lambda item: (item["tick"], item["seq"])):
        seq = int(event.get("seq", 0))
        if seq <= STATE["appliedEventSeq"] or int(event.get("tick", 0)) > STATE["authoritativeTick"]:
            continue
        event_type = event.get("type", "input")
        if event_type == "start":
            start_simulation()
        elif event_type == "pause":
            STATE["running"] = False
        elif event_type == "input":
            STATE["apiDirection"] = parse_direction(event.get("direction"))
            STATE["appliedInputSeq"] = max(STATE["appliedInputSeq"], seq)
        STATE["appliedEventSeq"] = max(STATE["appliedEventSeq"], seq)


def update_paddles(dt):
    ball = STATE["ball"]
    paddles = STATE["paddles"]
    target = float(ball["y"]) - SETTINGS["paddleHeight"] / 2
    left_start = float(paddles["leftY"])
    left_delta = clamp(target - left_start, -320 * dt, 320 * dt)
    left_y = clamp(left_start + left_delta, 0, SETTINGS["height"] - SETTINGS["paddleHeight"])
    paddles["leftY"] = left_y
    paddles["leftVy"] = (left_y - left_start) / dt if dt > 0 else 0

    right_start = float(paddles["rightY"])
    right_y = clamp(
        right_start + int(STATE["apiDirection"]) * 360 * dt,
        0,
        SETTINGS["height"] - SETTINGS["paddleHeight"],
    )
    paddles["rightY"] = right_y
    paddles["rightVy"] = (right_y - right_start) / dt if dt > 0 else 0


def paddle_hit(x, y):
    ball = STATE["ball"]
    return (
        float(ball["x"]) < x + SETTINGS["paddleWidth"]
        and float(ball["x"]) + SETTINGS["ballSize"] > x
        and float(ball["y"]) < y + SETTINGS["paddleHeight"]
        and float(ball["y"]) + SETTINGS["ballSize"] > y
    )


def bounce_from_paddle(paddle_y, paddle_vy, direction):
    import math

    ball = STATE["ball"]
    ball_center = float(ball["y"]) + SETTINGS["ballSize"] / 2
    paddle_center = paddle_y + SETTINGS["paddleHeight"] / 2
    normalized = clamp((ball_center - paddle_center) / (SETTINGS["paddleHeight"] / 2), -1, 1)
    speed = min(float(ball.get("speed", 275)) + 8, 520)
    incoming_vy = float(ball["vy"])
    max_vertical = speed * 0.88
    ball["speed"] = speed
    ball["vy"] = clamp(incoming_vy * 0.85 + paddle_vy * 0.45 + normalized * speed * 0.18, -max_vertical, max_vertical)
    ball["vx"] = direction * math.sqrt(max(0, speed**2 - float(ball["vy"]) ** 2))


def update_ball(dt):
    ball = STATE["ball"]
    ball["x"] = float(ball["x"]) + float(ball["vx"]) * dt
    ball["y"] = float(ball["y"]) + float(ball["vy"]) * dt

    if ball["y"] <= 0:
        ball["y"] = 0
        ball["vy"] = abs(float(ball["vy"]))
    elif ball["y"] + SETTINGS["ballSize"] >= SETTINGS["height"]:
        ball["y"] = SETTINGS["height"] - SETTINGS["ballSize"]
        ball["vy"] = -abs(float(ball["vy"]))

    left_x = 24
    right_x = SETTINGS["width"] - 34
    if float(ball["vx"]) < 0 and paddle_hit(left_x, float(STATE["paddles"]["leftY"])):
        ball["x"] = left_x + SETTINGS["paddleWidth"]
        bounce_from_paddle(float(STATE["paddles"]["leftY"]), float(STATE["paddles"].get("leftVy", 0)), 1)
    elif float(ball["vx"]) > 0 and paddle_hit(right_x, float(STATE["paddles"]["rightY"])):
        ball["x"] = right_x - SETTINGS["ballSize"]
        bounce_from_paddle(float(STATE["paddles"]["rightY"]), float(STATE["paddles"].get("rightVy", 0)), -1)

    if ball["x"] + SETTINGS["ballSize"] < 0:
        STATE["score"]["right"] += 1
        ball["speed"] = 275
        reset_ball(1)
    elif ball["x"] > SETTINGS["width"]:
        STATE["score"]["left"] += 1
        ball["speed"] = 275
        reset_ball(-1)


def rasterize_state():
    width = SETTINGS["width"]
    height = SETTINGS["height"]
    mask = bytearray(width * height)

    def mark_rect(x, y, rect_width, rect_height):
        left = int(clamp(int(x), 0, width))
        right = int(clamp(int(x + rect_width + 0.999), 0, width))
        top = int(clamp(int(y), 0, height))
        bottom = int(clamp(int(y + rect_height + 0.999), 0, height))
        if left >= right or top >= bottom:
            return
        for row in range(top, bottom):
            start = row * width + left
            mask[start : row * width + right] = b"\x01" * (right - left)

    for y in range(0, SETTINGS["height"], 24):
        mark_rect(SETTINGS["width"] / 2 - 1, y, 2, 12)
    mark_rect(24, float(STATE["paddles"]["leftY"]), SETTINGS["paddleWidth"], SETTINGS["paddleHeight"])
    mark_rect(SETTINGS["width"] - 34, float(STATE["paddles"]["rightY"]), SETTINGS["paddleWidth"], SETTINGS["paddleHeight"])
    mark_rect(float(STATE["ball"]["x"]), float(STATE["ball"]["y"]), SETTINGS["ballSize"], SETTINGS["ballSize"])
    return mask


def capture_event_frame():
    global EVENT_CAMERA_PREVIOUS, EVENT_CAMERA_TOKEN

    mask = rasterize_state()
    pixels = []
    changed_baseline = EVENT_CAMERA_PREVIOUS is None or EVENT_CAMERA_TOKEN != STATE["resetToken"]
    if not changed_baseline:
        pixels = [index for index, value in enumerate(mask) if value != EVENT_CAMERA_PREVIOUS[index]]
    EVENT_CAMERA_PREVIOUS = mask
    EVENT_CAMERA_TOKEN = STATE["resetToken"]
    return {
        "width": SETTINGS["width"],
        "height": SETTINGS["height"],
        "tick": STATE["authoritativeTick"],
        "frameSeq": STATE["frameSeq"],
        "resetToken": STATE["resetToken"],
        "renderedAt": STATE["renderedAt"],
        "source": "backend",
        "pixels": pixels,
        "count": len(pixels),
    }


def backend_step():
    apply_due_events()
    update_paddles(SETTINGS["fixedDt"])
    if STATE["running"]:
        update_ball(SETTINGS["fixedDt"])
    STATE["authoritativeTick"] += 1
    STATE["frameSeq"] += 1
    STATE["renderedAt"] = time.time() * 1000
    STATE["eventCamera"] = capture_event_frame()
    apply_snn_to_state()
    STATE["source"] = "backend-sim"
    STATE["updatedAt"] = time.time()
    compact_events()


def backend_loop():
    next_frame = time.perf_counter()
    while True:
        with STATE_LOCK:
            backend_step()
        next_frame += SETTINGS["fixedDt"]
        delay = next_frame - time.perf_counter()
        if delay < -0.25:
            next_frame = time.perf_counter()
            delay = SETTINGS["fixedDt"]
        time.sleep(max(0.001, delay))


def ensure_backend_loop():
    global SIM_THREAD

    with SIM_THREAD_LOCK:
        if SIM_THREAD and SIM_THREAD.is_alive():
            return
        SIM_THREAD = threading.Thread(target=backend_loop, daemon=True, name="pong-backend-sim")
        SIM_THREAD.start()


def reset_game_state(reset_score=True):
    global SIM_RNG, EVENT_CAMERA_PREVIOUS, EVENT_CAMERA_TOKEN

    STATE["sessionId"] = f"local-{int(time.time() * 1000):x}"
    STATE["resetToken"] += 1
    STATE["seed"] = int(time.time() * 1000) & 0xFFFFFFFF
    SIM_RNG = random.Random(STATE["seed"])
    EVENT_CAMERA_PREVIOUS = None
    EVENT_CAMERA_TOKEN = None
    STATE["authoritativeTick"] = 0
    STATE["eventSeq"] = 0
    STATE["events"] = []
    STATE["inputSeq"] = 0
    STATE["inputEvents"] = []
    STATE["apiDirection"] = 0
    STATE["running"] = False
    if reset_score:
        STATE["score"] = {"left": 0, "right": 0}
    STATE["ball"] = {"x": SETTINGS["width"] / 2, "y": SETTINGS["height"] / 2, "vx": 0, "vy": 0, "speed": 275}
    STATE["paddles"] = {
        "leftY": (SETTINGS["height"] - SETTINGS["paddleHeight"]) / 2,
        "rightY": (SETTINGS["height"] - SETTINGS["paddleHeight"]) / 2,
    }
    STATE["appliedInputSeq"] = 0
    STATE["appliedEventSeq"] = 0
    STATE["frameSeq"] = 0
    STATE["renderedAt"] = 0
    STATE["eventCamera"] = None
    STATE["source"] = "backend"
    STATE["updatedAt"] = time.time()


class PongHandler(BaseHTTPRequestHandler):
    server_version = "PongSNN/0.2"

    def log_message(self, fmt, *args):
        print("%s - - [%s] %s" % (self.address_string(), self.log_date_time_string(), fmt % args))

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        ensure_backend_loop()
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/" or path == "/index.html":
            self.serve_file(WEB_ROOT / "index.html")
            return
        if path == "/visualizer" or path == "/visualizer.html":
            self.serve_file(WEB_ROOT / "visualizer.html")
            return
        if path == "/api/config" or path == "/api/session":
            with STATE_LOCK:
                payload = session_payload()
            self.send_json(payload)
            return
        if path == "/api/inputs":
            query = parse_qs(parsed.query)
            has_since_seq = "sinceSeq" in query
            since_seq = int(query.get("sinceSeq", ["0"])[0])
            since_tick = int(query.get("sinceTick", ["-1"])[0])
            with STATE_LOCK:
                if has_since_seq:
                    events = [event for event in STATE["inputEvents"] if event["seq"] > since_seq]
                else:
                    events = [event for event in STATE["inputEvents"] if event["tick"] > since_tick]
                payload = {**session_payload(), "events": copy.deepcopy(events)}
            self.send_json(payload)
            return
        if path == "/api/events":
            query = parse_qs(parsed.query)
            has_since_seq = "sinceSeq" in query
            since_seq = int(query.get("sinceSeq", ["0"])[0])
            since_tick = int(query.get("sinceTick", ["-1"])[0])
            with STATE_LOCK:
                if has_since_seq:
                    events = [event for event in STATE["events"] if event["seq"] > since_seq]
                else:
                    events = [event for event in STATE["events"] if event["tick"] > since_tick]
                payload = {**session_payload(), "events": copy.deepcopy(events)}
            self.send_json(payload)
            return
        if path == "/api/state":
            with STATE_LOCK:
                payload = {**copy.deepcopy(STATE), "tick": STATE["authoritativeTick"], "settings": SETTINGS}
            self.send_json(payload)
            return
        if path == "/api/snn/status":
            with STATE_LOCK:
                payload = SNN.status()
            self.send_json(payload)
            return
        if path == "/api/snn/saves":
            with STATE_LOCK:
                payload = {"saves": SNN.list_saves()}
            self.send_json(payload)
            return
        if path.startswith("/web/"):
            requested = (WEB_ROOT / unquote(path.removeprefix("/web/"))).resolve()
            if WEB_ROOT in requested.parents or requested == WEB_ROOT:
                self.serve_file(requested)
                return
        self.send_error(404, "Not found")

    def do_POST(self):
        ensure_backend_loop()
        try:
            body = read_json(self)
            path = urlparse(self.path).path
            if path == "/api/control-mode":
                self.set_control_mode(body)
                return
            if path == "/api/input":
                self.set_api_input(body)
                return
            if path == "/api/start":
                self.add_event({**body, "type": "start"}, "start")
                return
            if path == "/api/event":
                self.add_event(body)
                return
            if path == "/api/state":
                self.update_game_state(body)
                return
            if path == "/api/reset":
                self.reset_game(body)
                return
            if path == "/api/snn/start":
                self.start_snn(body)
                return
            if path == "/api/snn/pause":
                self.pause_snn(body)
                return
            if path == "/api/snn/reset":
                self.reset_snn(body)
                return
            if path == "/api/snn/save":
                self.save_snn(body)
                return
            if path == "/api/snn/load":
                self.load_snn(body)
                return
            self.send_error(404, "Not found")
        except ValueError as exc:
            self.send_json({"error": str(exc)}, status=400)

    def set_control_mode(self, body):
        mode = body.get("mode")
        if mode not in ("human", "api"):
            raise ValueError('mode must be "human" or "api"')
        with STATE_LOCK:
            STATE["mode"] = mode
            STATE["updatedAt"] = time.time()
            payload = session_payload()
        self.send_json(payload)

    def set_api_input(self, body):
        self.add_event({**body, "type": "input"}, "input")

    def add_event(self, body, default_type="input"):
        with STATE_LOCK:
            event = add_session_event(body, default_type)
            payload = {**session_payload(), "event": copy.deepcopy(event)}
        self.send_json(payload)

    def update_game_state(self, body):
        with STATE_LOCK:
            if "tick" in body:
                STATE["authoritativeTick"] = max(STATE["authoritativeTick"], int(body["tick"]))
            if "score" in body:
                score = body["score"]
                STATE["score"] = {
                    "left": int(score.get("left", STATE["score"]["left"])),
                    "right": int(score.get("right", STATE["score"]["right"])),
                }
            if "running" in body:
                STATE["running"] = bool(body["running"])
            if "ball" in body:
                ball = body["ball"]
                STATE["ball"] = {
                    "x": float(ball.get("x", STATE["ball"]["x"])),
                    "y": float(ball.get("y", STATE["ball"]["y"])),
                    "vx": float(ball.get("vx", STATE["ball"]["vx"])),
                    "vy": float(ball.get("vy", STATE["ball"]["vy"])),
                }
            if "paddles" in body:
                paddles = body["paddles"]
                STATE["paddles"] = {
                    "leftY": float(paddles.get("leftY", STATE["paddles"]["leftY"])),
                    "rightY": float(paddles.get("rightY", STATE["paddles"]["rightY"])),
                }
            if "appliedInputSeq" in body:
                STATE["appliedInputSeq"] = max(STATE["appliedInputSeq"], int(body["appliedInputSeq"]))
            if "appliedEventSeq" in body:
                STATE["appliedEventSeq"] = max(STATE["appliedEventSeq"], int(body["appliedEventSeq"]))
            if "frameSeq" in body:
                STATE["frameSeq"] = max(STATE["frameSeq"], int(body["frameSeq"]))
            if "renderedAt" in body:
                STATE["renderedAt"] = float(body["renderedAt"])
            if "eventCamera" in body:
                STATE["eventCamera"] = normalize_event_camera(body["eventCamera"])
            if isinstance(body.get("source"), str):
                STATE["source"] = body["source"]
            snn_event = apply_snn_to_state()
            STATE["updatedAt"] = time.time()
            compact_events()
            payload = {
                **copy.deepcopy(STATE),
                "tick": STATE["authoritativeTick"],
                "settings": SETTINGS,
                "snnEvent": copy.deepcopy(snn_event),
            }
        self.send_json(payload)

    def reset_game(self, body):
        with STATE_LOCK:
            reset_game_state(body.get("resetScore", True) is not False)
            payload = {**copy.deepcopy(STATE), "tick": STATE["authoritativeTick"], "settings": SETTINGS}
        self.send_json(payload)

    def start_snn(self, body):
        with STATE_LOCK:
            STATE["mode"] = "api"
            SNN.start()
            start_event = add_session_event(
                {"type": "start", "tick": parse_tick(body.get("tick"), STATE["authoritativeTick"] + 1)},
                "start",
            )
            payload = {**session_payload(), "event": copy.deepcopy(start_event), "snn": SNN.status()}
        self.send_json(payload)

    def pause_snn(self, body):
        with STATE_LOCK:
            SNN.pause()
            pause_event = add_session_event(
                {"type": "pause", "tick": parse_tick(body.get("tick"), STATE["authoritativeTick"] + 1)},
                "pause",
            )
            payload = {**session_payload(), "event": copy.deepcopy(pause_event), "snn": SNN.status()}
        self.send_json(payload)

    def reset_snn(self, body):
        with STATE_LOCK:
            SNN.reset(reset_weights=body.get("resetWeights", True) is not False)
            reset_game_state(body.get("resetScore", True) is not False)
            payload = {**copy.deepcopy(STATE), "tick": STATE["authoritativeTick"], "settings": SETTINGS, "snn": SNN.status()}
        self.send_json(payload)

    def save_snn(self, body):
        with STATE_LOCK:
            name = SNN.save(body.get("name"))
            payload = {"saved": name, "saves": SNN.list_saves(), "snn": SNN.status()}
        self.send_json(payload)

    def load_snn(self, body):
        name = body.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("name is required")
        with STATE_LOCK:
            loaded = SNN.load(name)
            payload = {"loaded": loaded, "saves": SNN.list_saves(), "snn": SNN.status()}
        self.send_json(payload)

    def serve_file(self, path):
        if not path.exists() or not path.is_file():
            self.send_error(404, "Not found")
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload, status=200):
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description="Run the local Pong SNN game server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    ensure_backend_loop()
    server = ThreadingHTTPServer((args.host, args.port), PongHandler)
    print(f"Pong SNN server listening at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
