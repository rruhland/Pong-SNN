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
    "pollMs": 120,
    "statePushMs": 16,
    "eventPollMs": 8,
    "fixedDt": 1 / 60,
    "simulationSpeed": 1.0,
    "minSimulationSpeed": 0.25,
    "maxSimulationSpeed": 4.0,
    "snnStepHz": 60,
}


def initial_world_state():
    return {
        "sessionId": STATE["sessionId"],
        "resetToken": STATE["resetToken"],
        "seed": STATE["seed"],
        "tick": 0,
        "authoritativeTick": 0,
        "frameSeq": 0,
        "renderedAt": 0,
        "running": False,
        "settings": copy.deepcopy(SETTINGS),
        "score": {"left": 0, "right": 0},
        "ball": {
            "x": SETTINGS["width"] / 2,
            "y": SETTINGS["height"] / 2,
            "vx": 0,
            "vy": 0,
        },
        "paddles": {
            "leftY": (SETTINGS["height"] - SETTINGS["paddleHeight"]) / 2,
            "rightY": (SETTINGS["height"] - SETTINGS["paddleHeight"]) / 2,
        },
        "appliedInputSeq": 0,
        "appliedEventSeq": 0,
        "eventCamera": None,
        "source": "waiting-for-world",
        "updatedAt": time.time(),
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
    "running": False,
    "latestWorld": None,
    "latestAction": None,
    "worldFrameSeq": 0,
    "worldObservedAt": 0,
    "updatedAt": time.time(),
}
STATE["latestWorld"] = initial_world_state()

STATE_LOCK = threading.Lock()
SNN_LOCK = threading.Lock()
SNN = PongSNN(SETTINGS["width"], SETTINGS["height"], save_dir=NETWORK_SAVE_ROOT)
SNN_THREAD = None
SNN_THREAD_LOCK = threading.Lock()
SNN_RUNTIME = {
    "lastFrameSeq": -1,
    "lastStepAt": 0.0,
    "samples": 0,
    "skippedFrames": 0,
}


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


def parse_simulation_speed(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError("speed must be a number")
    return clamp(numeric, SETTINGS["minSimulationSpeed"], SETTINGS["maxSimulationSpeed"])


def read_json(handler):
    length = int(handler.headers.get("Content-Length", "0"))
    if length == 0:
        return {}
    try:
        return json.loads(handler.rfile.read(length).decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("request body must be valid JSON") from exc


def session_payload():
    latest_world = STATE["latestWorld"] or {}
    latest_action = STATE["latestAction"] or {
        "seq": 0,
        "direction": STATE["apiDirection"],
        "tick": STATE["authoritativeTick"],
        "source": "hold",
        "sessionId": STATE["sessionId"],
        "emittedAt": 0,
    }
    with SNN_LOCK:
        activity = copy.deepcopy(SNN.activity)
        training = SNN.training
        paused = SNN.paused
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
        "latestWorldFrameSeq": STATE["worldFrameSeq"],
        "latestAction": copy.deepcopy(latest_action),
        "settings": copy.deepcopy(SETTINGS),
        "streams": {
            "world": {
                "source": latest_world.get("source", "waiting-for-world"),
                "frameSeq": latest_world.get("frameSeq", 0),
                "observedAt": STATE["worldObservedAt"],
            },
            "eventCamera": {
                "frameSeq": (latest_world.get("eventCamera") or {}).get("frameSeq", 0),
                "count": (latest_world.get("eventCamera") or {}).get("count", 0),
            },
            "snn": {
                "samples": SNN_RUNTIME["samples"],
                "lastFrameSeq": SNN_RUNTIME["lastFrameSeq"],
                "lastStepAt": SNN_RUNTIME["lastStepAt"],
            },
        },
        "snn": {
            "training": training,
            "paused": paused,
            "winner": activity.get("winner"),
            "direction": activity.get("direction"),
        },
    }


def compact_events_locked():
    min_tick = max(0, STATE["authoritativeTick"] - 1200)
    if len(STATE["events"]) > 2500:
        STATE["events"] = [event for event in STATE["events"] if int(event.get("tick", 0)) >= min_tick]
        STATE["inputEvents"] = [event for event in STATE["events"] if event["type"] == "input"]


def add_session_event_locked(body, default_type="input"):
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
        source = body.get("source")
        event["source"] = source if source in ("human", "snn", "api") else "api"
        STATE["apiDirection"] = event["direction"]
        STATE["inputSeq"] = event["seq"]
        STATE["latestAction"] = {
            "seq": event["seq"],
            "direction": event["direction"],
            "tick": event["tick"],
            "source": event["source"],
            "sessionId": STATE["sessionId"],
            "emittedAt": event["receivedAt"],
        }
    STATE["eventSeq"] = event["seq"]
    STATE["events"].append(event)
    STATE["events"].sort(key=lambda item: (item["tick"], item["seq"]))
    STATE["inputEvents"] = [item for item in STATE["events"] if item["type"] == "input"]
    if event_type == "start":
        STATE["running"] = True
    elif event_type == "pause":
        STATE["running"] = False
    STATE["updatedAt"] = time.time()
    compact_events_locked()
    return event


def normalize_event_camera(value, fallback_tick=0, fallback_frame_seq=0, fallback_rendered_at=0):
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
        "tick": int(value.get("tick", fallback_tick) or 0),
        "frameSeq": int(value.get("frameSeq", fallback_frame_seq) or 0),
        "resetToken": value.get("resetToken"),
        "renderedAt": float(value.get("renderedAt", fallback_rendered_at) or 0),
        "source": value.get("source") if isinstance(value.get("source"), str) else "game",
        "pixels": pixels,
        "count": len(pixels),
    }


def normalize_world_observation(body):
    settings = {**SETTINGS, **(body.get("settings") if isinstance(body.get("settings"), dict) else {})}
    ball = body.get("ball") if isinstance(body.get("ball"), dict) else {}
    paddles = body.get("paddles") if isinstance(body.get("paddles"), dict) else {}
    score = body.get("score") if isinstance(body.get("score"), dict) else {}
    tick = int(body.get("tick", body.get("authoritativeTick", 0)) or 0)
    frame_seq = int(body.get("frameSeq", tick) or 0)
    rendered_at = float(body.get("renderedAt", time.time() * 1000) or 0)
    event_camera = normalize_event_camera(body.get("eventCamera"), tick, frame_seq, rendered_at)
    return {
        "sessionId": STATE["sessionId"],
        "resetToken": STATE["resetToken"],
        "seed": STATE["seed"],
        "tick": tick,
        "authoritativeTick": tick,
        "frameSeq": frame_seq,
        "renderedAt": rendered_at,
        "running": bool(body.get("running", STATE["running"])),
        "settings": copy.deepcopy(settings),
        "score": {
            "left": int(score.get("left", 0) or 0),
            "right": int(score.get("right", 0) or 0),
        },
        "ball": {
            "x": float(ball.get("x", settings["width"] / 2) or 0),
            "y": float(ball.get("y", settings["height"] / 2) or 0),
            "vx": float(ball.get("vx", 0) or 0),
            "vy": float(ball.get("vy", 0) or 0),
        },
        "paddles": {
            "leftY": float(paddles.get("leftY", (settings["height"] - settings["paddleHeight"]) / 2) or 0),
            "rightY": float(paddles.get("rightY", (settings["height"] - settings["paddleHeight"]) / 2) or 0),
        },
        "appliedInputSeq": int(body.get("appliedInputSeq", 0) or 0),
        "appliedEventSeq": int(body.get("appliedEventSeq", 0) or 0),
        "eventCamera": event_camera,
        "source": body.get("source") if isinstance(body.get("source"), str) else "game",
        "updatedAt": time.time(),
    }


def reset_game_state_locked(reset_score=True):
    STATE["sessionId"] = f"local-{int(time.time() * 1000):x}"
    STATE["resetToken"] += 1
    STATE["seed"] = int(time.time() * 1000) & 0xFFFFFFFF
    STATE["authoritativeTick"] = 0
    STATE["eventSeq"] = 0
    STATE["events"] = []
    STATE["inputSeq"] = 0
    STATE["inputEvents"] = []
    STATE["apiDirection"] = 0
    STATE["running"] = False
    STATE["latestAction"] = None
    STATE["worldFrameSeq"] = 0
    STATE["worldObservedAt"] = 0
    latest = initial_world_state()
    if not reset_score and STATE.get("latestWorld"):
        latest["score"] = copy.deepcopy(STATE["latestWorld"].get("score", latest["score"]))
    STATE["latestWorld"] = latest
    STATE["updatedAt"] = time.time()
    SNN_RUNTIME["lastFrameSeq"] = -1
    SNN_RUNTIME["lastStepAt"] = 0.0
    SNN_RUNTIME["samples"] = 0
    SNN_RUNTIME["skippedFrames"] = 0


def snn_loop():
    target_dt = 1.0 / max(1, int(SETTINGS.get("snnStepHz", 60)))
    while True:
        started_at = time.perf_counter()
        with STATE_LOCK:
            world = copy.deepcopy(STATE["latestWorld"])
            frame_seq = int((world or {}).get("frameSeq", -1))
            event_camera = copy.deepcopy((world or {}).get("eventCamera"))
            tick = int((world or {}).get("tick", STATE["authoritativeTick"]))
            should_sample = (
                event_camera is not None
                and frame_seq != SNN_RUNTIME["lastFrameSeq"]
            )
        if should_sample:
            with SNN_LOCK:
                active = SNN.training and not SNN.paused
                direction = SNN.step(event_camera, tick, world) if active else None
                should_emit = direction is not None and SNN.should_emit_command(direction, tick)
            with STATE_LOCK:
                if frame_seq > SNN_RUNTIME["lastFrameSeq"] + 1 and SNN_RUNTIME["lastFrameSeq"] >= 0:
                    SNN_RUNTIME["skippedFrames"] += frame_seq - SNN_RUNTIME["lastFrameSeq"] - 1
                SNN_RUNTIME["lastFrameSeq"] = frame_seq
                SNN_RUNTIME["lastStepAt"] = time.time()
                SNN_RUNTIME["samples"] += 1
                if should_emit:
                    add_session_event_locked(
                        {
                            "type": "input",
                            "tick": tick + 1,
                            "direction": direction,
                            "source": "snn",
                        },
                        "input",
                    )
        elapsed = time.perf_counter() - started_at
        time.sleep(max(0.001, target_dt - elapsed))


def ensure_snn_loop():
    global SNN_THREAD

    with SNN_THREAD_LOCK:
        if SNN_THREAD and SNN_THREAD.is_alive():
            return
        SNN_THREAD = threading.Thread(target=snn_loop, daemon=True, name="pong-snn-worker")
        SNN_THREAD.start()


class PongHandler(BaseHTTPRequestHandler):
    server_version = "PongSNN/0.3"

    def log_message(self, fmt, *args):
        print("%s - - [%s] %s" % (self.address_string(), self.log_date_time_string(), fmt % args))

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        ensure_snn_loop()
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
            self.send_events(parsed, only_inputs=True)
            return
        if path == "/api/events":
            self.send_events(parsed, only_inputs=False)
            return
        if path == "/api/actions/latest":
            with STATE_LOCK:
                payload = {**session_payload(), "action": copy.deepcopy(STATE["latestAction"])}
            self.send_json(payload)
            return
        if path == "/api/state" or path == "/api/world/state":
            with STATE_LOCK:
                payload = copy.deepcopy(STATE["latestWorld"])
                payload["settings"] = copy.deepcopy(SETTINGS)
                payload["latestAction"] = copy.deepcopy(STATE["latestAction"])
                payload["streams"] = session_payload()["streams"]
            self.send_json(payload)
            return
        if path == "/api/snn/status":
            with SNN_LOCK:
                payload = SNN.status()
            with STATE_LOCK:
                payload["runtime"] = copy.deepcopy(SNN_RUNTIME)
            self.send_json(payload)
            return
        if path == "/api/snn/saves":
            with SNN_LOCK:
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
        ensure_snn_loop()
        try:
            body = read_json(self)
            path = urlparse(self.path).path
            if path == "/api/control-mode":
                self.set_control_mode(body)
                return
            if path == "/api/sim-speed":
                self.set_simulation_speed(body)
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
            if path == "/api/state" or path == "/api/world/state":
                self.update_world_state(body)
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

    def send_events(self, parsed, only_inputs):
        query = parse_qs(parsed.query)
        has_since_seq = "sinceSeq" in query
        since_seq = int(query.get("sinceSeq", ["0"])[0])
        since_tick = int(query.get("sinceTick", ["-1"])[0])
        with STATE_LOCK:
            source = STATE["inputEvents"] if only_inputs else STATE["events"]
            if has_since_seq:
                events = [event for event in source if event["seq"] > since_seq]
            else:
                events = [event for event in source if event["tick"] > since_tick]
            payload = {**session_payload(), "events": copy.deepcopy(events)}
        self.send_json(payload)

    def set_control_mode(self, body):
        mode = body.get("mode")
        if mode not in ("human", "api"):
            raise ValueError('mode must be "human" or "api"')
        with STATE_LOCK:
            STATE["mode"] = mode
            STATE["updatedAt"] = time.time()
            payload = session_payload()
        self.send_json(payload)

    def set_simulation_speed(self, body):
        with STATE_LOCK:
            SETTINGS["simulationSpeed"] = parse_simulation_speed(body.get("speed"))
            STATE["updatedAt"] = time.time()
            payload = session_payload()
        self.send_json(payload)

    def set_api_input(self, body):
        self.add_event({**body, "type": "input"}, "input")

    def add_event(self, body, default_type="input"):
        with STATE_LOCK:
            event = add_session_event_locked(body, default_type)
            payload = {**session_payload(), "event": copy.deepcopy(event)}
        self.send_json(payload)

    def update_world_state(self, body):
        observation = normalize_world_observation(body)
        with STATE_LOCK:
            if observation["resetToken"] != STATE["resetToken"]:
                observation["resetToken"] = STATE["resetToken"]
            STATE["latestWorld"] = observation
            STATE["authoritativeTick"] = max(STATE["authoritativeTick"], observation["tick"])
            STATE["worldFrameSeq"] = max(STATE["worldFrameSeq"], observation["frameSeq"])
            STATE["worldObservedAt"] = time.time()
            STATE["running"] = observation["running"]
            STATE["updatedAt"] = time.time()
            compact_events_locked()
            payload = {
                "accepted": True,
                "sessionId": STATE["sessionId"],
                "resetToken": STATE["resetToken"],
                "tick": STATE["authoritativeTick"],
                "latestAction": copy.deepcopy(STATE["latestAction"]),
            }
        self.send_json(payload)

    def reset_game(self, body):
        with STATE_LOCK:
            reset_game_state_locked(body.get("resetScore", True) is not False)
            payload = copy.deepcopy(STATE["latestWorld"])
            payload["settings"] = copy.deepcopy(SETTINGS)
        self.send_json(payload)

    def start_snn(self, body):
        with SNN_LOCK:
            SNN.start()
            status = SNN.status()
        with STATE_LOCK:
            STATE["mode"] = "api"
            start_event = add_session_event_locked(
                {"type": "start", "tick": parse_tick(body.get("tick"), STATE["authoritativeTick"] + 1)},
                "start",
            )
            payload = {**session_payload(), "event": copy.deepcopy(start_event), "snn": status}
        self.send_json(payload)

    def pause_snn(self, body):
        with SNN_LOCK:
            SNN.pause()
            status = SNN.status()
        with STATE_LOCK:
            pause_event = add_session_event_locked(
                {"type": "pause", "tick": parse_tick(body.get("tick"), STATE["authoritativeTick"] + 1)},
                "pause",
            )
            payload = {**session_payload(), "event": copy.deepcopy(pause_event), "snn": status}
        self.send_json(payload)

    def reset_snn(self, body):
        with SNN_LOCK:
            SNN.reset(reset_weights=body.get("resetWeights", True) is not False)
            status = SNN.status()
        with STATE_LOCK:
            reset_game_state_locked(body.get("resetScore", True) is not False)
            payload = copy.deepcopy(STATE["latestWorld"])
            payload["settings"] = copy.deepcopy(SETTINGS)
            payload["snn"] = status
        self.send_json(payload)

    def save_snn(self, body):
        with SNN_LOCK:
            name = SNN.save(body.get("name"))
            payload = {"saved": name, "saves": SNN.list_saves(), "snn": SNN.status()}
        self.send_json(payload)

    def load_snn(self, body):
        name = body.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("name is required")
        with SNN_LOCK:
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
    parser = argparse.ArgumentParser(description="Run the local Pong SNN stream server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    ensure_snn_loop()
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
