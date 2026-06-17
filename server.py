from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
import argparse
import copy
import json
import mimetypes
import threading
import time


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"

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
    "mode": "human",
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
    "source": "backend",
    "updatedAt": time.time(),
}

STATE_LOCK = threading.Lock()


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


def reset_game_state(reset_score=True):
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
    if reset_score:
        STATE["score"] = {"left": 0, "right": 0}
    STATE["ball"] = {"x": SETTINGS["width"] / 2, "y": SETTINGS["height"] / 2, "vx": 0, "vy": 0}
    STATE["paddles"] = {
        "leftY": (SETTINGS["height"] - SETTINGS["paddleHeight"]) / 2,
        "rightY": (SETTINGS["height"] - SETTINGS["paddleHeight"]) / 2,
    }
    STATE["appliedInputSeq"] = 0
    STATE["appliedEventSeq"] = 0
    STATE["frameSeq"] = 0
    STATE["renderedAt"] = 0
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
            since_seq = int(query.get("sinceSeq", ["0"])[0])
            since_tick = int(query.get("sinceTick", ["-1"])[0])
            with STATE_LOCK:
                if since_seq > 0:
                    events = [event for event in STATE["inputEvents"] if event["seq"] > since_seq]
                else:
                    events = [event for event in STATE["inputEvents"] if event["tick"] > since_tick]
                payload = {**session_payload(), "events": copy.deepcopy(events)}
            self.send_json(payload)
            return
        if path == "/api/events":
            query = parse_qs(parsed.query)
            since_seq = int(query.get("sinceSeq", ["0"])[0])
            since_tick = int(query.get("sinceTick", ["-1"])[0])
            with STATE_LOCK:
                if since_seq > 0:
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
        if path.startswith("/web/"):
            requested = (WEB_ROOT / unquote(path.removeprefix("/web/"))).resolve()
            if WEB_ROOT in requested.parents or requested == WEB_ROOT:
                self.serve_file(requested)
                return
        self.send_error(404, "Not found")

    def do_POST(self):
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
            if isinstance(body.get("source"), str):
                STATE["source"] = body["source"]
            STATE["updatedAt"] = time.time()
            compact_events()
            payload = {**copy.deepcopy(STATE), "tick": STATE["authoritativeTick"], "settings": SETTINGS}
        self.send_json(payload)

    def reset_game(self, body):
        with STATE_LOCK:
            reset_game_state(body.get("resetScore", True) is not False)
            payload = {**copy.deepcopy(STATE), "tick": STATE["authoritativeTick"], "settings": SETTINGS}
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
