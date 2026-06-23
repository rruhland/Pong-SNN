import json
import math
import sys
import threading
import time
from pathlib import Path
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server


def request_json(base_url, path, body=None):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(f"{base_url}{path}", data=data, headers=headers)
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def event_pixels(frame, width=800):
    y = 120 + (frame * 7) % 190
    x = 430 + (frame * 5) % 130
    return [y * width + x + offset for offset in range(12)]


def post_world_frame(base_url, frame):
    y = 120 + math.sin(frame / 4) * 80
    right_y = 180 + math.sin(frame / 5) * 45
    left_score = 1 if frame > 28 else 0
    body = {
        "sessionId": "smoke-world",
        "resetToken": 0,
        "seed": 123,
        "tick": frame,
        "authoritativeTick": frame,
        "frameSeq": frame,
        "renderedAt": frame * 16.667,
        "running": True,
        "settings": {
            "width": 800,
            "height": 450,
            "paddleWidth": 10,
            "paddleHeight": 80,
            "ballSize": 10,
            "fixedDt": 1 / 60,
        },
        "score": {"left": left_score, "right": 0},
        "ball": {"x": 430 + frame * 3, "y": y, "vx": 210, "vy": 25},
        "paddles": {"leftY": y - 40, "rightY": right_y},
        "appliedInputSeq": 0,
        "appliedEventSeq": 0,
        "eventCamera": {
            "width": 800,
            "height": 450,
            "tick": frame,
            "frameSeq": frame,
            "resetToken": 0,
            "source": "game",
            "pixels": event_pixels(frame),
            "count": 12,
        },
        "source": "game-renderer",
    }
    return request_json(base_url, "/api/world/state", body)


def main():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.PongHandler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    try:
        started = request_json(base_url, "/api/snn/start", {})
        assert started["snn"]["training"] is True
        assert started["snn"]["paused"] is False

        waiting_state = request_json(base_url, "/api/state")
        assert waiting_state["source"] == "waiting-for-world"
        assert waiting_state["tick"] == 0

        startup_events = request_json(base_url, "/api/events?sinceSeq=0&sinceTick=9999")
        assert any(event["type"] == "start" for event in startup_events["events"])

        for frame in range(1, 36):
            accepted = post_world_frame(base_url, frame)
            assert accepted["accepted"] is True
            time.sleep(0.02)

        observed_state = request_json(base_url, "/api/state")
        assert observed_state["source"] == "game-renderer"
        assert observed_state["tick"] >= 30
        assert observed_state["eventCamera"]["source"] == "game"

        speed = request_json(base_url, "/api/sim-speed", {"speed": 1.75})
        assert speed["settings"]["simulationSpeed"] == 1.75
        clamped_speed = request_json(base_url, "/api/sim-speed", {"speed": 99})
        assert clamped_speed["settings"]["simulationSpeed"] == clamped_speed["settings"]["maxSimulationSpeed"]

        status = request_json(base_url, "/api/snn/status")
        assert status["activity"]["eventCount"] > 0
        assert status["activity"]["spikes"]["hidden1"] > 0
        assert status["activity"]["winner"] in {"move up", "move down", "stay put"}
        assert isinstance(status["reward"]["value"], (int, float))
        assert "inputH1" in status["eligibility"]
        assert "h3Output" in status["eligibility"]
        assert status["learning"]["step"] > 0
        assert status["runtime"]["samples"] > 0
        assert "eligibilityIncreased" in status["activity"]["stdp"]

        input_events = request_json(base_url, "/api/inputs?sinceSeq=0")
        assert any(event["type"] == "input" and event["source"] == "snn" for event in input_events["events"])

        paused = request_json(base_url, "/api/snn/pause", {})
        assert paused["snn"]["paused"] is True
        print(json.dumps({"ok": True, "winner": status["activity"]["winner"], "direction": status["activity"]["direction"]}))
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)
        time.sleep(0.05)


if __name__ == "__main__":
    main()
