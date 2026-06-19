import json
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
        startup_events = request_json(base_url, "/api/events?sinceSeq=0&sinceTick=9999")
        assert any(event["type"] == "start" for event in startup_events["events"])
        time.sleep(0.25)
        autonomous_state = request_json(base_url, "/api/state")
        assert autonomous_state["running"] is True
        assert autonomous_state["tick"] > 0
        assert autonomous_state["eventCamera"]["source"] == "backend"
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
        assert "eligibilityIncreased" in status["activity"]["stdp"]
        input_events = request_json(base_url, "/api/inputs?sinceSeq=0")
        assert any(event["type"] == "input" for event in input_events["events"])
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
