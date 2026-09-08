"""
Live Cockpit Server: Serves the interactive Turbine Cockpit UI and streams
real-time training telemetry from an active TurboLearningEngine.

Runs using Python's standard library (http.server and threading),
requiring zero external web server frameworks.
"""

import os
import json
import time
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
import numpy as np

from accelerator_ai.models.neural_core import PureNumPyMLP
from accelerator_ai.engine import TurboLearningEngine
from accelerator_ai.core.metrics import EngineTelemetry

DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard")
PORT = 8080

# Shared Global Engine State
class ServerEngineState:
    def __init__(self):
        self.lock = threading.Lock()
        self.model = PureNumPyMLP(layer_sizes=[2, 32, 16, 2], activation="tanh", momentum=0.9)
        self.engine = TurboLearningEngine(
            model=self.model,
            target_boost_psi=14.7,
            base_learning_rate=0.015,
            enable_default_injectors=True,
        )
        self.latest_telemetry: dict = {}
        self.running: bool = True

    def update_telemetry(self, t: EngineTelemetry):
        with self.lock:
            self.latest_telemetry = t.to_dict()

    def trigger_nos(self):
        with self.lock:
            self.engine.trigger_nos()

    def set_boost(self, psi: float):
        with self.lock:
            ratio = 1.0 + (psi / 14.7)
            self.engine.compressor.set_boost(ratio)


state = ServerEngineState()


def engine_training_worker():
    """Background thread continuously training the engine to generate live telemetry."""
    # Generate continuous training stream
    np.random.seed(42)
    state.engine.telemetry_hub.add_listener(state.update_telemetry)

    while state.running:
        # Generate batch on-the-fly
        batch_size = 32
        x = np.random.randn(batch_size, 2).astype(np.float32)
        y = (np.sum(x ** 2, axis=1) > 1.2).astype(np.int32)

        state.engine.step(x, y)
        time.sleep(0.08)  # ~12 steps per second for smooth telemetry streaming


class CockpitHTTPHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    def do_GET(self):
        if self.path == "/api/telemetry":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with state.lock:
                data = json.dumps(state.latest_telemetry)
            self.wfile.write(data.encode("utf-8"))
            return

        if self.path == "/api/stream":
            # Server-Sent Events (SSE)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                while True:
                    with state.lock:
                        payload = f"data: {json.dumps(state.latest_telemetry)}\n\n"
                    self.wfile.write(payload.encode("utf-8"))
                    self.wfile.flush()
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/action/nos":
            state.trigger_nos()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "nos_injected"}).encode("utf-8"))
            return

        if self.path.startswith("/api/action/boost"):
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            try:
                data = json.loads(body)
                state.set_boost(float(data.get("boost_psi", 14.7)))
            except Exception:
                pass
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "boost_updated"}).encode("utf-8"))
            return

        self.send_error(404, "Unknown action")


def run_server():
    # Start training thread
    t = threading.Thread(target=engine_training_worker, daemon=True)
    t.start()

    server_address = ("", PORT)
    httpd = HTTPServer(server_address, CockpitHTTPHandler)
    print("=" * 70)
    print(f"   ACCELERATOR-AI TURBINE COCKPIT SERVER RUNNING")
    print(f"   >> Open in browser: http://localhost:{PORT}")
    print(f"   >> Live Telemetry & Interactive Engine Active")
    print("=" * 70)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping turbine engine...")
        state.running = False
        httpd.server_close()


if __name__ == "__main__":
    run_server()
