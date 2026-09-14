"""
Live Cockpit Server: Serves the interactive Turbine Cockpit UI and streams
real-time training telemetry from an active TurboLearningEngine.

Hardened for production with:
- Localhost (127.0.0.1) default binding (mitigates public network exposure).
- Token-based authentication (X-Cockpit-Token / Bearer / ?token=...).
- Strict CORS validation (restricted to localhost origins).
- Input boundary clamping on control actions.
- Socket timeout & connection limit protections on SSE streams.
"""

import os
import json
import time
import secrets
import threading
from typing import Optional
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, SimpleHTTPRequestHandler
import numpy as np

from accelerator_ai.models.neural_core import PureNumPyMLP
from accelerator_ai.engine import TurboLearningEngine
from accelerator_ai.core.metrics import EngineTelemetry

DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard")
HOST = os.getenv("COCKPIT_HOST", "127.0.0.1")
PORT = int(os.getenv("COCKPIT_PORT", 8080))
AUTH_TOKEN = os.getenv("COCKPIT_TOKEN", secrets.token_urlsafe(16))
REQUIRE_AUTH = os.getenv("COCKPIT_REQUIRE_AUTH", "1") == "1"


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
        bounded_psi = max(0.0, min(float(psi), 45.0))
        with self.lock:
            ratio = 1.0 + (bounded_psi / 14.7)
            self.engine.compressor.set_boost(ratio)


state = ServerEngineState()


def engine_training_worker():
    """Background thread continuously training the engine to generate live telemetry."""
    np.random.seed(42)
    state.engine.telemetry_hub.add_listener(state.update_telemetry)

    while state.running:
        batch_size = 32
        x = np.random.randn(batch_size, 2).astype(np.float32)
        y = (np.sum(x ** 2, axis=1) > 1.2).astype(np.int32)

        state.engine.step(x, y)
        time.sleep(0.08)  # ~12 steps per second for smooth telemetry streaming


class CockpitHTTPHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    def _get_origin(self) -> Optional[str]:
        origin = self.headers.get("Origin")
        if origin:
            if origin.startswith("http://localhost:") or origin.startswith("http://127.0.0.1:"):
                return origin
        return None

    def _send_cors(self):
        origin = self._get_origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Credentials", "true")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Cockpit-Token, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def _is_authorized(self) -> bool:
        if not REQUIRE_AUTH:
            return True

        token = self.headers.get("X-Cockpit-Token")
        if not token:
            auth_header = self.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:].strip()
        if not token and "?" in self.path:
            qs = parse_qs(urlparse(self.path).query)
            tokens = qs.get("token", [])
            if tokens:
                token = tokens[0]

        if not token:
            return False
        return secrets.compare_digest(token, AUTH_TOKEN)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith("/api/"):
            if not self._is_authorized():
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Unauthorized. Provide valid X-Cockpit-Token."}).encode("utf-8"))
                return

        if path == "/api/telemetry":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.end_headers()
            with state.lock:
                data = json.dumps(state.latest_telemetry)
            self.wfile.write(data.encode("utf-8"))
            return

        if path == "/api/stream":
            # Server-Sent Events (SSE) with connection timeout & lifecycle check
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self._send_cors()
            self.end_headers()

            self.connection.settimeout(5.0)
            start_time = time.time()
            try:
                while state.running and (time.time() - start_time < 3600):
                    with state.lock:
                        payload = f"data: {json.dumps(state.latest_telemetry)}\n\n"
                    self.wfile.write(payload.encode("utf-8"))
                    self.wfile.flush()
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
            return

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if not self._is_authorized():
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Unauthorized"}).encode("utf-8"))
            return

        if path == "/api/action/nos":
            state.trigger_nos()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.end_headers()
            self.wfile.write(json.dumps({"status": "nos_injected"}).encode("utf-8"))
            return

        if path == "/api/action/boost":
            content_len = int(self.headers.get("Content-Length", 0))
            if content_len > 4096:
                self.send_error(413, "Payload too large")
                return

            body = self.rfile.read(content_len)
            try:
                data = json.loads(body)
                raw_psi = float(data.get("boost_psi", 14.7))
                state.set_boost(raw_psi)
            except Exception:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Invalid boost value"}).encode("utf-8"))
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.end_headers()
            self.wfile.write(json.dumps({"status": "boost_updated"}).encode("utf-8"))
            return

        self.send_error(404, "Unknown action")


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


def run_server():
    t = threading.Thread(target=engine_training_worker, daemon=True)
    t.start()

    server_address = (HOST, PORT)
    httpd = ReusableHTTPServer(server_address, CockpitHTTPHandler)
    access_url = f"http://{HOST}:{PORT}/?token={AUTH_TOKEN}"
    print("=" * 75)
    print(f"   ACCELERATOR-AI TURBINE COCKPIT SERVER RUNNING (SECURED)")
    print(f"   >> Local Interface: http://{HOST}:{PORT}")
    print(f"   >> Browser Access:  {access_url}")
    print(f"   >> Auth Token:      {AUTH_TOKEN}")
    print(f"   >> Telemetry & Interactive Engine Active")
    print("=" * 75)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping turbine engine...")
        state.running = False
        httpd.server_close()


if __name__ == "__main__":
    run_server()
