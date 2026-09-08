"""
TelemetryHub: Real-time event aggregator and logger for AcceleratorAI.
Dispatches engine telemetry to listeners (Cockpit UI, WebSocket, files, terminal).
"""

from collections import deque
from typing import List, Callable, Dict, Any, Optional
from accelerator_ai.core.metrics import EngineTelemetry


class TelemetryHub:
    """
    Central dispatch and history hub for all engine telemetry streams.
    """

    def __init__(self, history_len: int = 1000):
        self.history = deque(maxlen=history_len)
        self.listeners: List[Callable[[EngineTelemetry], None]] = []
        self.latest_telemetry: Optional[EngineTelemetry] = None

    def add_listener(self, callback: Callable[[EngineTelemetry], None]) -> None:
        """Subscribes a callback to receive every telemetry event."""
        self.listeners.append(callback)

    def emit(self, telemetry: EngineTelemetry) -> None:
        """Records telemetry into history and broadcasts to all active listeners."""
        self.latest_telemetry = telemetry
        self.history.append(telemetry)
        for listener in self.listeners:
            try:
                listener(telemetry)
            except Exception as e:
                pass

    def get_recent_history(self, count: int = 50) -> List[Dict[str, Any]]:
        """Returns the most recent N telemetry frames as dictionaries."""
        items = list(self.history)[-count:]
        return [t.to_dict() for t in items]
