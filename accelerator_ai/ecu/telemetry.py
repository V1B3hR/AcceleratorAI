"""
TelemetryHub: Real-time event aggregator and logger for AcceleratorAI.
Dispatches engine telemetry to listeners (Cockpit UI, WebSocket, files, WandB, TensorBoard).
"""

import json
import logging
from collections import deque
from typing import List, Callable, Dict, Any, Optional
from accelerator_ai.core.metrics import EngineTelemetry

logger = logging.getLogger("accelerator_ai.telemetry")


class TelemetryHub:
    """
    Central dispatch and history hub for all engine telemetry streams.
    Provides thread-safe listener dispatch, structured logging, and metric export.
    """

    def __init__(self, history_len: int = 1000):
        self.history = deque(maxlen=history_len)
        self.listeners: List[Callable[[EngineTelemetry], None]] = []
        self.latest_telemetry: Optional[EngineTelemetry] = None

    def add_listener(self, callback: Callable[[EngineTelemetry], None]) -> None:
        """Subscribes a callback to receive every telemetry event."""
        if callback not in self.listeners:
            self.listeners.append(callback)

    def remove_listener(self, callback: Callable[[EngineTelemetry], None]) -> None:
        """Unsubscribes a telemetry callback."""
        if callback in self.listeners:
            self.listeners.remove(callback)

    @property
    def latest(self) -> Optional[EngineTelemetry]:
        """Returns the most recent telemetry event recorded, or None."""
        return self.latest_telemetry

    def emit(self, telemetry: EngineTelemetry) -> None:
        """Records telemetry into history and broadcasts to all active listeners."""
        self.latest_telemetry = telemetry
        self.history.append(telemetry)
        for listener in self.listeners:
            try:
                listener(telemetry)
            except Exception as e:
                listener_name = getattr(listener, "__name__", type(listener).__name__)
                logger.warning(
                    "Telemetry listener '%s' raised exception: %s",
                    listener_name,
                    e,
                    exc_info=False,
                )

    def get_recent_history(self, count: int = 50) -> List[Dict[str, Any]]:
        """Returns the most recent N telemetry frames as dictionaries."""
        items = list(self.history)[-count:]
        return [t.to_dict() for t in items]

    def clear(self) -> None:
        """Clears telemetry history buffer."""
        self.history.clear()
        self.latest_telemetry = None


class WandBCallback:
    """
    Weights & Biases telemetry callback for production experiment tracking.
    Gracefully handles missing wandb package without crashing.
    """

    def __init__(self, prefix: str = "engine/"):
        self.prefix = prefix
        self._wandb = None
        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            logger.info("WandBCallback: 'wandb' package not installed. Telemetry forwarding is disabled.")

    def __call__(self, telemetry: EngineTelemetry) -> None:
        if self._wandb is not None and self._wandb.run is not None:
            metrics = {
                f"{self.prefix}rpm": telemetry.rpm,
                f"{self.prefix}boost_psi": telemetry.boost_psi,
                f"{self.prefix}loss": telemetry.loss,
                f"{self.prefix}learning_torque": telemetry.learning_torque_nm,
                f"{self.prefix}pyrometer_temp": telemetry.pyrometer_temp_c,
                f"{self.prefix}wastegate_open_pct": telemetry.wastegate_open_pct,
                f"{self.prefix}learning_rate": telemetry.learning_rate,
                f"{self.prefix}helical_resonance": telemetry.helical_resonance,
            }
            self._wandb.log(metrics, step=telemetry.step)


class FileLogCallback:
    """
    JSON-Lines file appender for offline audit trails and telemetry replay.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath

    def __call__(self, telemetry: EngineTelemetry) -> None:
        try:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(telemetry.to_dict()) + "\n")
        except OSError as e:
            logger.warning("FileLogCallback failed to write telemetry to %s: %s", self.filepath, e)

