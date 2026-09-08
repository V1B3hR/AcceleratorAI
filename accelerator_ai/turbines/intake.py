"""
Intake Turbine: First stage of the AcceleratorAI engine.
Handles data ingestion, laminar buffering, and throttle regulation.
"""

from collections import deque
from typing import Optional, Tuple
import numpy as np
from accelerator_ai.core.base_turbine import TurbineModule
from accelerator_ai.core.flow_packet import FlowPacket


class IntakeTurbine(TurbineModule):
    """
    Intake stage simulating the air intake manifold and throttle body.
    Maintains a laminar buffer of data and feeds FlowPackets at controlled rates.
    """

    def __init__(self, throttle_pct: float = 100.0, buffer_capacity: int = 1000):
        super().__init__(name="IntakeTurbine")
        self.throttle_pct = throttle_pct  # 0.0 to 100.0%
        self.buffer = deque(maxlen=buffer_capacity)
        self.flow_resistance = 0.05

    def ingest_raw(self, x: np.ndarray, y: np.ndarray, base_pressure: float = 1.0) -> FlowPacket:
        """Wraps raw tensors into a fresh FlowPacket and enters intake."""
        packet = FlowPacket(
            x=x,
            y=y,
            pressure=base_pressure,
            viscosity=1.0,
            temperature=1.0,
            source="intake",
        )
        return self.process(packet)

    def process(self, packet: FlowPacket) -> FlowPacket:
        """
        Applies throttle regulation and initial laminar stabilization.
        """
        effective_throttle = max(0.05, self.throttle_pct / 100.0)
        self.total_processed_packets += 1
        self.total_processed_samples += packet.batch_size

        # Spin rotor proportional to flow
        self.spin(delta_rpm=packet.batch_size * effective_throttle * 0.1)

        # Apply minor throttle throttling if restricted
        packet.pressure *= effective_throttle
        packet.metadata["intake_throttle"] = self.throttle_pct

        self.last_telemetry = {
            "throttle_pct": self.throttle_pct,
            "intake_flow_rate": packet.flow_rate,
        }
        return packet
