"""
Wastegate Valve: Pressure relief and gradient clipping mechanism.
Protects the model engine from knocking (exploding gradients) and over-boost damage.
"""

from typing import Tuple, Any
import numpy as np
from accelerator_ai.core.base_turbine import TurbineModule
from accelerator_ai.core.flow_packet import FlowPacket


class WastegateValve(TurbineModule):
    """
    Simulates a pneumatic/electronic turbocharger wastegate and blow-off valve.
    Monitors gradient norms and manifold pressure; when over-boost or explosive spikes
    occur, the valve cracks open, safely venting excess pressure (clipping gradients)
    and signaling the ECU to avoid engine detonation.
    """

    def __init__(
        self,
        max_gradient_norm: float = 5.0,
        cracking_pressure_ratio: float = 2.5,
    ):
        super().__init__(name="WastegateValve")
        self.max_gradient_norm = max_gradient_norm
        self.cracking_pressure_ratio = cracking_pressure_ratio
        self.open_pct: float = 0.0
        self.total_relief_events: int = 0

    def process(self, packet: FlowPacket) -> FlowPacket:
        """Passthrough placeholder for generic pipeline chaining."""
        return packet

    def inspect_and_regulate(
        self,
        model: Any,
        gradient_norm: float,
        boost_ratio: float,
    ) -> Tuple[float, bool]:
        """
        Inspects gradient magnitude and manifold pressure.
        If exceeding safety thresholds, clips gradients and opens wastegate.
        
        Returns:
            (clipped_gradient_norm, was_vented)
        """
        was_vented = False

        # 1. Gradient over-boost check
        if gradient_norm > self.max_gradient_norm:
            excess = gradient_norm - self.max_gradient_norm
            # Calculate valve opening percentage
            self.open_pct = min(100.0, (excess / self.max_gradient_norm) * 100.0)
            # Clip gradients in the model
            model.clip_gradients(max_norm=self.max_gradient_norm)
            clipped_norm = self.max_gradient_norm
            self.total_relief_events += 1
            was_vented = True
        else:
            # Check for excessive boost pressure without gradient explosion
            if boost_ratio > self.cracking_pressure_ratio:
                self.open_pct = min(100.0, (boost_ratio - self.cracking_pressure_ratio) * 40.0)
            else:
                self.open_pct = max(0.0, self.open_pct * 0.8)  # Smooth closure
            clipped_norm = gradient_norm

        self.last_telemetry = {
            "wastegate_open_pct": round(self.open_pct, 1),
            "vented": was_vented,
            "total_relief_events": self.total_relief_events,
            "clipped_norm": round(clipped_norm, 5),
        }
        return clipped_norm, was_vented
