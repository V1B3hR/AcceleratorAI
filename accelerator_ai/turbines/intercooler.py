"""
Intercooler: Charge air cooling and thermal stabilization module.
Prevents thermal runaway, overfitting, and exploding activations by normalizing
compressed representations before the combustion chamber.
"""

import numpy as np
from accelerator_ai.core.base_turbine import TurbineModule
from accelerator_ai.core.flow_packet import FlowPacket


class Intercooler(TurbineModule):
    """
    Simulates a high-efficiency charge-air intercooler.
    Cools charge temperature T and optionally normalizes batch activations.

    bypass_mode (default True): When True, skips external feature normalization
    which destroys learned scale/shift information before the model receives it.
    Only tracks temperature metadata and applies gentle thermal reduction.
    When False, applies the full batch-wise normalization (legacy behavior).
    """

    def __init__(self, cooling_efficiency: float = 0.85, epsilon: float = 1e-5, bypass_mode: bool = True):
        super().__init__(name="Intercooler")
        self.cooling_efficiency = cooling_efficiency  # 0.0 to 1.0
        self.epsilon = epsilon
        self.bypass_mode = bypass_mode
        self.thermal_delta_c: float = 0.0

    def process(self, packet: FlowPacket) -> FlowPacket:
        """
        Processes charge air through the intercooler.
        In bypass mode: only reduces temperature metadata (preserves features).
        In legacy mode: full batch normalization + temperature reduction.
        """
        x = packet.x

        if not self.bypass_mode:
            # Legacy mode: external batch normalization (can harm convergence)
            mean = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)
            normalized_x = (x - mean) / np.sqrt(var + self.epsilon)
            cooled_x = (self.cooling_efficiency * normalized_x) + ((1.0 - self.cooling_efficiency) * x)
            packet.x = cooled_x

        # Temperature reduction (always active — metadata-only in bypass mode)
        temp_reduction = packet.temperature * self.cooling_efficiency * 0.4
        packet.temperature = max(0.2, packet.temperature - temp_reduction)
        self.thermal_delta_c = float(temp_reduction * 100.0)

        self.total_processed_packets += 1
        self.total_processed_samples += packet.batch_size
        self.efficiency = self.cooling_efficiency

        packet.metadata["intercooler_temp_drop"] = round(self.thermal_delta_c, 2)

        self.last_telemetry = {
            "cooling_efficiency": self.cooling_efficiency,
            "bypass_mode": self.bypass_mode,
            "charge_temp_after_cooling": round(packet.temperature, 3),
            "temp_drop_c": round(self.thermal_delta_c, 1),
        }
        return packet
