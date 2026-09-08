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
    Cools charge temperature T, normalizes batch activations,
    and prevents detonation / thermal runaway (overfitting/exploding gradients).
    """

    def __init__(self, cooling_efficiency: float = 0.85, epsilon: float = 1e-5):
        super().__init__(name="Intercooler")
        self.cooling_efficiency = cooling_efficiency  # 0.0 to 1.0
        self.epsilon = epsilon
        self.thermal_delta_c: float = 0.0

    def process(self, packet: FlowPacket) -> FlowPacket:
        """
        Normalizes feature representations and drops charge temperature.
        """
        x = packet.x

        # 1. Thermal cooling of feature variance (Layer/Batch-wise stabilization)
        mean = np.mean(x, axis=0, keepdims=True)
        var = np.var(x, axis=0, keepdims=True)
        normalized_x = (x - mean) / np.sqrt(var + self.epsilon)

        # 2. Blend back slightly based on cooling efficiency to preserve subtle scale
        cooled_x = (self.cooling_efficiency * normalized_x) + ((1.0 - self.cooling_efficiency) * x)

        # 3. Drop packet temperature
        temp_reduction = packet.temperature * self.cooling_efficiency * 0.4
        packet.temperature = max(0.2, packet.temperature - temp_reduction)
        self.thermal_delta_c = float(temp_reduction * 100.0)

        self.total_processed_packets += 1
        self.total_processed_samples += packet.batch_size
        self.efficiency = self.cooling_efficiency

        packet.x = cooled_x
        packet.metadata["intercooler_temp_drop"] = round(self.thermal_delta_c, 2)

        self.last_telemetry = {
            "cooling_efficiency": self.cooling_efficiency,
            "charge_temp_after_cooling": round(packet.temperature, 3),
            "temp_drop_c": round(self.thermal_delta_c, 1),
        }
        return packet
