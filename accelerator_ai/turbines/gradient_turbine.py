"""
Gradient Turbine: Exhaust-driven turbine wheel physically mounted on the DriveShaft.

Extracts kinetic energy from backprop loss gradients, generating physical torque
that accelerates the DriveShaft and powers the intake compressor.
"""

from typing import Any, Dict, Tuple, Optional
import numpy as np
from accelerator_ai.core.base_turbine import TurbineModule
from accelerator_ai.core.flow_packet import FlowPacket
from accelerator_ai.core.shaft import DriveShaft
from accelerator_ai.core.metrics import calculate_learning_torque


class GradientTurbine(TurbineModule):
    """
    Simulates the exhaust turbine housing and wheel.
    The backward pass flows through this turbine; the magnitude of the gradients
    generates Learning Torque that drives the physical DriveShaft.
    """

    def __init__(
        self,
        shaft_mechanical_efficiency: float = 0.92,
        shaft: Optional[DriveShaft] = None,
    ):
        super().__init__(name="GradientTurbine")
        self.shaft_efficiency = shaft_mechanical_efficiency
        self.shaft = shaft
        self.learning_torque_nm: float = 0.0
        self.cumulative_torque: float = 0.0
        self.gradient_norm: float = 0.0

    def attach_shaft(self, shaft: DriveShaft) -> None:
        """Physically mounts turbine wheel onto a DriveShaft."""
        self.shaft = shaft

    def process(self, packet: FlowPacket) -> FlowPacket:
        """Passthrough placeholder for generic pipeline chaining."""
        return packet

    def harvest_gradients(
        self,
        model: Any,
        fused_packet: FlowPacket,
        boost_ratio: float,
    ) -> Tuple[float, float]:
        """
        Executes backward pass, calculates gradient norms, and extracts Learning Torque.
        
        Returns:
            (gradient_norm, learning_torque_nm)
        """
        # Trigger backward pass on the model
        grad_norm = model.backward()
        self.gradient_norm = float(grad_norm)

        # Torque = ||grad|| * Boost * Shaft Efficiency
        raw_torque = calculate_learning_torque(self.gradient_norm, boost_ratio)
        self.learning_torque_nm = float(raw_torque * self.shaft_efficiency)
        self.cumulative_torque += self.learning_torque_nm

        # Update local module telemetry
        if self.shaft is not None:
            self.rpm = self.shaft.rpm
        else:
            rpm_delta = min(500.0, self.learning_torque_nm * 80.0)
            self.spin(delta_rpm=rpm_delta)

        self.total_processed_packets += 1
        self.total_processed_samples += fused_packet.batch_size

        self.last_telemetry = {
            "gradient_norm": round(self.gradient_norm, 5),
            "learning_torque_nm": round(self.learning_torque_nm, 3),
            "cumulative_torque": round(self.cumulative_torque, 2),
            "turbine_rpm": round(self.rpm, 1),
        }
        return self.gradient_norm, self.learning_torque_nm
