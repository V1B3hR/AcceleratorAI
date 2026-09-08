"""
Gradient Turbine: Exhaust-driven turbine wheel physically mounted on the DriveShaft.

Extracts kinetic energy from backprop loss gradients, generating physical torque
that accelerates the DriveShaft and powers the intake compressor.

With VGT integration, the torque is curriculum-weighted: batches dominated by
hyper-flow (hard) samples generate more driving torque, creating a positive
feedback loop that naturally accelerates learning on difficult examples.
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

    Curriculum-weighted torque: when the fused packet carries VGT curriculum_weights,
    torque is scaled by the mean curriculum weight. Batches with more hyper-flow
    (hard) samples produce higher mean weight → more torque → shaft spins faster →
    compressor boosts harder → even more focus on hard samples.
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
        self.curriculum_torque_factor: float = 1.0

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

        If fused_packet carries curriculum_weights from VGT PortManifold, the torque
        is scaled by the mean curriculum weight. This completes the positive feedback
        loop: hard samples → high weight → more torque → faster shaft → more boost.

        Returns:
            (gradient_norm, learning_torque_nm)
        """
        # Trigger backward pass on the model
        grad_norm = model.backward()
        self.gradient_norm = float(grad_norm)

        # Curriculum torque multiplier from VGT port routing
        curriculum_weights = fused_packet.metadata.get("curriculum_weights", None)
        if curriculum_weights is not None and len(curriculum_weights) > 0:
            # Mean weight > 1.0 when batch is dominated by hyper-flow (hard) samples
            # Mean weight < 1.0 when batch is dominated by slow-mo (easy) samples
            self.curriculum_torque_factor = float(np.mean(curriculum_weights))
        else:
            self.curriculum_torque_factor = 1.0

        # Torque = ||grad|| * Boost * Shaft Efficiency * Curriculum Factor
        raw_torque = calculate_learning_torque(self.gradient_norm, boost_ratio)
        self.learning_torque_nm = float(
            raw_torque * self.shaft_efficiency * self.curriculum_torque_factor
        )
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
            "curriculum_torque_factor": round(self.curriculum_torque_factor, 3),
        }
        return self.gradient_norm, self.learning_torque_nm

