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


class TwinScrollHousing:
    """
    Twin-Scroll Divided Exhaust Turbine Housing.

    Keeps primary dataset gradient pulses isolated from auxiliary asynchronous
    injection shocks, preventing destructive wave interference in the exhaust runners.

    Scroll A: Dedicated to the primary training data stream (steady curriculum flow).
    Scroll B: Dedicated to asynchronous auxiliary injections (synthetic, edge, chaos shocks).
    """

    def __init__(self, pulse_isolation_factor: float = 0.95):
        self.pulse_isolation_factor = pulse_isolation_factor
        self.scroll_a_pressure: float = 0.0
        self.scroll_b_pressure: float = 0.0
        self.pulse_balance: float = 1.0  # 1.0 = pure Scroll A, 0.0 = pure Scroll B

    def divide_and_combine(
        self,
        base_torque: float,
        fused_packet: FlowPacket,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculates separate Scroll A and Scroll B kinetic pressures and combines
        them with zero destructive pulse cancellation.
        """
        # Determine injection fraction from packet metadata
        injected_count = fused_packet.metadata.get("injected_samples", 0)
        total_samples = max(1, fused_packet.batch_size)
        inj_ratio = float(np.clip(injected_count / total_samples, 0.0, 1.0))
        main_ratio = 1.0 - inj_ratio

        # Scroll A (Primary data stream)
        self.scroll_a_pressure = float(base_torque * main_ratio)

        # Scroll B (Auxiliary injection pulses)
        # High entropy injections exert high instantaneous impulse
        shock_fired = fused_packet.metadata.get("shock_fired", False)
        shock_multiplier = 1.35 if shock_fired else 1.0
        self.scroll_b_pressure = float(base_torque * inj_ratio * shock_multiplier)

        # Total combined torque with twin-scroll pulse isolation
        # In a single-scroll, pulses can interfere destructively (losses up to 20%)
        # Twin-scroll isolates nozzles, providing clean constructive summation
        combined_torque = self.scroll_a_pressure + self.scroll_b_pressure

        # Pulse balance index: 0.5 = balanced, 1.0 = pure main, 0.0 = pure injection
        total_pressure = self.scroll_a_pressure + self.scroll_b_pressure + 1e-8
        self.pulse_balance = float(self.scroll_a_pressure / total_pressure)

        telemetry = {
            "scroll_a_pressure": round(self.scroll_a_pressure, 3),
            "scroll_b_pressure": round(self.scroll_b_pressure, 3),
            "twin_scroll_balance": round(self.pulse_balance, 3),
            "injection_pulse_active": float(inj_ratio > 0.0),
        }
        return combined_torque, telemetry


class GradientTurbine(TurbineModule):
    """
    Simulates the exhaust turbine housing and wheel with Twin-Scroll runners.
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
        enable_twin_scroll: bool = True,
    ):
        super().__init__(name="GradientTurbine")
        self.shaft_efficiency = shaft_mechanical_efficiency
        self.shaft = shaft
        self.enable_twin_scroll = enable_twin_scroll
        self.twin_scroll = TwinScrollHousing() if enable_twin_scroll else None

        self.learning_torque_nm: float = 0.0
        self.cumulative_torque: float = 0.0
        self.gradient_norm: float = 0.0
        self.curriculum_torque_factor: float = 1.0
        self.twin_scroll_telemetry: Dict[str, float] = {}

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
        Executes backward pass, calculates gradient norms, routes through Twin-Scroll
        exhaust runners, and extracts Learning Torque.

        Returns:
            (gradient_norm, learning_torque_nm)
        """
        # Trigger backward pass on the model
        grad_norm = model.backward()
        self.gradient_norm = float(grad_norm)

        # Curriculum torque multiplier from VGT port routing
        curriculum_weights = fused_packet.metadata.get("curriculum_weights", None)
        if curriculum_weights is not None and len(curriculum_weights) > 0:
            self.curriculum_torque_factor = float(np.mean(curriculum_weights))
        else:
            self.curriculum_torque_factor = 1.0

        # Base torque = ||grad|| * Boost * Shaft Efficiency * Curriculum Factor
        raw_torque = calculate_learning_torque(self.gradient_norm, boost_ratio)
        base_torque = float(raw_torque * self.shaft_efficiency * self.curriculum_torque_factor)

        # Twin-Scroll Exhaust Runner Isolation
        if self.twin_scroll is not None:
            self.learning_torque_nm, self.twin_scroll_telemetry = self.twin_scroll.divide_and_combine(
                base_torque, fused_packet
            )
        else:
            self.learning_torque_nm = base_torque
            self.twin_scroll_telemetry = {}

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
            **self.twin_scroll_telemetry,
        }
        return self.gradient_norm, self.learning_torque_nm


