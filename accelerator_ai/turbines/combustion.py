"""
Combustion Chamber: The core ignition stage.
Mixes pressurized charge air with supplemental injected fuel (synthetic/real/shock),
executes the forward pass, and ignites informational loss energy into exhaust gases.
"""

from typing import List, Optional, Tuple, Any
import numpy as np
from accelerator_ai.core.base_turbine import TurbineModule
from accelerator_ai.core.flow_packet import FlowPacket
from accelerator_ai.turbines.dispersion_valve import SwirlDispersionValve


class CombustionResult:
    """Holds the products of combustion: predictions, loss, gradients, and exhaust energy."""
    def __init__(
        self,
        loss: float,
        predictions: np.ndarray,
        fused_packet: FlowPacket,
        exhaust_energy: float,
        air_fuel_ratio: float,
        homogeneity_pct: float = 100.0,
    ):
        self.loss = loss
        self.predictions = predictions
        self.fused_packet = fused_packet
        self.exhaust_energy = exhaust_energy
        self.air_fuel_ratio = air_fuel_ratio
        self.homogeneity_pct = homogeneity_pct


class CombustionChamber(TurbineModule):
    """
    Simulates the internal combustion cylinders where data fuel is ignited.
    Uses an integrated SwirlDispersionValve to atomize and swirl-disperse
    asynchronous injection pulses uniformly into the intake charge,
    achieving a homogeneous flame front without gradient knocking.
    """

    def __init__(self, stoichiometric_ratio: float = 14.7, dispersion_valve: Optional[SwirlDispersionValve] = None):
        super().__init__(name="CombustionChamber")
        self.stoichiometric_ratio = stoichiometric_ratio
        self.dispersion_valve = dispersion_valve or SwirlDispersionValve()
        self.ignition_count: int = 0
        self.cumulative_exhaust_energy: float = 0.0

    def process(self, packet: FlowPacket) -> FlowPacket:
        """Standard pipeline passthrough (when no model is explicitly bound)."""
        return packet

    def ignite(
        self,
        main_packet: FlowPacket,
        injected_packets: Optional[List[FlowPacket]],
        model: Any,
    ) -> CombustionResult:
        """
        Atomizes and swirl-disperses supplemental fuel pulses into main intake charge,
        executes forward pass, and measures exhaust enthalpy.
        """
        injected_sample_count = sum(p.batch_size for p in (injected_packets or []) if p is not None)

        # Disperse and atomize via SwirlDispersionValve
        fused = self.dispersion_valve.disperse_and_mix(main_packet, injected_packets)
        homogeneity = self.dispersion_valve.last_homogeneity_pct

        # Air-Fuel Ratio analog
        base_samples = main_packet.batch_size
        afr = (
            self.stoichiometric_ratio
            if injected_sample_count == 0
            else float(base_samples / max(1, injected_sample_count))
        )

        # Forward pass on the model (with VGT curriculum weighting if available)
        curriculum_weights = fused.metadata.get("curriculum_weights", None)
        predictions, loss = model.forward_and_loss(fused.x, fused.y, sample_weights=curriculum_weights)

        # Informational combustion energy = Loss * Manifold Pressure
        exhaust_energy = float(loss * fused.pressure)
        self.cumulative_exhaust_energy += exhaust_energy
        self.ignition_count += 1
        self.total_processed_packets += 1 + (len(injected_packets) if injected_packets else 0)
        self.total_processed_samples += fused.batch_size

        # Spin chamber crank
        self.spin(delta_rpm=15.0)

        self.last_telemetry = {
            "combustion_loss": round(float(loss), 4),
            "exhaust_energy": round(exhaust_energy, 4),
            "air_fuel_ratio": round(afr, 2),
            "injected_samples": injected_sample_count,
            "total_mixture_samples": fused.batch_size,
            "charge_homogeneity_pct": round(homogeneity, 1),
        }

        return CombustionResult(
            loss=float(loss),
            predictions=predictions,
            fused_packet=fused,
            exhaust_energy=exhaust_energy,
            air_fuel_ratio=afr,
            homogeneity_pct=homogeneity,
        )
