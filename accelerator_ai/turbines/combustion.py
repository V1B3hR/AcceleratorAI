"""
Combustion Chamber: The core ignition stage.
Mixes pressurized charge air with supplemental injected fuel (synthetic/real/shock),
executes the forward pass, and ignites informational loss energy into exhaust gases.
"""

from typing import List, Optional, Tuple, Any
import numpy as np
from accelerator_ai.core.base_turbine import TurbineModule
from accelerator_ai.core.flow_packet import FlowPacket


class CombustionResult:
    """Holds the products of combustion: predictions, loss, gradients, and exhaust energy."""
    def __init__(
        self,
        loss: float,
        predictions: np.ndarray,
        fused_packet: FlowPacket,
        exhaust_energy: float,
        air_fuel_ratio: float,
    ):
        self.loss = loss
        self.predictions = predictions
        self.fused_packet = fused_packet
        self.exhaust_energy = exhaust_energy
        self.air_fuel_ratio = air_fuel_ratio


class CombustionChamber(TurbineModule):
    """
    Simulates the internal combustion cylinders where data fuel is ignited.
    Fuses the main intercooled charge with asynchronous injection pulses,
    invokes the forward pass on the active model, and measures exhaust enthalpy.
    """

    def __init__(self, stoichiometric_ratio: float = 14.7):
        super().__init__(name="CombustionChamber")
        self.stoichiometric_ratio = stoichiometric_ratio
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
        Mixes main intake air with supplemental injected packets,
        runs the forward pass, and computes loss and exhaust energy.
        """
        all_packets = [main_packet]
        injected_sample_count = 0

        if injected_packets:
            for p in injected_packets:
                if p is not None and p.batch_size > 0:
                    all_packets.append(p)
                    injected_sample_count += p.batch_size

        # Blend all packets into one combustion mixture
        fused = FlowPacket.merge(all_packets, source_tag="combustion_mix")

        # Air-Fuel Ratio analog
        base_samples = main_packet.batch_size
        afr = (
            self.stoichiometric_ratio
            if injected_sample_count == 0
            else float(base_samples / max(1, injected_sample_count))
        )

        # Forward pass on the model
        predictions, loss = model.forward_and_loss(fused.x, fused.y)

        # Informational combustion energy = Loss * Manifold Pressure
        exhaust_energy = float(loss * fused.pressure)
        self.cumulative_exhaust_energy += exhaust_energy
        self.ignition_count += 1
        self.total_processed_packets += len(all_packets)
        self.total_processed_samples += fused.batch_size

        # Spin chamber crank
        self.spin(delta_rpm=15.0)

        self.last_telemetry = {
            "combustion_loss": round(float(loss), 4),
            "exhaust_energy": round(exhaust_energy, 4),
            "air_fuel_ratio": round(afr, 2),
            "injected_samples": injected_sample_count,
            "total_mixture_samples": fused.batch_size,
        }

        return CombustionResult(
            loss=float(loss),
            predictions=predictions,
            fused_packet=fused,
            exhaust_energy=exhaust_energy,
            air_fuel_ratio=afr,
        )
