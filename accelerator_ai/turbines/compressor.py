"""
Compressor Turbine: Centrifugal intake compressor wheel physically coupled to the DriveShaft.

Governed by Euler's turbomachinery equation:
    Pressure Rise Delta_P proportional to (Blade Tip Velocity)^2
    Compressor Load Torque proportional to Flow Mass * Pressure Delta
"""

from typing import Optional
import numpy as np
from accelerator_ai.core.base_turbine import TurbineModule
from accelerator_ai.core.flow_packet import FlowPacket
from accelerator_ai.core.shaft import DriveShaft


class CompressorTurbine(TurbineModule):
    """
    Compressor stage simulating the centrifugal turbocharger compressor wheel.
    Compresses input representations, amplifies information density,
    and mines high-gradient potential samples (hard examples).
    
    Can be physically coupled to a DriveShaft: its boost ratio is directly driven
    by shaft rotational velocity (RPM).
    """

    def __init__(
        self,
        base_boost_ratio: float = 1.0,
        max_boost_psi: float = 28.0,
        enable_augmentation: bool = True,
        augmentation_factor: float = 0.05,
        shaft: Optional[DriveShaft] = None,
        pressure_coefficient: float = 0.45,
    ):
        super().__init__(name="CompressorTurbine")
        self.boost_ratio = base_boost_ratio  # 1.0 = atmospheric (0 psi boost)
        self.max_boost_psi = max_boost_psi
        self.enable_augmentation = enable_augmentation
        self.augmentation_factor = augmentation_factor
        self.shaft = shaft
        self.pressure_coefficient = pressure_coefficient
        self.current_load_torque: float = 0.0

    @property
    def boost_psi(self) -> float:
        """Converts pressure ratio to gauge psi (1.0 ratio = 0 psi gauge; 2.0 ratio = 14.7 psi)."""
        return max(0.0, (self.boost_ratio - 1.0) * 14.7)

    def attach_shaft(self, shaft: DriveShaft) -> None:
        """Physically mounts compressor wheel onto a DriveShaft."""
        self.shaft = shaft

    def compute_reaction_load(self) -> float:
        """
        Calculates mechanical reaction load torque tau_load that compressor
        exerts back onto the shaft to compress the fluid medium.
        """
        delta_p = max(0.0, self.boost_ratio - 1.0)
        rpm_factor = (self.rpm / 1000.0) if self.rpm > 0 else 0.8
        # Load torque scales with pressure rise and speed
        self.current_load_torque = float(0.12 * delta_p * rpm_factor)
        return self.current_load_torque

    def update_from_shaft(self) -> float:
        """
        Calculates boost pressure ratio directly from shaft rotational speed
        according to Euler turbomachinery equation: Delta_P ~ (RPM / Idle_RPM)^1.6
        """
        if self.shaft is not None:
            self.rpm = self.shaft.rpm
            speed_ratio = self.shaft.rpm / self.shaft.idle_rpm
            # Euler centrifugal pressure curve
            boost_elevation = self.pressure_coefficient * ((speed_ratio ** 1.5) - 1.0)
            max_ratio = 1.0 + (self.max_boost_psi / 14.7)
            self.boost_ratio = float(np.clip(1.0 + max(0.0, boost_elevation), 1.0, max_ratio))
        return self.boost_ratio

    def set_boost(self, boost_ratio: float) -> None:
        """Overrides or sets target boost ratio, clamped by max gauge pressure."""
        max_ratio = 1.0 + (self.max_boost_psi / 14.7)
        self.boost_ratio = float(np.clip(boost_ratio, 1.0, max_ratio))

    def process(self, packet: FlowPacket) -> FlowPacket:
        """
        Compresses the incoming flow packet, elevating information pressure Psi.
        """
        # If coupled to shaft, update physical boost from shaft RPM
        if self.shaft is not None:
            self.update_from_shaft()
        else:
            self.spin(delta_rpm=packet.batch_size * self.boost_ratio * 0.5)

        # 1. Boost information pressure
        packet.pressure *= self.boost_ratio

        # 2. Informational compression & augmentation under boost
        if self.enable_augmentation and self.boost_ratio > 1.05:
            jitter_scale = self.augmentation_factor * (self.boost_ratio - 1.0)
            noise = np.random.normal(0.0, jitter_scale, size=packet.x.shape)
            packet.x = packet.x + noise
            packet.temperature += float(0.1 * (self.boost_ratio - 1.0))

        # Compute reaction load torque exerted on the shaft
        self.compute_reaction_load()

        self.total_processed_packets += 1
        self.total_processed_samples += packet.batch_size

        packet.metadata["boost_ratio"] = round(self.boost_ratio, 3)
        packet.metadata["manifold_psi"] = round(self.boost_psi, 2)

        self.last_telemetry = {
            "boost_ratio": round(self.boost_ratio, 3),
            "boost_psi": round(self.boost_psi, 2),
            "compression_rpm": round(self.rpm, 1),
            "compressor_load_nm": round(self.current_load_torque, 3),
        }
        return packet
