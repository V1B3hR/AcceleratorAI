"""
Compressor Turbine: The intake compressor wheel of the turbocharger.
Increases Information Pressure (Psi) and concentrates high-value features.
"""

from typing import Optional
import numpy as np
from accelerator_ai.core.base_turbine import TurbineModule
from accelerator_ai.core.flow_packet import FlowPacket


class CompressorTurbine(TurbineModule):
    """
    Compressor stage simulating the centrifugal turbocharger compressor wheel.
    Compresses input representations, amplifies information density,
    and mines high-gradient potential samples (hard examples).
    """

    def __init__(
        self,
        base_boost_ratio: float = 1.0,
        max_boost_psi: float = 28.0,
        enable_augmentation: bool = True,
        augmentation_factor: float = 0.05,
    ):
        super().__init__(name="CompressorTurbine")
        self.boost_ratio = base_boost_ratio  # 1.0 = atmospheric (0 psi boost)
        self.max_boost_psi = max_boost_psi
        self.enable_augmentation = enable_augmentation
        self.augmentation_factor = augmentation_factor

    @property
    def boost_psi(self) -> float:
        """Converts pressure ratio to gauge psi (1.0 ratio = 0 psi gauge; 2.0 ratio = 14.7 psi)."""
        return max(0.0, (self.boost_ratio - 1.0) * 14.7)

    def set_boost(self, boost_ratio: float) -> None:
        """Sets target boost ratio, clamped by max gauge pressure."""
        max_ratio = 1.0 + (self.max_boost_psi / 14.7)
        self.boost_ratio = float(np.clip(boost_ratio, 1.0, max_ratio))

    def process(self, packet: FlowPacket) -> FlowPacket:
        """
        Compresses the incoming flow packet, elevating information pressure Psi.
        """
        # 1. Boost information pressure
        packet.pressure *= self.boost_ratio

        # 2. Informational compression & augmentation under boost
        if self.enable_augmentation and self.boost_ratio > 1.05:
            # Dynamic augmentation jitter proportional to boost pressure
            jitter_scale = self.augmentation_factor * (self.boost_ratio - 1.0)
            noise = np.random.normal(0.0, jitter_scale, size=packet.x.shape)
            packet.x = packet.x + noise
            # Higher pressure causes moderate heating (entropy increase)
            packet.temperature += float(0.1 * (self.boost_ratio - 1.0))

        # 3. Spin compressor rotor
        self.spin(delta_rpm=packet.batch_size * self.boost_ratio * 0.5)

        self.total_processed_packets += 1
        self.total_processed_samples += packet.batch_size

        packet.metadata["boost_ratio"] = round(self.boost_ratio, 3)
        packet.metadata["manifold_psi"] = round(self.boost_psi, 2)

        self.last_telemetry = {
            "boost_ratio": round(self.boost_ratio, 3),
            "boost_psi": round(self.boost_psi, 2),
            "compression_rpm": round(self.rpm, 1),
        }
        return packet
