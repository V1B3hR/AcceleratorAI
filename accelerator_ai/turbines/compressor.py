"""
Compressor Turbine: Centrifugal intake compressor wheel physically coupled to the DriveShaft.

Governed by Euler's turbomachinery equation:
    Pressure Rise Delta_P proportional to (Blade Tip Velocity)^2
    Compressor Load Torque proportional to Flow Mass * Pressure Delta

Now equipped with a Variable Geometry inlet PortManifold:
    Hyper-flow port (narrow)  → high-velocity processing for hard samples
    Cruise port (medium)      → standard throughput
    Slow-mo port (wide)       → gentle exploratory processing for easy samples
"""

from typing import Optional, List
import numpy as np
from accelerator_ai.core.base_turbine import TurbineModule
from accelerator_ai.core.flow_packet import FlowPacket
from accelerator_ai.core.shaft import DriveShaft
from accelerator_ai.core.flow_port import FlowPort, PortManifold


class CompressorTurbine(TurbineModule):
    """
    Compressor stage simulating the centrifugal turbocharger compressor wheel.
    Compresses input representations, amplifies information density,
    and mines high-gradient potential samples (hard examples).
    
    Equipped with a Variable Geometry inlet PortManifold that routes samples
    through different ports based on per-sample information pressure,
    creating natural curriculum-from-physics.
    """

    def __init__(
        self,
        base_boost_ratio: float = 1.0,
        max_boost_psi: float = 28.0,
        enable_augmentation: bool = False,
        augmentation_factor: float = 0.05,
        shaft: Optional[DriveShaft] = None,
        pressure_coefficient: float = 0.45,
        inlet_manifold: Optional[PortManifold] = None,
    ):
        super().__init__(name="CompressorTurbine")
        self.boost_ratio = base_boost_ratio  # 1.0 = atmospheric (0 psi boost)
        self.max_boost_psi = max_boost_psi
        self.enable_augmentation = enable_augmentation
        self.augmentation_factor = augmentation_factor
        self.shaft = shaft
        self.pressure_coefficient = pressure_coefficient
        self.current_load_torque: float = 0.0

        # Variable Geometry inlet PortManifold
        self.inlet_manifold = inlet_manifold or PortManifold(side="inlet")

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
        Routes samples through the Variable Geometry inlet manifold to assign
        per-sample curriculum weights based on Venturi flow physics.
        """
        # If coupled to shaft, update physical boost from shaft RPM
        if self.shaft is not None:
            self.update_from_shaft()
        else:
            self.spin(delta_rpm=packet.batch_size * self.boost_ratio * 0.5)

        # --- Variable Geometry Port Routing ---
        # Route samples through the inlet manifold: stamps curriculum_weights
        packet = self.inlet_manifold.route_and_transform(packet)
        curriculum_weights = packet.metadata.get("curriculum_weights", None)

        # 1. Boost information pressure
        packet.pressure *= self.boost_ratio

        # 2. Informational compression & augmentation under boost
        # Augmentation intensity is modulated per-sample by port velocity:
        # Hyper-flow samples get LESS noise (preserve their hard-example signal)
        # Slow-mo samples get MORE noise (exploratory augmentation)
        if self.enable_augmentation and self.boost_ratio > 1.05:
            is_float = True
            if hasattr(packet.x, "is_floating_point"):
                is_float = packet.x.is_floating_point()
            elif hasattr(packet.x, "dtype") and not np.issubdtype(packet.x.dtype, np.floating):
                is_float = False

            if is_float:
                base_jitter = self.augmentation_factor * (self.boost_ratio - 1.0)
                if hasattr(packet.x, "is_cuda"):
                    import torch
                    noise = torch.randn_like(packet.x) * base_jitter
                    packet.x = packet.x + noise
                else:
                    if curriculum_weights is not None:
                        inv_weights = np.clip(2.0 - curriculum_weights, 0.5, 2.0)
                        shape_broadcast = [packet.x.shape[0]] + [1] * (packet.x.ndim - 1)
                        per_sample_jitter = base_jitter * inv_weights.reshape(shape_broadcast)
                        noise = np.random.normal(0.0, 1.0, size=packet.x.shape) * per_sample_jitter
                    else:
                        noise = np.random.normal(0.0, base_jitter, size=packet.x.shape)
                    packet.x = packet.x + noise
                packet.temperature += float(0.1 * (self.boost_ratio - 1.0))

        # Compute reaction load torque exerted on the shaft
        self.compute_reaction_load()

        self.total_processed_packets += 1
        self.total_processed_samples += packet.batch_size

        packet.metadata["boost_ratio"] = round(self.boost_ratio, 3)
        packet.metadata["manifold_psi"] = round(self.boost_psi, 2)

        # Port telemetry
        hyper_port = self.inlet_manifold.get_port("hyper")
        cruise_port = self.inlet_manifold.get_port("cruise")
        slowmo_port = self.inlet_manifold.get_port("slowmo")

        self.last_telemetry = {
            "boost_ratio": round(self.boost_ratio, 3),
            "boost_psi": round(self.boost_psi, 2),
            "compression_rpm": round(self.rpm, 1),
            "compressor_load_nm": round(self.current_load_torque, 3),
            "hyper_flow_aperture": round(hyper_port.aperture, 3) if hyper_port else 0.0,
            "cruise_flow_aperture": round(cruise_port.aperture, 3) if cruise_port else 0.0,
            "slowmo_flow_aperture": round(slowmo_port.aperture, 3) if slowmo_port else 0.0,
            "hyper_flow_samples": hyper_port.last_routed_count if hyper_port else 0,
            "slowmo_flow_samples": slowmo_port.last_routed_count if slowmo_port else 0,
        }
        return packet

