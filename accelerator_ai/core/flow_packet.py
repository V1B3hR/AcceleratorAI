"""
FlowPacket: The fluid-dynamic medium of information in AcceleratorAI.

In the VIBE Turbine model, training data is not a static list of tensors;
it is a pressurized fluid medium flowing through turbines, manifolds,
and combustion chambers.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class FlowPacket:
    """
    Encapsulates a batch of training data as a fluid-dynamic information packet.
    
    Attributes:
        x (np.ndarray): Input feature tensor.
        y (np.ndarray): Target / label tensor.
        pressure (float): Information pressure (Psi). Higher pressure = higher
                         information density, hard-example weight, or curriculum rank.
        viscosity (float): Resistance to flow (eta). Represents sample complexity,
                           non-linearity, or manifold curvature.
        temperature (float): Thermal entropy (T). Represents variance, noise,
                             or disorder within the batch.
        phase (float): Temporal phase angle (phi, in radians or normalized cycle)
                       used for phase-shifted asynchronous injections.
        source (str): Origin marker ('intake', 'synthetic', 'realworld', 'shock').
        metadata (dict): Diagnostics, routing tags, and telemetry tracing.
    """
    x: np.ndarray
    y: np.ndarray
    pressure: float = 1.0
    viscosity: float = 1.0
    temperature: float = 1.0
    phase: float = 0.0
    source: str = "intake"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        """Returns the number of samples in the packet."""
        return int(self.x.shape[0]) if hasattr(self.x, "shape") and len(self.x.shape) > 0 else 0

    @property
    def flow_rate(self) -> float:
        """
        Flow rate Q = Pressure / Viscosity (Poiseuille-inspired flow analog).
        Indicates throughput efficiency of this informational packet.
        """
        eps = 1e-6
        return float(self.pressure / (self.viscosity + eps))

    def split(self, fraction: float) -> tuple["FlowPacket", "FlowPacket"]:
        """Splits the packet into two sub-streams preserving physical attributes."""
        split_idx = int(self.batch_size * fraction)
        p1 = FlowPacket(
            x=self.x[:split_idx],
            y=self.y[:split_idx],
            pressure=self.pressure,
            viscosity=self.viscosity,
            temperature=self.temperature,
            phase=self.phase,
            source=self.source,
            metadata=self.metadata.copy(),
        )
        p2 = FlowPacket(
            x=self.x[split_idx:],
            y=self.y[split_idx:],
            pressure=self.pressure,
            viscosity=self.viscosity,
            temperature=self.temperature,
            phase=self.phase,
            source=self.source,
            metadata=self.metadata.copy(),
        )
        return p1, p2

    @classmethod
    def merge(cls, packets: list["FlowPacket"], source_tag: str = "fused") -> "FlowPacket":
        """
        Blends multiple flow packets into a unified fluid stream.
        Pressure and temperature are mass-weighted averages.
        """
        valid_packets = [p for p in packets if p is not None and p.batch_size > 0]
        if not valid_packets:
            raise ValueError("Cannot merge empty packet list")

        total_samples = sum(p.batch_size for p in valid_packets)
        xs = [p.x for p in valid_packets]
        ys = [p.y for p in valid_packets]

        merged_x = np.concatenate(xs, axis=0)
        merged_y = np.concatenate(ys, axis=0)

        # Mass-weighted pressure and temperature
        weighted_pressure = sum(p.pressure * p.batch_size for p in valid_packets) / total_samples
        weighted_viscosity = sum(p.viscosity * p.batch_size for p in valid_packets) / total_samples
        weighted_temp = sum(p.temperature * p.batch_size for p in valid_packets) / total_samples

        return cls(
            x=merged_x,
            y=merged_y,
            pressure=float(weighted_pressure),
            viscosity=float(weighted_viscosity),
            temperature=float(weighted_temp),
            phase=valid_packets[0].phase,
            source=source_tag,
            metadata={"merged_sources": [p.source for p in valid_packets]},
        )
