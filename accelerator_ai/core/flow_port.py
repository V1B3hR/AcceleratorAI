"""
FlowPort & PortManifold: Variable Geometry Multi-Port Turbine System.

Implements the Venturi-based variable-geometry orifice model for AcceleratorAI turbines.
Each turbine stage can have multiple ports with adjustable apertures. Narrower apertures
create higher velocity flow (Venturi effect), routing harder samples through more
intense processing. Wider apertures create gentle, exploratory slow-mo flow.

Physics:
    Continuity equation:  A₁·v₁ = A₂·v₂
    Port velocity factor: v_port = 1.0 / aperture
    Curriculum weight:    w_i = velocity_factor / mean(velocity_factors)

This creates a natural curriculum-from-physics: hard samples (high information pressure)
are routed through narrow hyper-flow ports and receive higher loss weighting,
while easy samples flow through wide slow-mo ports with gentler treatment.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from accelerator_ai.core.flow_packet import FlowPacket


class FlowPort:
    """
    A single variable-geometry orifice on a turbine manifold.

    Attributes:
        name: Human-readable port identifier.
        mode: Port category — "hyper", "cruise", or "slowmo".
        aperture: Current opening width (0.05 = nearly closed, 1.0 = wide open).
        velocity_factor: Derived Venturi velocity multiplier (1.0 / aperture).
        selectivity_range: (low, high) bounds on normalized sample pressure [0, 1]
                          that this port admits.
    """

    def __init__(
        self,
        name: str,
        mode: str = "cruise",
        aperture: float = 0.50,
        min_aperture: float = 0.05,
        max_aperture: float = 1.0,
        selectivity_range: Tuple[float, float] = (0.0, 1.0),
    ):
        self.name = name
        self.mode = mode
        self.min_aperture = min_aperture
        self.max_aperture = max_aperture
        self.selectivity_range = selectivity_range

        self._aperture: float = 0.0
        self.aperture = aperture  # Uses property setter

        # Telemetry counters
        self.total_routed_samples: int = 0
        self.last_routed_count: int = 0

    @property
    def aperture(self) -> float:
        return self._aperture

    @aperture.setter
    def aperture(self, value: float) -> None:
        self._aperture = float(np.clip(value, self.min_aperture, self.max_aperture))

    @property
    def velocity_factor(self) -> float:
        """Venturi velocity: inversely proportional to aperture."""
        return 1.0 / self._aperture

    def admits(self, sample_pressures: np.ndarray) -> np.ndarray:
        """
        Returns a boolean mask: True for samples whose normalized pressure
        falls within this port's selectivity range.

        Args:
            sample_pressures: Per-sample information pressure, normalized [0, 1].

        Returns:
            Boolean mask of shape (n_samples,).
        """
        low, high = self.selectivity_range
        return (sample_pressures >= low) & (sample_pressures <= high)

    def adjust(self, delta: float) -> None:
        """Narrow (negative delta) or widen (positive delta) the aperture."""
        self.aperture = self._aperture + delta

    def adjust_towards(self, target: float, speed: float = 0.1) -> None:
        """Smoothly move aperture towards a target value."""
        diff = target - self._aperture
        self.aperture = self._aperture + (diff * speed)

    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "aperture": round(self._aperture, 4),
            "velocity_factor": round(self.velocity_factor, 2),
            "selectivity_range": self.selectivity_range,
            "last_routed_count": self.last_routed_count,
            "total_routed_samples": self.total_routed_samples,
        }

    def __repr__(self) -> str:
        return (
            f"<FlowPort('{self.name}', mode={self.mode}, "
            f"aperture={self._aperture:.3f}, v={self.velocity_factor:.1f}x)>"
        )


class PortManifold:
    """
    Collection of FlowPorts on a turbine inlet or exhaust.

    Routes incoming FlowPackets through appropriate ports based on per-sample
    information pressure, applies Venturi-physics velocity transforms, and
    recombines into a unified FlowPacket with per-sample curriculum weights.
    """

    def __init__(
        self,
        ports: Optional[List[FlowPort]] = None,
        side: str = "inlet",
    ):
        self.side = side
        self.ports: List[FlowPort] = ports or self._default_inlet_ports()
        self.last_pressure_distribution: np.ndarray = np.array([])
        self.last_weight_mean: float = 1.0

    @staticmethod
    def _default_inlet_ports() -> List[FlowPort]:
        """Creates the standard 3-port variable-geometry inlet manifold."""
        return [
            FlowPort(
                name="hyper_flow",
                mode="hyper",
                aperture=0.15,
                selectivity_range=(0.70, 1.0),
            ),
            FlowPort(
                name="cruise_flow",
                mode="cruise",
                aperture=0.50,
                selectivity_range=(0.30, 0.70),
            ),
            FlowPort(
                name="slowmo_flow",
                mode="slowmo",
                aperture=0.85,
                selectivity_range=(0.0, 0.30),
            ),
        ]

    @staticmethod
    def compute_sample_pressure(packet: FlowPacket) -> np.ndarray:
        """
        Computes per-sample information pressure from feature energy.

        Uses normalized L2 feature norm as the physical pressure metric.
        Samples with higher feature energy (larger norms, more distant from
        manifold center) have higher information pressure — analogous to
        denser gas molecules exerting more force on turbine vanes.

        Returns:
            Per-sample pressure in [0, 1], shape (batch_size,).
        """
        x = packet.x
        norms = np.linalg.norm(x, axis=1)

        # Normalize to [0, 1] using robust min-max
        norm_min = np.min(norms)
        norm_range = np.max(norms) - norm_min
        if norm_range < 1e-8:
            # All samples have equal pressure → all cruise
            return np.full(len(norms), 0.5)

        return (norms - norm_min) / norm_range

    def route_and_transform(self, packet: FlowPacket) -> FlowPacket:
        """
        Routes samples through ports based on per-sample pressure, applies
        Venturi velocity transforms, and recombines with curriculum weights.

        The curriculum weight for each sample is proportional to its port's
        velocity factor, normalized so mean(weights) ≈ 1.0. This means
        hyper-flow samples contribute more to the loss gradient, while
        slow-mo samples contribute less — natural curriculum weighting
        from physics.

        Returns:
            FlowPacket with `metadata["curriculum_weights"]` stamped.
        """
        sample_pressures = self.compute_sample_pressure(packet)
        self.last_pressure_distribution = sample_pressures
        n = packet.batch_size

        # Assign per-sample curriculum weight based on port routing
        weights = np.ones(n, dtype=np.float64)
        port_assignments = np.full(n, -1, dtype=np.int32)

        for port_idx, port in enumerate(self.ports):
            mask = port.admits(sample_pressures)
            count = int(np.sum(mask))
            port.last_routed_count = count
            port.total_routed_samples += count

            if count > 0:
                weights[mask] = port.velocity_factor
                port_assignments[mask] = port_idx

        # Handle samples that fall into no port (shouldn't happen with
        # proper selectivity ranges, but safety net)
        unassigned = port_assignments == -1
        if np.any(unassigned):
            # Default to cruise velocity
            weights[unassigned] = 2.0

        # Normalize weights so mean ≈ 1.0 (preserves overall gradient scale)
        weight_mean = np.mean(weights)
        if weight_mean > 1e-8:
            weights = weights / weight_mean

        self.last_weight_mean = float(np.mean(weights * weight_mean))

        # Apply Venturi-physics temperature effect:
        # High velocity flow → adiabatic cooling (lower temperature on fast samples)
        # Low velocity flow → higher temperature (more thermal exploration)
        # This is stored but doesn't modify the packet directly — the weights
        # are the primary mechanism of action.

        # Stamp curriculum weights into packet metadata
        packet.metadata["curriculum_weights"] = weights
        packet.metadata["sample_pressures"] = sample_pressures
        packet.metadata["port_assignments"] = port_assignments

        return packet

    def get_port(self, mode: str) -> Optional[FlowPort]:
        """Get a port by its mode name."""
        for port in self.ports:
            if port.mode == mode:
                return port
        return None

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns telemetry for all ports in the manifold."""
        return {
            "side": self.side,
            "num_ports": len(self.ports),
            "weight_mean": round(self.last_weight_mean, 3),
            "ports": [p.get_telemetry() for p in self.ports],
        }

    def __repr__(self) -> str:
        port_str = ", ".join(str(p) for p in self.ports)
        return f"<PortManifold(side={self.side}, ports=[{port_str}])>"
