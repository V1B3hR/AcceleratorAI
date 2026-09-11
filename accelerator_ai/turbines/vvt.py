"""
VariableValveTiming (VVT): Dynamic Camshaft Phasing & Variable Valve Lift.

Inspired by automotive VVT/VTEC systems (e.g. BMW VANOS, Honda VTEC, Toyota VVT-i).
Governs the intake duration and breathing geometry of the combustion chamber.

Physics & ML Analogy:
    In internal combustion, fixed valve timing forces a compromise between low-end torque
    and high-end power. VVT resolves this by advancing camshaft phasing and elevating valve lift
    as engine speed climbs:

    - Low RPM / Spooling Phase:
      Retarded camshaft timing (-15° to 0°) and short valve lift (0.3 - 0.5).
      Limits cylinder intake duration, creating smaller micro-batches (e.g. 16-20 samples).
      This produces sharper, highly concentrated gradient pulses that spool the low-inertia
      HP turbine wheel instantly without bogging the engine down.

    - High RPM / Full Boost Phase:
      Advanced camshaft timing (+25° to +45°) and maximum valve lift (0.8 - 1.0).
      Maximizes volumetric efficiency (eta_v > 100%), inducting expanded micro-batches
      (e.g. 44-64 samples). This prevents localized thermal knocking, broadens stochastic
      gradient averaging, and maximizes throughput when the shaft is spinning at peak RPM.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np


class VariableValveTiming:
    """
    Camshaft phaser, intake valve lift controller, and Discrete Gearbox ("Skrzynia Biegów").
    Dynamically adjusts volumetric efficiency and micro-batch window sizing across 3 static gears.
    """

    def __init__(
        self,
        base_batch_size: int = 32,
        min_batch_size: int = 16,
        max_batch_size: int = 64,
        max_advance_deg: float = 45.0,
        max_retard_deg: float = -30.0,
        gears: Optional[Tuple[int, ...]] = None,
    ):
        self.base_batch_size = base_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.max_advance_deg = max_advance_deg
        self.max_retard_deg = max_retard_deg

        # 3 Discrete Pre-Allocated Gears (Skrzynia Biegów)
        # Prevents constant CUDA Graph invalidations and PyTorch compile JIT recompilations
        if gears is not None:
            self.gears = tuple(sorted(list(set(gears))))
        else:
            self.gears = tuple(sorted(list(set([min_batch_size, base_batch_size, max_batch_size]))))

        # Valve state
        self.cam_advance_deg: float = 0.0     # Camshaft advance angle (-30° to +45°)
        self.valve_lift: float = 0.50          # Valve lift height fraction (0.20 to 1.00)
        self.volumetric_efficiency: float = 0.85 # eta_v (0.40 to 1.30)

        # Gear state: default to cruise gear (Gear 2 if 3 gears available)
        self.current_gear: int = 2 if len(self.gears) >= 2 else 1
        self.current_batch_size: int = self.gears[min(self.current_gear - 1, len(self.gears) - 1)]

    def update(
        self,
        shaft_rpm: float,
        boost_psi: float,
        resonance_index: float = 0.0,
        override_gear: Optional[int] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Dynamically calculates optimal camshaft angle, valve lift, and discrete gearbox selection.

        Args:
            shaft_rpm: Current DriveShaft RPM.
            boost_psi: Manifold pressure gauge PSI.
            resonance_index: Helical resonance H from BraidedDNAController.
            override_gear: Optional master-commanded gear for distributed lockstep synchronization.

        Returns:
            Tuple of (discrete_micro_batch_size, vvt_telemetry_dict).
        """
        # 1. Camshaft Phasing Calculation
        # Low RPM (< 1500) -> Retard cam (-15° to -5°) for rapid exhaust gas scavenging
        # High RPM (> 3000) -> Advance cam (+20° to +45°) for maximum ram-air induction
        rpm_factor = float(np.clip((shaft_rpm - 800.0) / 4200.0, 0.0, 1.0))
        target_advance = self.max_retard_deg * (1.0 - rpm_factor) + self.max_advance_deg * rpm_factor

        # Modulate by DNA resonance: constructive resonance advances cam for higher throughput
        target_advance += resonance_index * 8.0
        self.cam_advance_deg = float(np.clip(target_advance, self.max_retard_deg, self.max_advance_deg))

        # 2. Variable Valve Lift Calculation
        # Boost and RPM combine to lift intake valves higher
        boost_lift_factor = float(np.clip(boost_psi / 22.0, 0.0, 0.5))
        base_lift = 0.35 + (0.45 * rpm_factor) + boost_lift_factor
        self.valve_lift = float(np.clip(base_lift, 0.25, 1.00))

        # 3. Volumetric Efficiency (eta_v)
        # Optimal filling occurs when cam advance matches engine air velocity
        cam_norm = (self.cam_advance_deg - 15.0) / 45.0
        wave_tuning = np.cos(cam_norm * (np.pi / 3.0))
        self.volumetric_efficiency = float(np.clip(0.60 + 0.45 * rpm_factor * wave_tuning * self.valve_lift, 0.40, 1.25))

        # 4. Discrete Gearbox Selection (Skrzynia Biegów)
        # If distributed Master ECU specifies override_gear, lock to commanded gear:
        if override_gear is not None:
            gear_idx = min(max(0, override_gear - 1), len(self.gears) - 1)
        elif shaft_rpm < 1400.0:
            gear_idx = 0
        elif shaft_rpm >= 3000.0 and self.volumetric_efficiency >= 0.85:
            gear_idx = len(self.gears) - 1
        else:
            gear_idx = 1 if len(self.gears) >= 3 else 0

        self.current_gear = gear_idx + 1
        self.current_batch_size = self.gears[gear_idx]

        telemetry = {
            "cam_advance_deg": round(self.cam_advance_deg, 2),
            "valve_lift": round(self.valve_lift, 3),
            "volumetric_efficiency": round(self.volumetric_efficiency, 3),
            "vvt_batch_size": self.current_batch_size,
            "vvt_gear": self.current_gear,
            "vvt_mode": self._get_vvt_mode(),
        }
        return self.current_batch_size, telemetry

    def slice_batch(self, x: Any, y: Any) -> Tuple[Any, Any]:
        """
        Slices an incoming batch to the discrete gearbox micro-batch size.
        Uses contiguous zero-copy slicing (x[:target_n]) to prevent memory allocations
        in CUDA Graph / torch.compile execution.
        """
        target_n = self.current_batch_size
        n = len(x)
        if n == target_n:
            return x, y
        elif n > target_n:
            # Contiguous zero-copy slice for pre-allocated static graph buffers
            return x[:target_n], y[:target_n]
        else:
            # Repeat / oversample if incoming batch is smaller than current gear
            if hasattr(x, "is_cuda") or hasattr(x, "device"):
                import torch
                idx = torch.randint(0, n, (target_n,), device=x.device)
                return x[idx], y[idx]
            indices = np.random.choice(n, size=target_n, replace=True)
            return x[indices], y[indices]

    def _get_vvt_mode(self) -> str:
        if self.cam_advance_deg < -5.0:
            return "RETARD_SCAVENGE"
        elif self.cam_advance_deg > 20.0 and self.valve_lift > 0.75:
            return "VTEC_HIGH_LIFT"
        else:
            return "CONTINUOUS_CRUISE"

    def state_dict(self) -> Dict[str, Any]:
        """Serializes VVT state for checkpointing."""
        return {
            "cam_advance_deg": float(self.cam_advance_deg),
            "valve_lift": float(self.valve_lift),
            "volumetric_efficiency": float(self.volumetric_efficiency),
            "current_batch_size": int(self.current_batch_size),
            "current_gear": int(self.current_gear),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restores VVT state from checkpoint."""
        self.cam_advance_deg = float(state_dict.get("cam_advance_deg", 0.0))
        self.valve_lift = float(state_dict.get("valve_lift", 0.50))
        self.volumetric_efficiency = float(state_dict.get("volumetric_efficiency", 0.85))
        self.current_batch_size = int(state_dict.get("current_batch_size", self.base_batch_size))
        self.current_gear = int(state_dict.get("current_gear", 2))

    def __repr__(self) -> str:
        return (
            f"<VariableValveTiming(gear={self.current_gear}/{len(self.gears)}, mode={self._get_vvt_mode()}, "
            f"advance={self.cam_advance_deg:+.1f}°, lift={self.valve_lift:.2f}, "
            f"eta_v={self.volumetric_efficiency:.2f}, batch={self.current_batch_size})>"
        )

