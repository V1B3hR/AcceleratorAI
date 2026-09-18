"""
EngineState: Central Atomic Single Source of Truth for AcceleratorAI.

Maintains unified, thread-safe, and synchronized state across all turbine stages:
DriveShaft, Compressor, VVT, Wastegate, BraidedECU, Kalman Governor, and VRAM Guard.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import numpy as np


@dataclass
class EngineState:
    """
    Atomic synchronized snapshot of all internal turbine and training dynamics.
    Eliminates fragmented, independent decision-making between subsystems.
    """
    step: int = 0
    epoch: int = 0
    
    # Mechanical and Fluid Kinetics
    rpm: float = 0.0
    boost_psi: float = 0.0
    boost_ratio: float = 1.0
    compressor_load: float = 0.0
    learning_torque: float = 0.0
    pyrometer_temp: float = 200.0
    
    # Transmission & Valve Actuation
    vvt_gear: int = 2
    vvt_batch_size: int = 32
    cam_advance_deg: float = 0.0
    valve_lift: float = 0.5
    wastegate_open_pct: float = 0.0
    
    # Optimizer & Helical ECU
    learning_rate: float = 0.001
    helical_resonance: float = 0.0
    phase_tension: float = 0.0
    
    # Closed-Loop Kalman Dynamics
    raw_loss: float = 1.0
    filtered_loss: float = 1.0
    loss_velocity: float = 0.0
    is_plateau_stall: bool = False
    
    # Proactive Hardware / Memory Awareness
    vram_allocated_bytes: int = 0
    vram_free_bytes: int = 0
    vram_free_pct: float = 100.0
    vram_pressure_level: str = "NORMAL"  # NORMAL, WARNING, CRITICAL, EMERGENCY
    
    # Fault Isolation & Circuit Breakers
    isolated_modules: List[str] = field(default_factory=list)
    
    # Real-Time Profiling & Efficiency
    step_duration_ms: float = 0.0
    model_compute_ms: float = 0.0
    engine_overhead_ms: float = 0.0
    compute_efficiency_pct: float = 100.0

    def update_from_telemetry(self, telemetry: Dict[str, Any]) -> None:
        """Updates internal state fields from a telemetry payload safely."""
        for k, v in telemetry.items():
            if hasattr(self, k) and v is not None:
                setattr(self, k, v)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes state to a dictionary with rounded values."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, (np.floating, float)):
                d[k] = round(float(v), 4)
            elif isinstance(v, (np.integer, int)):
                d[k] = int(v)
        return d
