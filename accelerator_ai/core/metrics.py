"""
Turbine Metrics: Fluid dynamics and mechanical metrics for AI training telemetry.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any
import numpy as np


@dataclass
class EngineTelemetry:
    """
    Real-time telemetry snapshot of the AcceleratorAI turbine engine.
    """
    step: int = 0
    epoch: int = 0
    rpm: float = 0.0                      # Virtual engine RPM (training velocity)
    boost_psi: float = 0.0                # Intake manifold boost pressure (psi)
    manifold_pressure: float = 1.0        # Normalized pressure ratio Psi
    pyrometer_temp_c: float = 200.0       # Exhaust / Loss thermal index (degrees C analog)
    loss: float = 0.0                     # Current combustion loss
    learning_torque_nm: float = 0.0       # Gradient magnitude * boost (Nm analog)
    wastegate_open_pct: float = 0.0       # Wastegate relief percentage (0 - 100%)
    air_fuel_ratio: float = 14.7          # Ratio of base intake to injected supplemental fuel
    injected_entropy: float = 0.0         # Chaos perturbation magnitude injected
    active_injectors: int = 0             # Number of async injectors active in this cycle
    learning_rate: float = 0.001          # Current ECU-tuned learning rate

    def to_dict(self) -> Dict[str, Any]:
        """Convert telemetry to serialized dictionary for JSON / WebSocket streaming."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, (np.floating, float)):
                d[k] = round(float(v), 4)
            elif isinstance(v, (np.integer, int)):
                d[k] = int(v)
        return d


def calculate_learning_torque(gradient_norm: float, boost_ratio: float) -> float:
    """
    Learning Torque tau = ||grad|| * Boost.
    Measures the rotational work applied to the weight tensor parameter space.
    """
    return float(gradient_norm * max(1.0, boost_ratio))


def calculate_pyrometer_temp(loss: float, base_temp: float = 200.0, max_temp: float = 950.0) -> float:
    """
    Exhaust gas temperature (Pyrometer) analog.
    High loss or gradient volatility generates intense heat.
    """
    scaled = base_temp + (loss * 150.0)
    return float(min(max_temp, max(base_temp, scaled)))
