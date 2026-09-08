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
    rpm: float = 0.0                      # Physical DriveShaft RPM
    boost_psi: float = 0.0                # Intake manifold boost pressure (psi)
    manifold_pressure: float = 1.0        # Normalized pressure ratio Psi
    pyrometer_temp_c: float = 200.0       # Exhaust / Loss thermal index (degrees C analog)
    loss: float = 0.0                     # Current combustion loss
    learning_torque_nm: float = 0.0       # Gradient magnitude * boost (Nm analog)
    compressor_load_nm: float = 0.0       # Mechanical reaction load of compressing fluid
    shaft_kinetic_energy_j: float = 0.0   # Stored mechanical rotational kinetic energy (0.5 * I * omega^2)
    angular_accel_rad_s2: float = 0.0     # DriveShaft rotational acceleration
    wastegate_open_pct: float = 0.0       # Wastegate relief percentage (0 - 100%)
    air_fuel_ratio: float = 14.7          # Ratio of base intake to injected supplemental fuel
    injected_entropy: float = 0.0         # Chaos perturbation magnitude injected
    active_injectors: int = 0             # Number of async injectors active in this cycle
    learning_rate: float = 0.001          # Current Braided ECU-tuned learning rate
    helical_resonance: float = 0.0        # DNA Braided resonance index H in [-1.0, 1.0]
    phase_tension: float = 0.0            # Braided multi-strand tension (creative friction)
    winding_number: float = 0.0           # Helical winding revolutions
    homogeneity_pct: float = 100.0        # Swirl dispersion & atomization homogeneity index (0 - 100%)
    hyper_flow_aperture: float = 0.15     # VGT hyper-flow port aperture (0.05 - 1.0)
    cruise_flow_aperture: float = 0.50    # VGT cruise port aperture
    slowmo_flow_aperture: float = 0.85    # VGT slow-mo port aperture
    hyper_flow_pct: float = 0.0           # % of samples routed through hyper-flow port
    curriculum_weight_mean: float = 1.0   # Mean curriculum weight from VGT routing
    sequential_stage: str = "HP_PRIMARY"  # Sequential Turbo stage (HP_PRIMARY, TRANSITION, LP_COMPOUND)
    transition_valve_pct: float = 0.0     # Sequential bypass transition valve % (0 - 100%)
    hp_rpm: float = 1200.0                # High-Pressure low-inertia turbo RPM
    lp_rpm: float = 600.0                 # Low-Pressure compound turbo RPM
    cam_advance_deg: float = 0.0          # VVT Camshaft advance angle (-30° to +45°)
    valve_lift: float = 0.50              # VVT Intake valve lift fraction (0.25 - 1.00)
    volumetric_efficiency: float = 0.85   # VVT Volumetric efficiency eta_v
    twin_scroll_balance: float = 1.0      # Twin-Scroll exhaust pulse balance (Scroll A / (A+B))

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
