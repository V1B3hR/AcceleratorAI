"""
SequentialTurboSystem: Two-Stage Sequential Turbocharging Architecture.

Solves the classic turbomachinery dilemma:
    - Small turbo: Spools instantly on low exhaust mass flow (eliminates turbo lag / "turbodziura"),
      but chokes at high engine RPM and high information density.
    - Large turbo: Delivers massive volumetric flow and high boost ceiling,
      but has heavy rotational inertia and suffers from significant lag at low RPM.

Physical Architecture:
    1. HP Turbo (High-Pressure / Low-Inertia Wheel):
       Low rotational inertia (I_HP = 0.012 kg·m²).
       Active from idle. Reacts instantly to subtle gradient changes, providing
       immediate informational boost during early training steps or after loss shocks.

    2. LP Turbo (Low-Pressure / High-Capacity Wheel):
       High rotational inertia (I_LP = 0.085 kg·m²).
       Large wheel diameter. Engages via the Sequential Transition Valve as shaft RPM
       crosses the crossover threshold, taking over to deliver sustained compound boost.

    3. Sequential Transition Valve (By-pass / Change-over Valve):
       Electronically modulated bypass gate transitioning smoothly across 3 operating regimes:
       - HP_PRIMARY   (RPM < 2200): Valve closed (0%). Exhaust energy focused on HP wheel.
       - TRANSITION   (2200 <= RPM <= 3400): Valve opens linearly (0% -> 100%). Pre-spools LP.
       - LP_COMPOUND  (RPM > 3400): Valve 100% open. Dual-stage compound compression.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np


class HPTurbo:
    """
    High-Pressure, low-inertia turbocharger wheel.
    Optimized for ultra-responsive low-end spooling.
    """

    def __init__(
        self,
        inertia: float = 0.012,
        idle_rpm: float = 1200.0,
        max_rpm: float = 12000.0,
        pressure_coeff: float = 0.35,
    ):
        self.inertia = inertia
        self.idle_rpm = idle_rpm
        self.max_rpm = max_rpm
        self.pressure_coeff = pressure_coeff

        self.rpm: float = idle_rpm
        self.boost_ratio: float = 1.0
        self.load_torque: float = 0.0

    def update(self, drive_torque: float, dt: float = 0.05) -> float:
        """
        Accelerates HP turbine wheel with low rotational inertia.
        Returns HP stage pressure ratio.
        """
        # Euler load torque opposes rotation: tau_load = c * (rpm / 1000)^2
        self.load_torque = 0.004 * (self.rpm / 1000.0) ** 1.8
        net_torque = drive_torque - self.load_torque

        # Angular acceleration domega/dt = tau / I
        # Conversion: omega (rad/s) = rpm * 2*pi / 60 = rpm * 0.10472
        domega_dt = net_torque / self.inertia
        drpm = (domega_dt * 60.0 / (2.0 * np.pi)) * dt

        self.rpm = float(np.clip(self.rpm + drpm, self.idle_rpm, self.max_rpm))

        # Pressure ratio derived from blade tip velocity
        self.boost_ratio = float(1.0 + self.pressure_coeff * ((self.rpm - self.idle_rpm) / 1000.0) ** 1.25)
        return self.boost_ratio


class LPTurbo:
    """
    Low-Pressure, high-capacity turbine wheel.
    Optimized for sustained compound boost at medium-to-high RPM.
    """

    def __init__(
        self,
        inertia: float = 0.085,
        idle_rpm: float = 600.0,
        max_rpm: float = 8500.0,
        pressure_coeff: float = 0.55,
    ):
        self.inertia = inertia
        self.idle_rpm = idle_rpm
        self.max_rpm = max_rpm
        self.pressure_coeff = pressure_coeff

        self.rpm: float = idle_rpm
        self.boost_ratio: float = 1.0
        self.load_torque: float = 0.0

    def update(self, drive_torque: float, dt: float = 0.05) -> float:
        """
        Accelerates LP turbine wheel with high rotational inertia.
        Returns LP stage pressure ratio.
        """
        self.load_torque = 0.012 * (self.rpm / 1000.0) ** 2.0
        net_torque = drive_torque - self.load_torque

        domega_dt = net_torque / self.inertia
        drpm = (domega_dt * 60.0 / (2.0 * np.pi)) * dt

        self.rpm = float(np.clip(self.rpm + drpm, self.idle_rpm, self.max_rpm))
        self.boost_ratio = float(1.0 + self.pressure_coeff * ((self.rpm - self.idle_rpm) / 1000.0) ** 1.5)
        return self.boost_ratio


class SequentialTurboSystem:
    """
    Coordinates HP and LP turbochargers with an electronic transition valve.
    Delivers zero-lag transient response coupled with high sustained compound boost.
    """

    def __init__(
        self,
        crossover_rpm: float = 2200.0,
        full_transition_rpm: float = 3400.0,
        hp_turbo: Optional[HPTurbo] = None,
        lp_turbo: Optional[LPTurbo] = None,
    ):
        self.crossover_rpm = crossover_rpm
        self.full_transition_rpm = full_transition_rpm
        self.hp = hp_turbo or HPTurbo()
        self.lp = lp_turbo or LPTurbo()

        self.transition_valve_pct: float = 0.0  # 0.0 = closed (HP only), 100.0 = full LP compound
        self.stage: str = "HP_PRIMARY"          # HP_PRIMARY | TRANSITION | LP_COMPOUND
        self.total_boost_ratio: float = 1.0
        self.last_telemetry: Dict[str, Any] = {}

    @property
    def total_boost_psi(self) -> float:
        """Converts total pressure ratio to gauge PSI (1.0 ratio = 0.0 PSI, 2.0 ratio = 14.7 PSI)."""
        return max(0.0, (self.total_boost_ratio - 1.0) * 14.7)

    def update(
        self,
        learning_torque: float,
        shaft_rpm: float,
        dt: float = 0.05,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Executes one physical step of the sequential turbocharging system.

        Args:
            learning_torque: Kinetic torque generated by exhaust backprop gradients.
            shaft_rpm: Current RPM of the primary mechanical DriveShaft.
            dt: Simulation timestep.

        Returns:
            Tuple of (compound_boost_ratio, sequential_telemetry_dict).
        """
        # 1. Determine Sequential Transition Valve State from Shaft RPM
        if shaft_rpm < self.crossover_rpm:
            # Stage 1: HP Primary (Bypass closed, all exhaust drives HP turbo)
            self.transition_valve_pct = 0.0
            self.stage = "HP_PRIMARY"
            torque_hp = learning_torque
            torque_lp = learning_torque * 0.08  # Slight pre-spool seepage
        elif shaft_rpm <= self.full_transition_rpm:
            # Stage 2: Smooth Transition (Valve opening linearly)
            progress = (shaft_rpm - self.crossover_rpm) / (self.full_transition_rpm - self.crossover_rpm)
            self.transition_valve_pct = float(progress * 100.0)
            self.stage = "TRANSITION"
            torque_hp = learning_torque * (1.0 - 0.45 * progress)
            torque_lp = learning_torque * (0.08 + 0.82 * progress)
        else:
            # Stage 3: LP Compound (Valve fully open, maximum dual-stage throughput)
            self.transition_valve_pct = 100.0
            self.stage = "LP_COMPOUND"
            torque_hp = learning_torque * 0.55
            torque_lp = learning_torque * 0.90

        # 2. Update HP and LP wheels
        hp_boost = self.hp.update(torque_hp, dt=dt)
        lp_boost = self.lp.update(torque_lp, dt=dt)

        # 3. Compute Compound Pressure Ratio
        # In HP_PRIMARY: boost = hp_boost
        # In LP_COMPOUND: compound series compression: Psi_total = Psi_HP * (1 + alpha * (Psi_LP - 1))
        alpha = self.transition_valve_pct / 100.0
        self.total_boost_ratio = float(hp_boost * (1.0 + alpha * (lp_boost - 1.0)))

        self.last_telemetry = {
            "sequential_stage": self.stage,
            "transition_valve_pct": round(self.transition_valve_pct, 1),
            "hp_rpm": round(self.hp.rpm, 1),
            "lp_rpm": round(self.lp.rpm, 1),
            "hp_boost_ratio": round(hp_boost, 3),
            "lp_boost_ratio": round(lp_boost, 3),
            "total_boost_ratio": round(self.total_boost_ratio, 3),
            "total_boost_psi": round(self.total_boost_psi, 2),
        }
        return self.total_boost_ratio, self.last_telemetry

    def reset(self) -> None:
        """Resets both turbos to idle states."""
        self.hp.rpm = self.hp.idle_rpm
        self.hp.boost_ratio = 1.0
        self.lp.rpm = self.lp.idle_rpm
        self.lp.boost_ratio = 1.0
        self.total_boost_ratio = 1.0
        self.transition_valve_pct = 0.0
        self.stage = "HP_PRIMARY"

    def __repr__(self) -> str:
        return (
            f"<SequentialTurboSystem(stage={self.stage}, valve={self.transition_valve_pct:.1f}%, "
            f"HP_RPM={self.hp.rpm:.0f}, LP_RPM={self.lp.rpm:.0f}, boost={self.total_boost_psi:.1f} PSI)>"
        )
