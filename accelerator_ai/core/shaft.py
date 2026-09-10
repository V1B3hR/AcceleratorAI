"""
DriveShaft: Physical mechanical link coupling the Gradient Turbine and Compressor Wheel.

Implements true mechanical inertia, angular acceleration, kinetic energy storage,
and frictional drag. Eliminates heuristic telemetry by making rotor RPM a true
physical state of the system governed by Newton-Euler rotational dynamics:
    I * d(omega)/dt = tau_in - tau_load - beta * omega
"""

from typing import Dict, Any, Optional
import numpy as np


class DriveShaft:
    """
    Physical rotating shaft connecting the hot exhaust turbine to the cold intake compressor.
    
    Attributes:
        inertia (float): Mass moment of inertia (I, in kg*m^2 analog).
        omega (float): Angular velocity (rad/s).
        idle_rpm (float): Baseline idle rotational speed.
        friction_coeff (float): Viscous bearing drag coefficient (beta).
        kinetic_energy (float): Stored rotational energy (0.5 * I * omega^2).
    """

    def __init__(
        self,
        inertia: float = 0.08,
        idle_rpm: float = 800.0,
        friction_coeff: float = 0.015,
        max_rpm: float = 12000.0,
    ):
        self.inertia = max(1e-4, inertia)
        self.idle_rpm = idle_rpm
        self.idle_omega = (idle_rpm * 2.0 * np.pi) / 60.0
        self.omega = self.idle_omega
        self.friction_coeff = friction_coeff
        self.max_rpm = max_rpm
        self.max_omega = (max_rpm * 2.0 * np.pi) / 60.0

        self.last_torque_in: float = 0.0
        self.last_load_torque: float = 0.0
        self.last_angular_accel: float = 0.0
        self.total_revolutions: float = 0.0

    @property
    def rpm(self) -> float:
        """Converts angular velocity (rad/s) to RPM."""
        return float((self.omega * 60.0) / (2.0 * np.pi))

    @property
    def kinetic_energy(self) -> float:
        """Rotational kinetic energy E_k = 0.5 * I * omega^2 (Joules analog)."""
        return float(0.5 * self.inertia * (self.omega ** 2))

    def step(self, torque_in: float, load_torque: float, dt: float = 0.05) -> float:
        """
        Integrates rotational dynamics over time step dt.
        
        Args:
            torque_in (float): Driving torque from Gradient Turbine (Nm).
            load_torque (float): Reaction load from Compressor compressing air (Nm).
            dt (float): Time integration step (seconds).
            
        Returns:
            Current RPM.
        """
        self.last_torque_in = float(torque_in)
        self.last_load_torque = float(load_torque)

        # Viscous bearing drag above idle = beta * max(0.0, omega - idle_omega)
        drag_torque = self.friction_coeff * max(0.0, self.omega - self.idle_omega)

        # Net torque = Driving Torque - Compressor Load - Bearing Drag
        net_torque = torque_in - load_torque - drag_torque

        # Angular acceleration alpha = Net Torque / Inertia
        alpha = net_torque / self.inertia
        self.last_angular_accel = float(alpha)

        # Integrate angular velocity: omega = omega + alpha * dt
        new_omega = self.omega + (alpha * dt)

        # Clamp between idle_omega (engine keeps idling) and max_omega
        self.omega = float(np.clip(new_omega, self.idle_omega, self.max_omega))

        # Track revolutions
        revolutions_in_step = (self.omega * dt) / (2.0 * np.pi)
        self.total_revolutions += revolutions_in_step

        return self.rpm

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns real-time mechanical state of the drive shaft."""
        return {
            "shaft_rpm": round(self.rpm, 1),
            "angular_velocity_rad_s": round(self.omega, 2),
            "angular_accel_rad_s2": round(self.last_angular_accel, 2),
            "stored_kinetic_energy_j": round(self.kinetic_energy, 2),
            "driving_torque_nm": round(self.last_torque_in, 3),
            "compressor_load_nm": round(self.last_load_torque, 3),
            "total_revolutions": round(self.total_revolutions, 1),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Serializes drive shaft physical dynamics for checkpointing."""
        return {
            "omega": float(self.omega),
            "last_torque_in": float(self.last_torque_in),
            "last_load_torque": float(self.last_load_torque),
            "last_angular_accel": float(self.last_angular_accel),
            "total_revolutions": float(self.total_revolutions),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restores drive shaft physical state from checkpoint."""
        self.omega = float(state_dict.get("omega", self.idle_omega))
        self.last_torque_in = float(state_dict.get("last_torque_in", 0.0))
        self.last_load_torque = float(state_dict.get("last_load_torque", 0.0))
        self.last_angular_accel = float(state_dict.get("last_angular_accel", 0.0))
        self.total_revolutions = float(state_dict.get("total_revolutions", 0.0))

    def __repr__(self) -> str:
        return (
            f"<DriveShaft(rpm={self.rpm:.1f}, inertia={self.inertia}, "
            f"Ek={self.kinetic_energy:.1f}J)>"
        )

