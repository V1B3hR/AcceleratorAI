"""
BraidedDNAController: Multi-strand helical control architecture for AcceleratorAI.

Replaces rigid, top-down scalar PID loops with an interwoven multi-strand system
analogous to the double/multi-helix of DNA.

Tracks 4 coupled dynamic strands:
  1. Gradient Strand (S_grad): Energy, momentum, and weight update torque.
  2. Pressure Strand (S_press): Information density and manifold compression.
  3. Injection Strand (S_inj): Asynchronous phase-lagged fuel pulses.
  4. Thermal Strand (S_therm): Entropy, loss variance, and thermal dissipation.

Cross-strand phase interference (H in [-1, 1]) is a deliberate feature:
Constructive resonance accelerates convergence; destructive tension induces
creative bifurcation, shattering loss plateaus.
"""

from collections import deque
from typing import Dict, Any, List, Tuple
import numpy as np


class BraidedDNAController:
    """
    Orchestrates the 4 interwoven physical strands of AcceleratorAI.
    """

    def __init__(
        self,
        base_learning_rate: float = 0.015,
        coupling_strength: float = 0.35,
        history_len: int = 100,
    ):
        self.base_learning_rate = base_learning_rate
        self.coupling_strength = coupling_strength
        self.history_len = history_len

        # Strand phase clocks (radians)
        self.phase_grad: float = 0.0
        self.phase_press: float = 0.52   # ~30 deg offset
        self.phase_inj: float = 1.57     # ~90 deg offset (quadrature)
        self.phase_therm: float = 3.14   # ~180 deg offset (counter-phase)

        # Strand values (normalized 0.0 - 1.0)
        self.val_grad: float = 0.0
        self.val_press: float = 0.0
        self.val_inj: float = 0.0
        self.val_therm: float = 0.0

        # Helical Resonance Metrics
        self.resonance_index: float = 0.0   # H in [-1.0, 1.0]
        self.winding_number: float = 0.0    # Total helical turns
        self.phase_tension: float = 0.0     # Metric of desynchronization / creative friction

        self.resonance_history = deque(maxlen=history_len)
        self.current_learning_rate: float = base_learning_rate
        self.total_phase_shocks: int = 0

    def update(
        self,
        step: int,
        learning_torque: float,
        boost_ratio: float,
        injected_entropy: float,
        pyrometer_temp: float,
        loss: float,
    ) -> Dict[str, Any]:
        """
        Executes one helical braiding cycle.
        
        Args:
            step: Current engine iteration step.
            learning_torque: Gradient torque tau (Nm).
            boost_ratio: Compressor manifold pressure ratio Psi.
            injected_entropy: Total entropy injected in this cycle.
            pyrometer_temp: Exhaust temperature T (deg C).
            loss: Current combustion loss.
            
        Returns:
            Dictionary containing updated learning rate, resonance index,
            and strand coordinates for visualization.
        """
        # 1. Normalize strand magnitudes
        self.val_grad = float(np.tanh(learning_torque * 0.8))
        self.val_press = float(np.clip((boost_ratio - 1.0) / 1.5, 0.0, 1.0))
        self.val_inj = float(np.tanh(injected_entropy * 0.1))
        self.val_therm = float(np.clip((pyrometer_temp - 200.0) / 700.0, 0.0, 1.0))

        # 2. Advance strand phase angles with natural non-linear frequencies
        # Frequencies are slightly incommensurate (golden ratio / irrational roots)
        # to prevent uncreative periodic locking
        omega_grad = 0.22 + (0.15 * self.val_grad)
        omega_press = 0.18 + (0.12 * self.val_press)
        omega_inj = 0.31 + (0.20 * self.val_inj)
        omega_therm = 0.14 + (0.08 * self.val_therm)

        self.phase_grad = (self.phase_grad + omega_grad) % (2.0 * np.pi)
        self.phase_press = (self.phase_press + omega_press) % (2.0 * np.pi)
        self.phase_inj = (self.phase_inj + omega_inj) % (2.0 * np.pi)
        self.phase_therm = (self.phase_therm + omega_therm) % (2.0 * np.pi)

        # 3. Compute 4-Strand Pairwise Interference Matrix M_ij = cos(theta_i - theta_j)
        phases = [self.phase_grad, self.phase_press, self.phase_inj, self.phase_therm]
        weights = [self.val_grad, self.val_press, self.val_inj, self.val_therm]

        pairwise_cos = []
        for i in range(len(phases)):
            for j in range(i + 1, len(phases)):
                cos_diff = np.cos(phases[i] - phases[j])
                # Amplitude-weighted interference
                weighted_interf = cos_diff * (0.5 + 0.5 * (weights[i] + weights[j]))
                pairwise_cos.append(weighted_interf)

        # Helical Resonance Index H in [-1.0, 1.0]
        self.resonance_index = float(np.mean(pairwise_cos))
        self.resonance_history.append(self.resonance_index)

        # Phase Tension = variance among strand phases (high tension = high creative divergence)
        self.phase_tension = float(1.0 - max(0.0, self.resonance_index))

        # Helical Winding Number
        self.winding_number += float(np.mean([omega_grad, omega_press, omega_inj, omega_therm])) / (2.0 * np.pi)

        # 4. Modulate Learning Rate based on Braided Resonance & Pressure
        # In constructive resonance (H > 0): learning accelerates cleanly
        # In high tension (H < 0): learning rate slows down for careful valley exploration
        # Boost ratio provides a geometric acceleration factor
        resonance_mod = 1.0 + (self.coupling_strength * self.resonance_index)
        boost_mod = np.sqrt(max(1.0, boost_ratio))

        # Thermal protection damping if pyrometer overheats
        thermal_damping = 1.0 - (0.5 * max(0.0, self.val_therm - 0.75))

        self.current_learning_rate = float(
            self.base_learning_rate * resonance_mod * boost_mod * thermal_damping
        )
        self.current_learning_rate = float(np.clip(self.current_learning_rate, 1e-5, 0.1))

        # 5. Check for Phase Symmetry Breaking (Prolonged Tension)
        # If the strands remain in destructive dissonance for > 15 steps,
        # trigger a phase shock to break symmetry
        should_phase_shock = False
        if len(self.resonance_history) >= 15:
            recent_avg = np.mean(list(self.resonance_history)[-15:])
            if recent_avg < -0.25:
                should_phase_shock = True
                self.total_phase_shocks += 1
                # Re-align phases with an injection kick
                self.phase_inj = (self.phase_grad + 0.5) % (2.0 * np.pi)

        # 6. Variable Geometry Aperture Control Signals
        # Dynamic port apertures driven by resonance state:
        #   Constructive resonance (H > 0) → FOCUS: narrow hyper, narrow slow-mo
        #   Destructive tension   (H < 0) → EXPLORE: widen hyper, widen slow-mo
        #   Neutral               (H ≈ 0) → balanced defaults
        h = self.resonance_index
        aperture_signals = {
            "hyper": float(np.clip(0.15 + 0.25 * max(0.0, -h), 0.05, 0.40)),
            "cruise": 0.50,
            "slowmo": float(np.clip(0.85 - 0.20 * max(0.0, h), 0.50, 0.95)),
        }

        return {
            "learning_rate": round(self.current_learning_rate, 6),
            "resonance_index": round(self.resonance_index, 4),
            "phase_tension": round(self.phase_tension, 4),
            "winding_number": round(self.winding_number, 2),
            "should_phase_shock": should_phase_shock,
            "aperture_signals": aperture_signals,
            "strands": {
                "grad": {"val": round(self.val_grad, 3), "phase": round(self.phase_grad, 2)},
                "press": {"val": round(self.val_press, 3), "phase": round(self.phase_press, 2)},
                "inj": {"val": round(self.val_inj, 3), "phase": round(self.phase_inj, 2)},
                "therm": {"val": round(self.val_therm, 3), "phase": round(self.phase_therm, 2)},
            },
        }

