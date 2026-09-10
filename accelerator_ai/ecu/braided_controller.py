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
        ema_alpha: float = 0.1,           # Smoothing factor for resonance index
        lr_envelope_max: float = 1.3,     # Max LR = base * envelope (±30%)
    ):
        self.base_learning_rate = base_learning_rate
        self.coupling_strength = coupling_strength
        self.history_len = history_len
        self.ema_alpha = ema_alpha
        self.lr_envelope_max = lr_envelope_max
        self.lr_envelope_min = 1.0 / lr_envelope_max  # Symmetric lower bound

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
        self.resonance_index: float = 0.0     # Raw H in [-1.0, 1.0]
        self.smoothed_resonance: float = 0.0  # EMA-smoothed H (drives LR)
        self.winding_number: float = 0.0      # Total helical turns
        self.phase_tension: float = 0.0       # Metric of desynchronization

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

        # 3. Compute 4-Strand Pairwise Interference — vectorized
        phases = np.array([self.phase_grad, self.phase_press, self.phase_inj, self.phase_therm])
        weights = np.array([self.val_grad, self.val_press, self.val_inj, self.val_therm])

        # Pairwise phase difference matrix (upper triangle)
        phase_diff = np.subtract.outer(phases, phases)
        cos_diff = np.cos(phase_diff)
        weight_sum = np.add.outer(weights, weights)
        weighted_interf = cos_diff * (0.5 + 0.5 * weight_sum)

        # Extract upper triangle (6 unique pairs)
        triu_idx = np.triu_indices(4, k=1)
        pairwise_values = weighted_interf[triu_idx]

        # Raw Helical Resonance Index H in [-1.0, 1.0]
        self.resonance_index = float(np.mean(pairwise_values))
        self.resonance_history.append(self.resonance_index)

        # EMA-smoothed resonance (drives LR — prevents chaotic oscillations)
        self.smoothed_resonance = (
            self.ema_alpha * self.resonance_index
            + (1.0 - self.ema_alpha) * self.smoothed_resonance
        )

        # Phase Tension = variance among strand phases
        self.phase_tension = float(1.0 - max(0.0, self.smoothed_resonance))

        # Helical Winding Number
        self.winding_number += float(np.mean([omega_grad, omega_press, omega_inj, omega_therm])) / (2.0 * np.pi)

        # 4. Modulate Learning Rate based on SMOOTHED Resonance
        # Constructive resonance (H > 0): learning accelerates cleanly
        # High tension (H < 0): learning rate slows for careful exploration
        resonance_mod = 1.0 + (self.coupling_strength * self.smoothed_resonance)
        boost_mod = np.sqrt(max(1.0, boost_ratio))

        # Thermal protection damping
        thermal_damping = 1.0 - (0.5 * max(0.0, self.val_therm - 0.75))

        # Fine-tuning convergence settle: when loss is low (<0.1), smoothly damp LR
        # so the optimizer settles into the narrow global minimum rather than bouncing out
        loss_settle = 1.0
        if loss is not None and loss < 0.1:
            loss_settle = float(np.clip(0.35 + 0.65 * (loss / 0.1), 0.35, 1.0))

        raw_lr = float(
            self.base_learning_rate * resonance_mod * boost_mod * thermal_damping * loss_settle
        )

        # LR Envelope: clamp to [base*envelope_min, base*envelope_max]
        # When settling, allow lower floor down to 0.25 * base
        floor_multiplier = min(self.lr_envelope_min, loss_settle)
        lr_min = self.base_learning_rate * floor_multiplier
        lr_max = self.base_learning_rate * self.lr_envelope_max
        self.current_learning_rate = float(np.clip(raw_lr, max(1e-5, lr_min), min(0.1, lr_max)))

        # 5. Check for Phase Symmetry Breaking (Prolonged Tension)
        # If the strands remain in destructive dissonance for > 15 steps,
        # trigger a phase shock to break symmetry
        should_phase_shock = False
        if len(self.resonance_history) >= 15:
            # Use deque directly instead of list() conversion
            recent = list(self.resonance_history)[-15:]  # deque slice
            recent_avg = float(np.mean(recent))
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

    def state_dict(self) -> Dict[str, Any]:
        """Serializes braided controller state for checkpointing."""
        return {
            "current_learning_rate": float(self.current_learning_rate),
            "smoothed_resonance": float(self.smoothed_resonance),
            "resonance_index": float(self.resonance_index),
            "winding_number": float(self.winding_number),
            "phase_tension": float(self.phase_tension),
            "phase_grad": float(self.phase_grad),
            "phase_press": float(self.phase_press),
            "phase_inj": float(self.phase_inj),
            "phase_therm": float(self.phase_therm),
            "total_phase_shocks": int(self.total_phase_shocks),
            "resonance_history": list(self.resonance_history),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restores braided controller state from checkpoint."""
        self.current_learning_rate = float(state_dict.get("current_learning_rate", self.base_learning_rate))
        self.smoothed_resonance = float(state_dict.get("smoothed_resonance", 0.0))
        self.resonance_index = float(state_dict.get("resonance_index", 0.0))
        self.winding_number = float(state_dict.get("winding_number", 0.0))
        self.phase_tension = float(state_dict.get("phase_tension", 0.0))
        self.phase_grad = float(state_dict.get("phase_grad", 0.0))
        self.phase_press = float(state_dict.get("phase_press", 0.52))
        self.phase_inj = float(state_dict.get("phase_inj", 1.57))
        self.phase_therm = float(state_dict.get("phase_therm", 3.14))
        self.total_phase_shocks = int(state_dict.get("total_phase_shocks", 0))
        if "resonance_history" in state_dict:
            self.resonance_history = deque(state_dict["resonance_history"], maxlen=self.history_len)


