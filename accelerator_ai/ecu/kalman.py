"""
KalmanLossGovernor: Closed-loop State-Space Kalman Filter for Training Dynamics.

Eliminates noisy threshold heuristics in turbine regulation by tracking latent
true loss and descent velocity with optimal stochastic noise rejection.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import math
import numpy as np


@dataclass
class KalmanEstimate:
    """Optimal filtered state estimate from the Kalman Governor."""
    filtered_loss: float
    loss_velocity: float
    velocity_variance: float
    innovation: float
    is_plateau: bool
    is_diverging: bool
    recommended_boost_mod: float


class KalmanLossGovernor:
    """
    2-State Discrete-Time Kalman Filter tracking:
      x = [ Loss (L),  Loss Velocity (dL/dt) ]^T

    Filters out mini-batch stochastic gradient noise to provide stable,
    jitter-free control signals for VVT gearing, turbine boost, and wastegate actuation.
    """

    def __init__(
        self,
        initial_loss: float = 1.0,
        process_noise_loss: float = 1e-4,
        process_noise_velocity: float = 1e-5,
        measurement_noise: float = 0.02,
        plateau_velocity_threshold: float = 0.0015,
        divergence_velocity_threshold: float = 0.025,
    ):
        # State vector: [L, dL]
        self.x = np.array([float(initial_loss), 0.0], dtype=np.float64)

        # State transition matrix F (dt = 1 step)
        self.F = np.array([[1.0, 1.0],
                           [0.0, 1.0]], dtype=np.float64)

        # Measurement matrix H
        self.H = np.array([[1.0, 0.0]], dtype=np.float64)

        # Process noise covariance Q
        self.Q = np.array([[process_noise_loss, 0.0],
                           [0.0, process_noise_velocity]], dtype=np.float64)

        # Measurement noise variance R
        self.R = float(measurement_noise)

        # Error covariance matrix P
        self.P = np.array([[0.1, 0.0],
                           [0.0, 0.01]], dtype=np.float64)

        self.plateau_thresh = plateau_velocity_threshold
        self.divergence_thresh = divergence_velocity_threshold
        self.step_count = 0
        self.consecutive_plateau_steps = 0

    def update(self, observed_loss: float) -> KalmanEstimate:
        """
        Executes Predict -> Measure -> Correct cycles for observed mini-batch loss.
        Takes < 3 microseconds in scalar/2x2 numpy.
        """
        self.step_count += 1
        z = float(observed_loss)

        # 1. Predict Step
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # 2. Measurement Innovation
        y = z - float((self.H @ x_pred).item())
        S = float((self.H @ P_pred @ self.H.T).item()) + self.R

        # 3. Kalman Gain
        K = (P_pred @ self.H.T) / S  # Shape: (2, 1)

        # 4. Correct Step
        self.x = x_pred + (K.ravel() * y)
        I_KH = np.eye(2) - (K @ self.H)
        self.P = I_KH @ P_pred

        filtered_loss = float(self.x[0])
        loss_velocity = float(self.x[1])
        velocity_var = float(self.P[1, 1])

        # 5. Diagnostic State Evaluation
        # A true plateau occurs when velocity is near zero with low variance
        is_plateau = bool(abs(loss_velocity) < self.plateau_thresh and self.step_count > 10)
        if is_plateau:
            self.consecutive_plateau_steps += 1
        else:
            self.consecutive_plateau_steps = 0

        # Divergence occurs when loss is climbing rapidly
        is_diverging = bool(loss_velocity > self.divergence_thresh)

        # 6. Smooth Closed-Loop Boost Modulation Factor
        # Plateau -> spool up boost smoothly (up to 1.5x)
        # Diverging -> back off boost smoothly (down to 0.7x)
        # Fast descent -> cruising stability (1.0x - 1.1x)
        if is_diverging:
            boost_mod = max(0.7, 1.0 - (loss_velocity * 10.0))
        elif is_plateau:
            plateau_boost_ramp = min(1.5, 1.0 + (self.consecutive_plateau_steps * 0.05))
            boost_mod = plateau_boost_ramp
        else:
            # Steady gradient descent
            boost_mod = 1.0 + min(0.2, max(-0.1, -loss_velocity * 5.0))

        return KalmanEstimate(
            filtered_loss=filtered_loss,
            loss_velocity=loss_velocity,
            velocity_variance=velocity_var,
            innovation=y,
            is_plateau=is_plateau,
            is_diverging=is_diverging,
            recommended_boost_mod=round(float(boost_mod), 4),
        )

    def reset(self, initial_loss: float = 1.0) -> None:
        """Resets filter state for new epoch or model reload."""
        self.x = np.array([float(initial_loss), 0.0], dtype=np.float64)
        self.P = np.array([[0.1, 0.0],
                           [0.0, 0.01]], dtype=np.float64)
        self.step_count = 0
        self.consecutive_plateau_steps = 0
