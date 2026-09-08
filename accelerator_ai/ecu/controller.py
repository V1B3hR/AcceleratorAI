"""
ECU Boost Controller: The electronic engine control unit of AcceleratorAI.
Implements closed-loop PID control over turbo boost, learning rates,
and injector fuel trims to optimize training velocity without engine knock.
"""

from typing import Dict, Any, Optional
import numpy as np


class BoostController:
    """
    Simulates a digital turbocharger ECU.
    Uses PID control on loss convergence and gradient dynamics to continuously
    tune compressor boost, adaptive learning rates, and injection timings.
    """

    def __init__(
        self,
        target_boost_ratio: float = 1.5,
        base_learning_rate: float = 0.01,
        kp: float = 0.4,       # Proportional gain
        ki: float = 0.05,      # Integral gain
        kd: float = 0.1,       # Derivative gain
        max_boost_ratio: float = 2.5,
        min_boost_ratio: float = 1.0,
    ):
        self.target_boost_ratio = target_boost_ratio
        self.current_boost_ratio = target_boost_ratio
        self.learning_rate = base_learning_rate
        self.base_learning_rate = base_learning_rate

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_boost_ratio = max_boost_ratio
        self.min_boost_ratio = min_boost_ratio

        self.prev_error: float = 0.0
        self.integral_error: float = 0.0
        self.thermal_throttle_active: bool = False

    def update(
        self,
        current_loss: float,
        previous_loss: float,
        pyrometer_temp: float,
        wastegate_open_pct: float,
    ) -> Dict[str, float]:
        """
        Runs one ECU control cycle.
        
        Adjusts:
        - boost_ratio
        - learning_rate
        - injection_trim
        """
        # Loss reduction rate (positive means loss is dropping)
        loss_reduction = previous_loss - current_loss
        error = loss_reduction  # Target is positive progress

        # PID computation
        self.integral_error = float(np.clip(self.integral_error + error, -2.0, 2.0))
        derivative_error = error - self.prev_error
        self.prev_error = error

        pid_output = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * derivative_error)

        # Thermal protection check (preventing overfitting / detonation)
        if pyrometer_temp > 750.0 or wastegate_open_pct > 50.0:
            # Emergency thermal cut: pull boost and reduce learning rate
            self.thermal_throttle_active = True
            self.current_boost_ratio = max(
                self.min_boost_ratio, self.current_boost_ratio * 0.92
            )
            self.learning_rate = max(1e-5, self.learning_rate * 0.9)
        else:
            self.thermal_throttle_active = False
            # Modulate boost based on PID output
            new_boost = self.current_boost_ratio + (pid_output * 0.1)
            self.current_boost_ratio = float(
                np.clip(new_boost, self.min_boost_ratio, self.max_boost_ratio)
            )

            # Modulate learning rate proportional to boost ratio
            boost_factor = np.sqrt(self.current_boost_ratio)
            self.learning_rate = self.base_learning_rate * float(boost_factor)

        return {
            "boost_ratio": round(self.current_boost_ratio, 3),
            "learning_rate": round(self.learning_rate, 6),
            "thermal_throttle": self.thermal_throttle_active,
            "pid_output": round(pid_output, 4),
        }
