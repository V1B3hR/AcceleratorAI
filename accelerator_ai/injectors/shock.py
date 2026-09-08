"""
EntropyShockInjector: The "kopniak z boku" (Chaos Shock) injector.

Monitors training convergence for stagnating plateaus or rigid, monotonous behavior.
When triggered, injects sudden, high-entropy adversarial and orthogonal perturbations,
shaking the optimization trajectory out of local minima and fostering emergent adaptability.
"""

from collections import deque
from typing import Optional
import numpy as np
from accelerator_ai.core.flow_packet import FlowPacket
from accelerator_ai.injectors.base_injector import AsyncDataInjector


class EntropyShockInjector(AsyncDataInjector):
    """
    Simulates a high-pressure nitrous/chaos injector nozzle.
    Fires non-linear entropy kicks into the combustion chamber to break local minima
    and force rapid internal representation reorganization.
    """

    def __init__(
        self,
        name: str = "EntropyShockInjector",
        phase_offset: float = 3.14,      # Half-period shifted (180 deg out of phase)
        pulse_frequency: float = 0.15,
        jitter: float = 0.3,
        threshold: float = 0.75,         # Rare, dramatic firings
        batch_size: int = 12,
        plateau_window: int = 10,
        plateau_std_threshold: float = 0.005,
    ):
        super().__init__(
            name=name,
            phase_offset=phase_offset,
            pulse_frequency=pulse_frequency,
            jitter=jitter,
            threshold=threshold,
            batch_size=batch_size,
        )
        self.plateau_window = plateau_window
        self.plateau_std_threshold = plateau_std_threshold
        self.loss_history = deque(maxlen=plateau_window)
        self.force_shock: bool = False
        self.last_shock_intensity: float = 0.0

    def record_loss(self, loss: float) -> None:
        """Tracks loss history to detect plateau conditions."""
        self.loss_history.append(loss)

    def is_plateaued(self) -> bool:
        """Returns True if the loss has stagnated in a local minimum flatland."""
        if len(self.loss_history) < self.plateau_window:
            return False
        stdev = float(np.std(list(self.loss_history)))
        return stdev < self.plateau_std_threshold

    def trigger_manual_shock(self) -> None:
        """Manually triggers a high-entropy NOS kick from the cockpit or ECU."""
        self.force_shock = True

    def step_clock(self, step: int) -> bool:
        """
        Fires either if manual shock is tripped, plateau is detected,
        or when the stochastic asynchronous clock wave breaches threshold.
        """
        if self.force_shock:
            self.force_shock = False
            self.last_fired_step = step
            self.total_injections += 1
            return True

        # Check plateau trigger
        if self.is_plateaued():
            self.last_fired_step = step
            self.total_injections += 1
            return True

        return super().step_clock(step)

    def generate_fuel(self, context_packet: Optional[FlowPacket] = None) -> FlowPacket:
        """
        Generates high-entropy, orthogonal perturbation data packets ("kopniak z boku").
        """
        if context_packet is not None and context_packet.batch_size > 0:
            x_src = context_packet.x
            y_src = context_packet.y
            n_samples = min(self.batch_size, x_src.shape[0])
            indices = np.random.choice(x_src.shape[0], size=n_samples, replace=True)

            # High-intensity orthogonal noise + sign flip shock
            perturbation = np.random.laplace(0.0, 0.35, size=x_src[indices].shape)
            shocked_x = x_src[indices] + perturbation

            # Occasional target uncertainty shock (soft label smoothing / swap)
            shocked_y = y_src[indices].copy()
            if np.random.rand() > 0.5:
                noise_y = np.random.normal(0.0, 0.2, size=shocked_y.shape)
                shocked_y = np.clip(shocked_y + noise_y, 0.0, 1.0)

            self.last_shock_intensity = float(np.mean(np.abs(perturbation)))
            pressure = float(context_packet.pressure * 2.0)  # Extreme pressure kick
            temperature = float(context_packet.temperature * 2.2)  # Fiery combustion
        else:
            dim = context_packet.x.shape[1] if context_packet is not None and len(context_packet.x.shape) > 1 else 10
            shocked_x = np.random.randn(self.batch_size, dim) * 1.5
            shocked_y = np.random.randint(0, 2, size=(self.batch_size,)).astype(np.int32)
            pressure = 2.0
            temperature = 2.5
            self.last_shock_intensity = 1.0

        return FlowPacket(
            x=shocked_x,
            y=shocked_y,
            pressure=pressure,
            viscosity=0.5,        # Low viscosity (fast explosive expansion)
            temperature=temperature,
            phase=self.internal_clock,
            source="shock_injector",
            metadata={
                "injector": self.name,
                "pulse_id": self.total_injections,
                "shock_intensity": round(self.last_shock_intensity, 4),
                "plateau_break": self.is_plateaued(),
            },
        )
