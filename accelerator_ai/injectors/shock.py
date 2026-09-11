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
        base_noise_scale: float = 0.35,
        min_scale_ratio: float = 0.08,
        loss_threshold: float = 0.15,
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
        self.base_noise_scale = base_noise_scale
        self.min_scale_ratio = min_scale_ratio
        self.loss_threshold = loss_threshold
        self.current_loss: float = 1.0
        self.loss_history = deque(maxlen=plateau_window)
        self.force_shock: bool = False
        self.last_shock_intensity: float = 0.0

    def record_loss(self, loss: float) -> None:
        """Tracks loss history to detect plateau conditions and calibrate shock amplitude."""
        self.current_loss = float(loss)
        self.loss_history.append(float(loss))

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
        Fires if:
        1. Manual shock is explicitly triggered.
        2. Engine stall / plateau is detected while loss is still significant.
        3. Stochastic wave breaches threshold AND loss is not yet in fine-tuning basin.
        """
        if self.force_shock:
            self.force_shock = False
            self.last_fired_step = step
            self.total_injections += 1
            return True

        # Stall condition: plateauing at high/mid loss warrants an emergency entropy shock.
        # If loss is already <= 0.05, suppress shocks to allow flat-minima settlement.
        if self.is_plateaued() and self.current_loss > 0.05:
            self.last_fired_step = step
            self.total_injections += 1
            return True

        # Periodic background clock wave: only fires if model is still converging (loss > 0.08)
        # Prevents violent noise injections from disrupting late-stage convergence.
        if self.current_loss > 0.08:
            return super().step_clock(step)

        return False

    def generate_fuel(self, context_packet: Optional[FlowPacket] = None) -> FlowPacket:
        """
        Generates annealed entropy perturbation data packets ("kopniak z boku").
        Noise amplitude dynamically decays as loss decreases, protecting narrow optimal basins.
        """
        # Dynamic noise annealing: scale down noise as loss approaches convergence threshold
        loss_factor = max(self.min_scale_ratio, min(1.0, self.current_loss / self.loss_threshold))
        effective_noise = self.base_noise_scale * loss_factor

        if context_packet is not None and context_packet.batch_size > 0:
            x_src = context_packet.x
            y_src = context_packet.y
            n_samples = min(self.batch_size, x_src.shape[0])
            indices = np.random.choice(x_src.shape[0], size=n_samples, replace=True)

            # Annealed orthogonal Laplace perturbation
            perturbation = np.random.laplace(0.0, effective_noise, size=x_src[indices].shape)
            shocked_x = x_src[indices] + perturbation

            # Target uncertainty shock: only apply label smoothing if loss is still high (> 0.20)
            shocked_y = y_src[indices].copy()
            if self.current_loss > 0.20 and np.random.rand() > 0.5:
                noise_y = np.random.normal(0.0, 0.2, size=shocked_y.shape)
                shocked_y = np.clip(shocked_y + noise_y, 0.0, 1.0)

            self.last_shock_intensity = float(np.mean(np.abs(perturbation)))
            pressure = float(context_packet.pressure * 2.0)  # High pressure kick
            temperature = float(context_packet.temperature * 2.2)  # Fiery combustion
        else:
            dim = context_packet.x.shape[1] if context_packet is not None and len(context_packet.x.shape) > 1 else 10
            shocked_x = np.random.randn(self.batch_size, dim) * (1.5 * loss_factor)
            shocked_y = np.random.randint(0, 2, size=(self.batch_size,)).astype(np.int32)
            pressure = 2.0
            temperature = 2.5
            self.last_shock_intensity = 1.0 * loss_factor

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
                "annealed_noise_scale": round(effective_noise, 4),
                "plateau_break": self.is_plateaued(),
            },
        )
