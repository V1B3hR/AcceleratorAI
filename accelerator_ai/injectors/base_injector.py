"""
AsyncDataInjector: Base class for asynchronous multi-point data injectors.

Unlike traditional synchronized mini-batching, injectors operate with independent
phase angles (delta_phi), variable pulse velocities, and non-linear injection timings.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np
from accelerator_ai.core.flow_packet import FlowPacket


class AsyncDataInjector(ABC):
    """
    Simulates a fuel injector nozzle with independent phase clock and variable velocity.
    Operates asynchronously to introduce deliberate informational perturbation,
    forcing the model to break rigid habits, adapt rapidly, and explore creative optima.
    """

    def __init__(
        self,
        name: str,
        phase_offset: float = 0.0,       # Phase shift delta_phi (in radians)
        pulse_frequency: float = 0.3,     # Base oscillation frequency
        jitter: float = 0.15,             # Stochastic timing jitter
        threshold: float = 0.6,           # Firing threshold on the oscillation wave
        batch_size: int = 16,
    ):
        self.name = name
        self.phase_offset = phase_offset
        self.pulse_frequency = pulse_frequency
        self.jitter = jitter
        self.threshold = threshold
        self.batch_size = batch_size

        self.internal_clock: float = 0.0
        self.total_injections: int = 0
        self.total_injected_samples: int = 0
        self.last_fired_step: int = -1
        self.active: bool = True
        self.last_telemetry: Dict[str, Any] = {}

    def step_clock(self, step: int) -> bool:
        """
        Advances the injector phase clock and evaluates if the nozzle fires on this step.
        Firing condition: sin(frequency * step + phase + random_jitter) >= threshold.
        """
        if not self.active:
            return False

        self.internal_clock = (self.pulse_frequency * step) + self.phase_offset
        stochastic_shift = np.random.normal(0.0, self.jitter)
        wave_value = np.sin(self.internal_clock + stochastic_shift)

        should_fire = wave_value >= self.threshold
        if should_fire:
            self.last_fired_step = step
            self.total_injections += 1

        self.last_telemetry = {
            "name": self.name,
            "fired": should_fire,
            "wave_value": round(float(wave_value), 3),
            "total_injections": self.total_injections,
            "phase": round(float(self.internal_clock % (2 * np.pi)), 2),
        }
        return should_fire

    @abstractmethod
    def generate_fuel(self, context_packet: Optional[FlowPacket] = None) -> FlowPacket:
        """
        Synthesizes or retrieves the supplemental data packet to be injected.
        Must be implemented by concrete injectors.
        """
        pass

    def pulse(self, step: int, context_packet: Optional[FlowPacket] = None) -> Optional[FlowPacket]:
        """
        Checks trigger condition and fires an asynchronous fuel packet if ready.
        """
        if self.step_clock(step):
            packet = self.generate_fuel(context_packet)
            self.total_injected_samples += packet.batch_size
            return packet
        return None

    def adapt_dynamics(self, reward: float) -> None:
        """
        ECU feedback adaptation: if recent injections reduced loss or improved torque,
        adjust pulse frequency or phase to reinforce; if disruptive, back off slightly.
        """
        if reward > 0:
            # Positive adaptation: slightly tighten frequency towards current sweet spot
            self.pulse_frequency = float(np.clip(self.pulse_frequency * 1.05, 0.05, 1.5))
        else:
            # Negative adaptation: shift phase slightly to disrupt resonance
            self.phase_offset = float((self.phase_offset + 0.2) % (2 * np.pi))

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}(name='{self.name}', "
            f"phase={self.phase_offset:.2f}, freq={self.pulse_frequency:.2f})>"
        )
