"""
RealWorldReservoirInjector: Injects rare, high-viscosity real-world benchmark data
to prevent drift and anchor representations to real-world ground truth.
"""

from typing import Optional
import numpy as np
from accelerator_ai.core.flow_packet import FlowPacket
from accelerator_ai.injectors.base_injector import AsyncDataInjector


class RealWorldReservoirInjector(AsyncDataInjector):
    """
    Simulates a high-pressure secondary injector delivering dense, real-world
    empirical data into the combustion chamber with distinct phase timing.
    """

    def __init__(
        self,
        name: str = "RealWorldInjector",
        reservoir_x: Optional[np.ndarray] = None,
        reservoir_y: Optional[np.ndarray] = None,
        phase_offset: float = 1.8,       # Shifted out of phase with synthetic injector
        pulse_frequency: float = 0.18,    # Lower frequency, higher impact
        jitter: float = 0.2,
        batch_size: int = 16,
    ):
        super().__init__(
            name=name,
            phase_offset=phase_offset,
            pulse_frequency=pulse_frequency,
            jitter=jitter,
            batch_size=batch_size,
        )
        self.reservoir_x = reservoir_x
        self.reservoir_y = reservoir_y

    def set_reservoir(self, x: np.ndarray, y: np.ndarray) -> None:
        """Loads or updates the real-world sample reservoir."""
        self.reservoir_x = x
        self.reservoir_y = y

    def generate_fuel(self, context_packet: Optional[FlowPacket] = None) -> FlowPacket:
        """
        Samples an asynchronous batch from the real-world reservoir.
        """
        has_valid_reservoir = (
            self.reservoir_x is not None 
            and len(self.reservoir_x) > 0
            and (context_packet is None or self.reservoir_x.shape[1] == context_packet.x.shape[1])
        )

        if has_valid_reservoir:
            n = len(self.reservoir_x)
            indices = np.random.choice(n, size=min(self.batch_size, n), replace=False)
            x_batch = self.reservoir_x[indices]
            y_batch = self.reservoir_y[indices]
        elif context_packet is not None and context_packet.batch_size > 0:
            # Fallback: slice from context packet if reservoir empty or dimension mismatched
            n = context_packet.batch_size
            indices = np.random.choice(n, size=min(self.batch_size, n), replace=True)
            x_batch = context_packet.x[indices]
            y_batch = context_packet.y[indices]
        else:
            dim = context_packet.x.shape[1] if context_packet is not None and len(context_packet.x.shape) > 1 else 10
            x_batch = np.zeros((self.batch_size, dim))
            y_batch = np.zeros((self.batch_size,), dtype=np.int32)

        return FlowPacket(
            x=x_batch,
            y=y_batch,
            pressure=1.5,       # Dense real-world ground truth
            viscosity=1.4,      # Higher viscosity (complex real manifold)
            temperature=0.8,    # Low entropy / clean signal
            phase=self.internal_clock,
            source="realworld_injector",
            metadata={"injector": self.name, "pulse_id": self.total_injections},
        )
