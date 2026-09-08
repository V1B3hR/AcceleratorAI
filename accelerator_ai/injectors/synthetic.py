"""
SyntheticInjector: Asynchronously injects synthetic boundary perturbation packets.
Uses convex manifold interpolation and variational jitter to explore latent space.
"""

from typing import Optional
import numpy as np
from accelerator_ai.core.flow_packet import FlowPacket
from accelerator_ai.injectors.base_injector import AsyncDataInjector


class SyntheticInjector(AsyncDataInjector):
    """
    Simulates a synthetic data injector nozzle.
    Blends features between random sample pairs (manifold mixup) to forge novel,
    continuous boundary examples on-the-fly.
    """

    def __init__(
        self,
        name: str = "SyntheticInjector",
        phase_offset: float = 0.5,
        pulse_frequency: float = 0.25,
        jitter: float = 0.1,
        batch_size: int = 16,
        interpolation_alpha: float = 0.4,
    ):
        super().__init__(
            name=name,
            phase_offset=phase_offset,
            pulse_frequency=pulse_frequency,
            jitter=jitter,
            batch_size=batch_size,
        )
        self.interpolation_alpha = interpolation_alpha

    def generate_fuel(self, context_packet: Optional[FlowPacket] = None) -> FlowPacket:
        """
        Synthesizes novel samples by interpolating existing context or generating synthetic features.
        """
        if context_packet is not None and context_packet.batch_size > 1:
            x_src = context_packet.x
            y_src = context_packet.y
            n_samples = min(self.batch_size, x_src.shape[0])

            # Select two random permutation index vectors
            idx1 = np.random.choice(x_src.shape[0], size=n_samples, replace=True)
            idx2 = np.random.choice(x_src.shape[0], size=n_samples, replace=True)

            # Draw beta-like convex weights
            lam = np.random.beta(self.interpolation_alpha, self.interpolation_alpha, size=(n_samples, 1))
            lam_y = lam if len(y_src.shape) > 1 else lam.squeeze(1)

            synth_x = lam * x_src[idx1] + (1.0 - lam) * x_src[idx2]
            synth_y = lam_y * y_src[idx1] + (1.0 - lam_y) * y_src[idx2]

            pressure = float(context_packet.pressure * 1.25)
            temperature = float(context_packet.temperature * 1.15)
        else:
            # Fallback random exploration manifold
            synth_x = np.random.randn(self.batch_size, 10)
            synth_y = np.random.randint(0, 2, size=(self.batch_size, 1)).astype(np.float32)
            pressure = 1.2
            temperature = 1.3

        return FlowPacket(
            x=synth_x,
            y=synth_y,
            pressure=pressure,
            viscosity=0.8,
            temperature=temperature,
            phase=self.internal_clock,
            source="synthetic_injector",
            metadata={"injector": self.name, "pulse_id": self.total_injections},
        )
