"""
Unit tests for Asynchronous Multi-Point Injectors.
"""

import unittest
import numpy as np

from accelerator_ai.core.flow_packet import FlowPacket
from accelerator_ai.injectors.synthetic import SyntheticInjector
from accelerator_ai.injectors.realworld import RealWorldReservoirInjector
from accelerator_ai.injectors.shock import EntropyShockInjector


class TestInjectors(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.x = np.random.randn(20, 5).astype(np.float32)
        self.y = np.random.randint(0, 2, size=(20, 1)).astype(np.float32)
        self.context_packet = FlowPacket(x=self.x, y=self.y)

    def test_synthetic_injector_generation(self):
        injector = SyntheticInjector(batch_size=8)
        fuel = injector.generate_fuel(self.context_packet)

        self.assertEqual(fuel.batch_size, 8)
        self.assertEqual(fuel.x.shape, (8, 5))
        self.assertEqual(fuel.source, "synthetic_injector")
        self.assertGreater(fuel.pressure, self.context_packet.pressure)

    def test_realworld_reservoir_injector(self):
        injector = RealWorldReservoirInjector(batch_size=6)
        injector.set_reservoir(self.x, self.y)
        fuel = injector.generate_fuel()

        self.assertEqual(fuel.batch_size, 6)
        self.assertEqual(fuel.source, "realworld_injector")

    def test_entropy_shock_injector_plateau_trigger(self):
        shock_inj = EntropyShockInjector(
            threshold=0.99,  # high wave threshold so wave alone doesn't fire
            plateau_window=5,
            plateau_std_threshold=0.01,
        )

        # Feed flat losses to trigger plateau
        for _ in range(5):
            shock_inj.record_loss(0.500)

        self.assertTrue(shock_inj.is_plateaued())

        # When plateaued, step_clock must fire!
        fired = shock_inj.step_clock(step=10)
        self.assertTrue(fired)

        shock_fuel = shock_inj.generate_fuel(self.context_packet)
        self.assertEqual(shock_fuel.source, "shock_injector")
        self.assertGreater(shock_fuel.temperature, 2.0)

    def test_manual_shock_trigger(self):
        shock_inj = EntropyShockInjector()
        shock_inj.trigger_manual_shock()
        fired = shock_inj.step_clock(step=1)
        self.assertTrue(fired)


if __name__ == "__main__":
    unittest.main()
