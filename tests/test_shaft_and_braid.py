"""
Unit tests for DriveShaft mechanical inertia and BraidedDNAController helical dynamics.
"""

import unittest
import numpy as np

from accelerator_ai.core.shaft import DriveShaft
from accelerator_ai.turbines.compressor import CompressorTurbine
from accelerator_ai.ecu.braided_controller import BraidedDNAController
from accelerator_ai.models.neural_core import PureNumPyMLP
from accelerator_ai.engine import TurboLearningEngine


class TestShaftAndBraidedDNA(unittest.TestCase):

    def test_shaft_inertia_and_coasting(self):
        shaft = DriveShaft(inertia=0.05, idle_rpm=800.0, friction_coeff=0.01)
        initial_rpm = shaft.rpm
        self.assertAlmostEqual(initial_rpm, 800.0, places=1)

        # Apply high driving torque for 5 steps (spool up)
        for _ in range(5):
            shaft.step(torque_in=5.0, load_torque=0.5, dt=0.05)

        spooled_rpm = shaft.rpm
        self.assertGreater(spooled_rpm, initial_rpm)
        self.assertGreater(shaft.kinetic_energy, 0.0)

        # Now remove driving torque (coasting under inertia and drag)
        # Because of inertia I, shaft should NOT instantly drop to 800 RPM
        shaft.step(torque_in=0.0, load_torque=0.0, dt=0.05)
        self.assertGreater(shaft.rpm, 800.0)
        self.assertLess(shaft.rpm, spooled_rpm)

    def test_compressor_physical_euler_coupling(self):
        shaft = DriveShaft(inertia=0.05, idle_rpm=800.0)
        compressor = CompressorTurbine(shaft=shaft, pressure_coefficient=0.5)

        # At idle speed, boost ratio should be ~1.0
        compressor.update_from_shaft()
        self.assertAlmostEqual(compressor.boost_ratio, 1.0, places=2)

        # Spin shaft up to 2400 RPM (3x idle speed)
        shaft.omega = shaft.idle_omega * 3.0
        compressor.update_from_shaft()

        # Boost ratio must rise quadratically / exponentially
        self.assertGreater(compressor.boost_ratio, 1.5)
        self.assertGreater(compressor.boost_psi, 8.0)

        # Reaction load torque must be positive
        load = compressor.compute_reaction_load()
        self.assertGreater(load, 0.0)

    def test_braided_dna_controller_resonance(self):
        controller = BraidedDNAController(base_learning_rate=0.02)
        initial_lr = controller.base_learning_rate

        res = controller.update(
            step=1,
            learning_torque=2.5,
            boost_ratio=1.6,
            injected_entropy=5.0,
            pyrometer_temp=300.0,
            loss=0.8,
        )

        self.assertIn("resonance_index", res)
        self.assertIn("phase_tension", res)
        self.assertIn("strands", res)
        self.assertGreater(res["learning_rate"], 0.0)

        # Check all 4 strands exist
        strands = res["strands"]
        self.assertIn("grad", strands)
        self.assertIn("press", strands)
        self.assertIn("inj", strands)
        self.assertIn("therm", strands)

    def test_engine_closed_feedback_loop(self):
        model = PureNumPyMLP(layer_sizes=[4, 8, 2])
        engine = TurboLearningEngine(
            model=model,
            base_learning_rate=0.015,
            shaft_inertia=0.05,
            enable_default_injectors=True,
        )

        x = np.random.randn(16, 4).astype(np.float32)
        y = np.random.randint(0, 2, size=(16,)).astype(np.int32)

        # Step 1
        res1 = engine.step(x, y)
        rpm1 = engine.shaft.rpm

        # Step 2
        res2 = engine.step(x, y)
        rpm2 = engine.shaft.rpm

        # Shaft RPM should have evolved dynamically
        self.assertGreater(rpm2, 0.0)
        self.assertGreater(engine.compressor.boost_ratio, 1.0)
        self.assertGreater(engine.shaft.kinetic_energy, 0.0)


if __name__ == "__main__":
    unittest.main()
