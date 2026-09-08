"""
Unit tests for TurboLearningEngine and ECU Boost Controller.
"""

import unittest
import numpy as np

from accelerator_ai.models.neural_core import PureNumPyMLP
from accelerator_ai.engine import TurboLearningEngine
from accelerator_ai.ecu.controller import BoostController


class TestEngine(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.model = PureNumPyMLP(layer_sizes=[4, 8, 2])
        self.engine = TurboLearningEngine(
            model=self.model,
            target_boost_psi=10.0,
            base_learning_rate=0.01,
            enable_default_injectors=True,
        )
        self.x = np.random.randn(32, 4).astype(np.float32)
        self.y = np.random.randint(0, 2, size=(32,)).astype(np.int32)

    def test_engine_single_step(self):
        res = self.engine.step(self.x, self.y)
        self.assertGreater(res.loss, 0.0)
        self.assertEqual(self.engine.current_step, 1)
        self.assertGreater(self.engine.virtual_rpm, 0.0)

    def test_telemetry_emission(self):
        received = []
        self.engine.telemetry_hub.add_listener(lambda t: received.append(t))

        self.engine.step(self.x, self.y)
        self.assertEqual(len(received), 1)
        t = received[0]
        self.assertEqual(t.step, 1)
        self.assertGreater(t.rpm, 0.0)
        self.assertGreater(t.pyrometer_temp_c, 0.0)

    def test_ecu_thermal_cut(self):
        ecu = BoostController(target_boost_ratio=2.0, base_learning_rate=0.02)
        # Simulate extreme pyrometer heat (overheating / detonation)
        res = ecu.update(
            current_loss=10.0,
            previous_loss=1.0,
            pyrometer_temp=850.0,
            wastegate_open_pct=75.0,
        )
        self.assertTrue(res["thermal_throttle"])
        self.assertLess(res["boost_ratio"], 2.0)
        self.assertLess(res["learning_rate"], 0.02)

    def test_pytorch_wrapper_graceful_import(self):
        """If torch is missing, PyTorchTurbineWrapper must raise clear informative ImportError."""
        from accelerator_ai.models.torch_adapter import PyTorchTurbineWrapper
        try:
            import torch
            # If torch is installed, verify wrapper instantiation
            m = torch.nn.Linear(4, 2)
            opt = torch.optim.SGD(m.parameters(), lr=0.01)
            loss_fn = torch.nn.CrossEntropyLoss()
            wrapper = PyTorchTurbineWrapper(m, opt, loss_fn)
            self.assertIsNotNone(wrapper)
        except ImportError:
            with self.assertRaises(ImportError):
                PyTorchTurbineWrapper(None, None, None)


if __name__ == "__main__":
    unittest.main()
