"""
Unit tests for TurboLearningEngine checkpointing (state_dict / load_state_dict)
and fault tolerance graceful bypass.
"""

import unittest
import numpy as np

from accelerator_ai.engine import TurboLearningEngine
from accelerator_ai.models.neural_core import PureNumPyMLP
from accelerator_ai.ecu.telemetry import TelemetryHub, WandBCallback, FileLogCallback
from accelerator_ai.core.metrics import EngineTelemetry


class TestCheckpointingAndReliability(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.x = np.random.randn(64, 8).astype(np.float32)
        self.y = np.random.randint(0, 2, size=(64,)).astype(np.int32)
        self.model = PureNumPyMLP(layer_sizes=[8, 16, 2])

    def test_engine_state_dict_and_restore(self):
        engine_original = TurboLearningEngine(model=self.model, base_learning_rate=0.02)

        # Run 10 steps to evolve physical dynamics (RPM, resonance, filter clog)
        for i in range(10):
            batch_x = self.x[i * 4 : (i + 1) * 4]
            batch_y = self.y[i * 4 : (i + 1) * 4]
            engine_original.step(batch_x, batch_y)

        saved_state = engine_original.state_dict()
        original_rpm = engine_original.shaft.rpm
        original_step = engine_original.current_step
        original_clog = engine_original.filter.clog_level
        original_lr = engine_original.braided_ecu.current_learning_rate

        self.assertEqual(saved_state["current_step"], 10)
        self.assertGreater(original_rpm, 800.0)

        # Create fresh engine instance
        fresh_model = PureNumPyMLP(layer_sizes=[8, 16, 2])
        engine_restored = TurboLearningEngine(model=fresh_model, base_learning_rate=0.02)

        # Confirm fresh state is at idle
        self.assertEqual(engine_restored.current_step, 0)
        self.assertAlmostEqual(engine_restored.shaft.rpm, 800.0, places=1)

        # Restore checkpoint
        engine_restored.load_state_dict(saved_state)

        # Verify exact restoration
        self.assertEqual(engine_restored.current_step, original_step)
        self.assertAlmostEqual(engine_restored.shaft.rpm, original_rpm, places=2)
        self.assertAlmostEqual(engine_restored.filter.clog_level, original_clog, places=4)
        self.assertAlmostEqual(
            engine_restored.braided_ecu.current_learning_rate, original_lr, places=5
        )

    def test_fault_tolerance_graceful_bypass(self):
        engine = TurboLearningEngine(model=self.model, fault_tolerance_mode=True)
        # Mock an error in pipeline.flow_step
        def crashing_flow_step(*args, **kwargs):
            raise RuntimeError("Simulated transient CUDA/hardware glitch")

        engine.pipeline.flow_step = crashing_flow_step

        # Step should not crash when fault_tolerance_mode is True; it gracefully bypasses
        res = engine.step(self.x[:16], self.y[:16])
        self.assertIsNotNone(res)
        self.assertGreater(res.loss, 0.0)
        self.assertEqual(res.predictions.shape, (16, 2))

    def test_telemetry_listener_resilience(self):
        hub = TelemetryHub()
        # Add a faulty listener that raises an exception
        def bad_listener(t: EngineTelemetry):
            raise ValueError("Buggy client dashboard callback")

        calls = []
        def good_listener(t: EngineTelemetry):
            calls.append(t.step)

        hub.add_listener(bad_listener)
        hub.add_listener(good_listener)

        t = EngineTelemetry(step=1, epoch=0, rpm=1000.0)
        # Hub should not crash, and good_listener should still receive the event
        hub.emit(t)
        self.assertEqual(calls, [1])


if __name__ == "__main__":
    unittest.main()
