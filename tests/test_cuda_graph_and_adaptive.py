"""
Unit tests for CUDA Graph capture, Adaptive Turbocharging, Fused Wastegate Soft-Clipping,
AsyncTelemetryHub, and FlowPacketPool.
"""

import unittest
import time
import numpy as np

from accelerator_ai.core.flow_packet import FlowPacket, FlowPacketPool
from accelerator_ai.ecu.telemetry import TelemetryHub, AsyncTelemetryHub
from accelerator_ai.core.metrics import EngineTelemetry
from accelerator_ai.config import EngineConfig
from accelerator_ai.models.neural_core import PureNumPyMLP
from accelerator_ai.engine import TurboLearningEngine


class TestFlowPacketPool(unittest.TestCase):

    def test_acquire_and_release(self):
        pool = FlowPacketPool(max_size=4)
        x = np.ones((8, 4), dtype=np.float32)
        y = np.zeros((8,), dtype=np.int32)

        p1 = pool.acquire(x=x, y=y, pressure=2.0, source="test")
        self.assertEqual(p1.batch_size, 8)
        self.assertEqual(p1.pressure, 2.0)
        self.assertEqual(p1.source, "test")

        pool.release(p1)
        self.assertIsNone(p1.x)
        self.assertIsNone(p1.y)
        self.assertEqual(len(pool.pool), 1)

        p2 = pool.acquire(x=x, y=y, pressure=1.5, source="recycled")
        self.assertIs(p2, p1)
        self.assertEqual(p2.source, "recycled")
        self.assertEqual(p2.pressure, 1.5)
        self.assertEqual(len(pool.pool), 0)

    def test_pool_capacity_limit(self):
        pool = FlowPacketPool(max_size=2)
        pkts = [FlowPacket(x=np.zeros(2), y=np.zeros(2)) for _ in range(5)]
        for p in pkts:
            pool.release(p)
        self.assertEqual(len(pool.pool), 2)


class TestAsyncTelemetry(unittest.TestCase):

    def test_async_telemetry_hub_dispatch(self):
        hub = AsyncTelemetryHub(history_len=50, max_queue_size=10, enabled=True)
        received = []
        hub.add_listener(lambda t: received.append(t))

        telem = EngineTelemetry(step=1, epoch=0, rpm=2500.0, loss=0.5)

        hub.emit(telem)
        hub.flush(timeout=2.0)
        time.sleep(0.05)  # brief grace period for listener invocation

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].step, 1)
        self.assertEqual(hub.latest.step, 1)
        hub.close()

    def test_telemetry_disabled_toggle(self):
        hub = TelemetryHub(enabled=False)
        received = []
        hub.add_listener(lambda t: received.append(t))

        telem = EngineTelemetry(step=1, epoch=0, rpm=2000.0, loss=0.5)
        hub.emit(telem)
        self.assertEqual(len(received), 0)
        self.assertIsNone(hub.latest)


class TestAdaptiveTurboGovernor(unittest.TestCase):

    def setUp(self):
        model = PureNumPyMLP(layer_sizes=[4, 8, 2])
        self.engine = TurboLearningEngine(
            model=model,
            adaptive_turbo=True,
            enable_default_injectors=False,
            fault_tolerance_mode=False,
        )

    def test_should_activate_turbo_conditions(self):
        # Step <= 50 -> always spool up
        self.assertTrue(self.engine._should_activate_turbo(current_step=1))
        self.assertTrue(self.engine._should_activate_turbo(current_step=50))

        # Step 51 not periodic -> False
        self.assertFalse(self.engine._should_activate_turbo(current_step=51))

        # Periodic calibration step % 10 == 0 -> True
        self.assertTrue(self.engine._should_activate_turbo(current_step=60))

        # Wastegate knock active -> True
        self.engine.wastegate.open_pct = 45.0
        self.assertTrue(self.engine._should_activate_turbo(current_step=53))
        self.engine.wastegate.open_pct = 0.0

        # Loss plateau detection
        self.engine.loss_history = [1.0] * 12
        self.engine.previous_loss = 1.0 - 0.0005  # delta = 0.0005 < 0.001
        self.assertTrue(self.engine._should_activate_turbo(current_step=53))


class TestPyTorchAdapterAndCUDAGraph(unittest.TestCase):

    def test_fused_harvest_and_amp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        m = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 4),
        ).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, capturable=True if "cuda" in device else False)
        loss_fn = torch.nn.CrossEntropyLoss()

        from accelerator_ai.models.torch_adapter import PyTorchTurbineWrapper
        wrapper = PyTorchTurbineWrapper(
            model=m,
            optimizer=opt,
            loss_fn=loss_fn,
            enable_amp=("cuda" in device),
            amp_dtype="bfloat16",
        )
        wrapper.enable_capturable_optimizer()

        x = torch.randn(16, 8, device=device)
        y = torch.randint(0, 4, (16,), device=device)

        preds, loss = wrapper.forward_and_loss(x, y)
        self.assertGreater(loss, 0.0)

        raw_norm, clipped_norm, vented = wrapper.harvest_and_regulate_fused(
            threshold=5.0, boost_ratio=1.5, enable_soft_clipping=True
        )
        self.assertGreater(raw_norm, 0.0)
        self.assertGreater(clipped_norm, 0.0)
        self.assertLessEqual(clipped_norm, raw_norm + 1e-5)

        wrapper.apply_updates(learning_rate=0.002)

    def test_cuda_graph_capture_on_gpu(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for CUDA Graph test")

        m = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 2),
        ).cuda()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, capturable=True)
        loss_fn = torch.nn.CrossEntropyLoss()

        from accelerator_ai.models.torch_adapter import PyTorchTurbineWrapper
        wrapper = PyTorchTurbineWrapper(m, opt, loss_fn=loss_fn)

        engine = TurboLearningEngine(
            model=wrapper,
            enable_default_injectors=False,
            enable_vvt=False,
            enable_cuda_graph=True,
        )

        x = torch.randn(32, 10, device="cuda")
        y = torch.randint(0, 2, (32,), device="cuda")

        # Capture CUDA Graph
        engine.capture_cuda_graph(x, y)
        self.assertIsNotNone(engine.cuda_graph)

        # Replay via step
        res = engine.step(x, y)
        self.assertGreater(res.loss, 0.0)
        self.assertEqual(engine.current_step, 1)


class TestPhase3AdvancedOptimizations(unittest.TestCase):

    def test_vvt_detect_optimal_gears(self):
        from accelerator_ai.turbines.vvt import VariableValveTiming
        gears = VariableValveTiming.detect_optimal_gears()
        self.assertIsInstance(gears, tuple)
        self.assertGreaterEqual(len(gears), 3)
        self.assertTrue(all(g > 0 for g in gears))
        self.assertEqual(gears, tuple(sorted(gears)))

        vvt = VariableValveTiming(auto_detect_hardware=True)
        self.assertEqual(vvt.gears, gears)

    def test_vvt_predictive_shifting(self):
        from accelerator_ai.turbines.vvt import VariableValveTiming
        vvt = VariableValveTiming(gears=(16, 32, 64))
        # Initial gear is Gear 2 (batch 32)
        self.assertEqual(vvt.current_gear, 2)

        # 1. Accelerating descent (loss_delta = 0.08) -> Predictive upshift to Gear 3
        batch_size, telem = vvt.update(
            shaft_rpm=2000.0,
            boost_psi=10.0,
            loss_delta=0.08,
        )
        self.assertEqual(vvt.current_gear, 3)
        self.assertEqual(batch_size, 64)
        self.assertTrue(telem["predictive_shifted"])

        # 2. Loss destabilization / spike (loss_delta = -0.05) -> Predictive downshift to Gear 1 (agile recovery)
        batch_size, telem = vvt.update(
            shaft_rpm=2000.0,
            boost_psi=10.0,
            loss_delta=-0.05,
        )
        self.assertEqual(vvt.current_gear, 1)
        self.assertEqual(batch_size, 16)
        self.assertTrue(telem["predictive_shifted"])

    def test_cached_air_filter_bounds(self):
        import torch
        from accelerator_ai.turbines.filter import AirFilter
        from accelerator_ai.core.flow_packet import FlowPacket

        af = AirFilter(cache_window=5, fast_mode=False)
        x = torch.randn(16, 8)
        packet = FlowPacket(x=x, y=None)

        # Step 0: computes and caches
        af.process(packet)
        self.assertIsNotNone(af._cached_mean)
        self.assertIsNotNone(af._cached_bound)
        initial_mean = af._cached_mean.clone()

        # Step 1: reuses cached mean
        x2 = torch.randn(16, 8)
        p2 = FlowPacket(x=x2, y=None)
        af.process(p2)
        self.assertTrue(torch.equal(af._cached_mean, initial_mean))

    def test_braided_dna_zero_churn_buffers(self):
        from accelerator_ai.ecu.braided_controller import BraidedDNAController
        controller = BraidedDNAController(base_learning_rate=0.02)
        self.assertEqual(controller._phases_buf.shape, (4,))
        self.assertEqual(controller._weights_buf.shape, (4,))

        status = controller.update(
            step=1,
            learning_torque=1.2,
            boost_ratio=1.4,
            injected_entropy=0.1,
            pyrometer_temp=450.0,
            loss=1.8,
        )
        self.assertIn("learning_rate", status)
        self.assertIn("resonance_index", status)
        self.assertGreater(status["learning_rate"], 0.0)

    def test_hierarchical_turbo_levels(self):
        from accelerator_ai.engine import (
            TurboLearningEngine,
            TURBO_LEVEL_CRUISE,
            TURBO_LEVEL_REGULATED,
            TURBO_LEVEL_RESONANT,
            TURBO_LEVEL_FULL,
        )
        from accelerator_ai.models.neural_core import PureNumPyMLP
        m = PureNumPyMLP(layer_sizes=[4, 8, 2])
        engine = TurboLearningEngine(
            model=m,
            enable_default_injectors=False,
            adaptive_turbo=True,
            adaptive_check_interval=10,
        )

        # Early spool (step <= 50) -> TURBO_LEVEL_FULL
        self.assertEqual(engine._get_turbo_level(1), TURBO_LEVEL_FULL)
        self.assertEqual(engine._get_turbo_level(50), TURBO_LEVEL_FULL)

        # Step 51 not periodic, not div 3 -> TURBO_LEVEL_CRUISE
        self.assertEqual(engine._get_turbo_level(52), TURBO_LEVEL_CRUISE)

        # Step 54 divisible by 3 -> TURBO_LEVEL_RESONANT
        self.assertEqual(engine._get_turbo_level(54), TURBO_LEVEL_RESONANT)

        # Wastegate knock active -> TURBO_LEVEL_FULL
        engine.wastegate.open_pct = 40.0
        self.assertEqual(engine._get_turbo_level(52), TURBO_LEVEL_FULL)
        engine.wastegate.open_pct = 0.0


if __name__ == "__main__":
    unittest.main()
