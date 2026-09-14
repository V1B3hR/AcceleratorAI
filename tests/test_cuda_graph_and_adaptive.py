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


if __name__ == "__main__":
    unittest.main()
