"""
Unit tests for the 5 new performance optimizations:
1. Asynchronous CUDAPrefetcher (non-blocking stream overlap)
2. Fast Minimax Taylor Trigonometry & Global Spool-Down (BraidedDNAController)
3. Micro-Batch Gradient Accumulation with intra-step Wastegate (TurboLearningEngine)
4. PyTorch Native On-Device Swirl Dispersion (SwirlDispersionValve)
5. Zero-Grad Manual Decoupling (PyTorchTurbineWrapper)
"""

import unittest
import torch
import torch.nn as nn
import numpy as np

from accelerator_ai.core.prefetcher import CUDAPrefetcher
from accelerator_ai.core.flow_packet import FlowPacket
from accelerator_ai.ecu.braided_controller import BraidedDNAController
from accelerator_ai.turbines.dispersion_valve import SwirlDispersionValve
from accelerator_ai.models.torch_adapter import PyTorchTurbineWrapper
from accelerator_ai.engine import TurboLearningEngine
from accelerator_ai.config import EngineConfig


class DummyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 2)

    def forward(self, x):
        return self.linear(x)


class TestPerfEnhancements(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

    def test_cuda_prefetcher_cpu_and_cuda(self):
        """Test CUDAPrefetcher loads batches correctly with non-blocking stream."""
        batches = [
            (torch.randn(4, 8), torch.randint(0, 2, (4,))),
            (torch.randn(4, 8), torch.randint(0, 2, (4,))),
            (torch.randn(4, 8), torch.randint(0, 2, (4,))),
        ]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prefetcher = CUDAPrefetcher(batches, device=device)

        collected = []
        for x, y in prefetcher:
            self.assertIsNotNone(x)
            self.assertIsNotNone(y)
            self.assertEqual(x.device.type, device.type)
            self.assertEqual(y.device.type, device.type)
            self.assertEqual(x.shape, (4, 8))
            collected.append((x, y))

        self.assertEqual(len(collected), 3)

        # Test prefetcher reset
        prefetcher.reset()
        x1, y1 = prefetcher.next()
        self.assertIsNotNone(x1)
        self.assertEqual(x1.shape, (4, 8))

    def test_fast_trig_and_global_spool_down(self):
        """Test Minimax Taylor Cosine accuracy and global spool-down LR decay."""
        # 1. Compare fast_trig=True vs fast_trig=False
        ctrl_fast = BraidedDNAController(base_learning_rate=0.015, fast_trig=True)
        ctrl_std = BraidedDNAController(base_learning_rate=0.015, fast_trig=False)

        metrics = {
            "epoch": 1,
            "loss": 1.5,
            "pyrometer_temp": 55.0,
            "rpm": 3200.0,
            "compressor_boost": 1.2,
            "exhaust_wastegate_open_pct": 0.0,
            "wastegate_state": "CLOSED",
        }

        for s in range(1, 30):
            res_fast = ctrl_fast.update(
                step=s,
                learning_torque=1.5,
                boost_ratio=1.2,
                injected_entropy=0.5,
                pyrometer_temp=350.0,
                loss=1.5,
            )
            res_std = ctrl_std.update(
                step=s,
                learning_torque=1.5,
                boost_ratio=1.2,
                injected_entropy=0.5,
                pyrometer_temp=350.0,
                loss=1.5,
            )
            lr_fast = res_fast["learning_rate"]
            lr_std = res_std["learning_rate"]
            # Relative difference must be within 5% (fast polynomial approximation accumulated over steps)
            rel_diff = abs(lr_fast - lr_std) / lr_std
            self.assertLess(rel_diff, 0.05, f"Step {s}: fast_trig diverged by {rel_diff * 100:.3f}%")

        # 2. Test global spool-down decay
        total_steps = 100
        ctrl_spool = BraidedDNAController(
            base_learning_rate=0.01,
            total_steps=total_steps,
            min_lr_ratio=0.1,
        )
        early_res = ctrl_spool.update(step=1, learning_torque=1.0, boost_ratio=1.0, injected_entropy=0.0, pyrometer_temp=200.0, loss=2.0)
        mid_res = ctrl_spool.update(step=50, learning_torque=1.0, boost_ratio=1.0, injected_entropy=0.0, pyrometer_temp=200.0, loss=1.5)
        late_res = ctrl_spool.update(step=100, learning_torque=1.0, boost_ratio=1.0, injected_entropy=0.0, pyrometer_temp=200.0, loss=1.0)

        early_lr = early_res["learning_rate"]
        mid_lr = mid_res["learning_rate"]
        late_lr = late_res["learning_rate"]

        # Late LR should have decayed significantly towards min_lr_ratio * base_lr (0.001)
        self.assertGreater(early_lr, mid_lr)
        self.assertGreater(mid_lr, late_lr)
        self.assertAlmostEqual(late_lr, 0.001, delta=0.002)

    def test_gradient_accumulation_step(self):
        """Test step_accumulated micro-batching without premature optimizer steps."""
        model = DummyModule()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        wrapper = PyTorchTurbineWrapper(model=model, optimizer=optimizer, loss_fn=loss_fn)

        engine = TurboLearningEngine(
            model=wrapper,
            config=EngineConfig(gradient_accumulation_steps=2),
        )

        initial_param = model.linear.weight.clone().detach()

        x1 = np.random.randn(8, 8).astype(np.float32)
        y1 = np.random.randint(0, 2, (8,)).astype(np.int64)
        x2 = np.random.randn(8, 8).astype(np.float32)
        y2 = np.random.randint(0, 2, (8,)).astype(np.int64)

        # Step 1 of 2 (micro-step: accumulate gradients, no weight update)
        res1 = engine.step_accumulated(x1, y1, step_in_cycle=1, total_cycle_steps=2)
        self.assertIsNotNone(res1.loss)
        self.assertEqual(engine.current_step, 0)
        self.assertTrue(torch.allclose(model.linear.weight, initial_param))
        self.assertIsNotNone(model.linear.weight.grad)

        # Step 2 of 2 (final step: apply updates, increment step)
        res2 = engine.step_accumulated(x2, y2, step_in_cycle=2, total_cycle_steps=2)
        self.assertIsNotNone(res2.loss)
        self.assertEqual(engine.current_step, 1)
        # Weights should now have changed
        self.assertFalse(torch.allclose(model.linear.weight, initial_param))

    def test_pytorch_native_dispersion_tokens_and_floats(self):
        """Test SwirlDispersionValve handles PyTorch discrete tokens and float tensors on-device."""
        valve = SwirlDispersionValve(diffusive_mode=True, stratified_swirl=True)

        # Case 1: Discrete integer tokens (LLM prompt tokens)
        tokens_main = torch.randint(0, 1000, (16, 64), dtype=torch.long)
        tokens_inj = torch.randint(0, 1000, (4, 64), dtype=torch.long)
        main_packet = FlowPacket(x=tokens_main, y=None, pressure=1.5)
        inj_packet = FlowPacket(x=tokens_inj, y=None, source="synthetic_injector")

        out_packet = valve.disperse_and_mix(main_packet, [inj_packet])
        self.assertIsInstance(out_packet.x, torch.Tensor)
        self.assertEqual(out_packet.x.dtype, torch.long)
        self.assertEqual(out_packet.x.shape, (20, 64))
        self.assertEqual(valve.last_homogeneity_pct, 100.0)

        # Case 2: Continuous float embeddings
        embed_main = torch.randn(16, 32, dtype=torch.float32)
        embed_inj = torch.randn(8, 32, dtype=torch.float32)
        main_packet_float = FlowPacket(x=embed_main, y=None, pressure=1.2)
        inj_packet_float = FlowPacket(x=embed_inj, y=None, source="synthetic_injector")

        out_float = valve.disperse_and_mix(main_packet_float, [inj_packet_float])
        self.assertIsInstance(out_float.x, torch.Tensor)
        self.assertEqual(out_float.x.dtype, torch.float32)
        self.assertEqual(out_float.x.shape, (16, 32))
        self.assertGreater(valve.last_homogeneity_pct, 50.0)

    def test_zero_grad_manual_decoupling(self):
        """Test PyTorchTurbineWrapper zero_grad toggle in forward_and_loss."""
        model = DummyModule()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        wrapper = PyTorchTurbineWrapper(model=model, optimizer=optimizer, loss_fn=loss_fn)

        x = torch.randn(4, 8)
        y = torch.randint(0, 2, (4,))

        # Backward once to create non-zero gradients
        _, loss = wrapper.forward_and_loss(x, y, zero_grad=True)
        wrapper.backward()
        self.assertIsNotNone(model.linear.weight.grad)
        grad_sum_before = model.linear.weight.grad.abs().sum().item()
        self.assertGreater(grad_sum_before, 0.0)

        # Forward with zero_grad=False preserves existing grads
        _, loss2 = wrapper.forward_and_loss(x, y, zero_grad=False)
        self.assertIsNotNone(model.linear.weight.grad)
        self.assertGreater(model.linear.weight.grad.abs().sum().item(), 0.0)

        # Explicit manual zero_grad clears them
        wrapper.zero_grad()
        if model.linear.weight.grad is not None:
            self.assertEqual(model.linear.weight.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
