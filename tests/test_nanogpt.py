"""
Unit tests for NanoGPT Causal Transformer and LLM token sequence training dynamics.
"""

import unittest
import torch
import numpy as np

from accelerator_ai.benchmarks.nanogpt import NanoGPT, GPTConfig
from accelerator_ai.models.torch_adapter import PyTorchTurbineWrapper
from accelerator_ai.engine import TurboLearningEngine
from accelerator_ai.turbines.wastegate import WastegateValve


class TestNanoGPTIntegration(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.config = GPTConfig(
            vocab_size=128,
            block_size=32,
            n_layer=2,
            n_head=2,
            n_embd=32,
            dropout=0.0,
        )
        self.model = NanoGPT(self.config)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.device = "cpu"  # unit tests run robustly on CPU

    def test_nanogpt_forward_and_loss(self):
        """Verifies forward pass with targets calculates cross entropy loss."""
        x = torch.randint(0, 128, (4, 32))
        y = torch.randint(0, 128, (4, 32))

        logits, loss = self.model(x, targets=y)
        self.assertIsNotNone(loss)
        self.assertEqual(logits.shape, (4, 32, 128))
        self.assertGreater(float(loss.item()), 0.0)

    def test_pytorch_turbine_wrapper_with_nanogpt(self):
        """Verifies wrapper executes internal loss calculation and backward pass."""
        wrapper = PyTorchTurbineWrapper(self.model, self.optimizer, loss_fn=None)

        x = torch.randint(0, 128, (4, 32))
        y = torch.randint(0, 128, (4, 32))

        preds, loss_val = wrapper.forward_and_loss(x, y)
        self.assertGreater(loss_val, 0.0)

        grad_norm = wrapper.backward()
        self.assertGreater(grad_norm, 0.0)

        wrapper.apply_updates(learning_rate=1e-3)

    def test_engine_closed_loop_with_nanogpt(self):
        """Verifies TurboLearningEngine completes a full cycle with discrete tokens."""
        wrapper = PyTorchTurbineWrapper(self.model, self.optimizer, loss_fn=None)
        engine = TurboLearningEngine(
            model=wrapper,
            enable_default_injectors=False,
            enable_vvt=True,
            fault_tolerance_mode=False,
        )

        x = torch.randint(0, 128, (32, 32))
        y = torch.randint(0, 128, (32, 32))

        res1 = engine.step(x, y)
        self.assertGreater(res1.loss, 0.0)
        self.assertGreater(res1.exhaust_energy, 0.0)

        res2 = engine.step(x, y)
        self.assertGreater(res2.loss, 0.0)

    def test_nanogpt_soft_clipping(self):
        """Verifies soft-clipping smoothly bounds gradients on transformer weights."""
        wrapper = PyTorchTurbineWrapper(self.model, self.optimizer, loss_fn=None)
        x = torch.randint(0, 128, (4, 32))
        y = torch.randint(0, 128, (4, 32))

        wrapper.forward_and_loss(x, y)
        raw_norm = wrapper.backward()

        clipped_norm = wrapper.soft_clip_gradients(threshold=0.1)
        self.assertLessEqual(clipped_norm, 0.15)


if __name__ == "__main__":
    unittest.main()
