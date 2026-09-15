"""
Unit tests for AcceleratorAI integration layer:
- accelerator_ai.wrap() 1-liner API
- engine.__call__ and engine.eval_step
- Hugging Face Transformers AcceleratorAICallback
- PyTorch Lightning AcceleratorAILightningCallback
"""

import pytest
import numpy as np
import accelerator_ai
from accelerator_ai.engine import TurboLearningEngine
from accelerator_ai.integrations.huggingface import AcceleratorAICallback
from accelerator_ai.integrations.lightning import AcceleratorAILightningCallback

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch required")
class SimpleLinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch required")
def test_accelerator_ai_wrap_pytorch():
    """Tests the 1-liner wrap() helper with PyTorch model and optimizer."""
    model = SimpleLinearNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    engine = accelerator_ai.wrap(
        model=model,
        optimizer=optimizer,
        target_boost_psi=12.0,
        fast_physics=True,
    )

    assert isinstance(engine, TurboLearningEngine)
    assert engine.target_boost_psi == 12.0
    assert engine.fast_physics is True

    # Test engine(x, y) execution via __call__
    x = torch.randn(16, 4)
    y = torch.randint(0, 2, (16,))
    res = engine(x, y)

    assert res.loss is not None
    assert res.loss > 0.0
    assert engine.current_step == 1

    # Test engine.eval_step(x, y)
    preds, val_loss = engine.eval_step(x, y)
    assert preds is not None
    assert val_loss > 0.0
    # Step counter shouldn't advance on eval_step
    assert engine.current_step == 1


def test_accelerator_ai_wrap_numpy():
    """Tests wrap() with PureNumPyMLP or custom model."""
    from accelerator_ai.models.neural_core import PureNumPyMLP
    model = PureNumPyMLP(layer_sizes=[2, 4, 2], activation="tanh")

    engine = accelerator_ai.wrap(model=model, target_boost_psi=10.0)
    assert isinstance(engine, TurboLearningEngine)

    x = np.random.randn(8, 2).astype(np.float32)
    y = np.random.randint(0, 2, size=(8,)).astype(np.int64)
    res = engine(x, y)
    assert res.loss is not None


class DummyControl:
    pass


class DummyState:
    def __init__(self):
        self.log_history = [{"loss": 1.25}]


class DummyModel(nn.Module if _TORCH_AVAILABLE else object):
    def __init__(self):
        if _TORCH_AVAILABLE:
            super().__init__()
            self.p = nn.Parameter(torch.ones(3, 3))


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch required")
def test_huggingface_callback_mock():
    """Tests AcceleratorAICallback steps and soft-clipping regulation."""
    callback = AcceleratorAICallback(
        target_boost_psi=14.7,
        enable_soft_clipping=True,
        clip_threshold=1.0,
    )

    state = DummyState()
    control = DummyControl()
    model = DummyModel()
    model.p.grad = torch.randn(3, 3) * 5.0

    callback.on_step_begin(None, state, control)
    callback.on_step_end(None, state, control, model=model)

    clipped_norm = float(model.p.grad.norm(2).item())
    assert clipped_norm <= 5.0

    callback.on_evaluate(None, state, control, metrics={"eval_loss": 1.0})
    initial_boost = callback.boost_ratio
    callback.on_evaluate(None, state, control, metrics={"eval_loss": 0.999})
    assert callback.boost_ratio >= initial_boost


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch required")
def test_lightning_callback_mock():
    """Tests AcceleratorAILightningCallback logic."""
    from accelerator_ai.integrations.lightning import _LIGHTNING_AVAILABLE
    if not _LIGHTNING_AVAILABLE:
        pytest.skip("Neither 'lightning' nor 'pytorch_lightning' is installed in environment")

    callback = AcceleratorAILightningCallback(
        target_boost_psi=14.7,
        enable_soft_clipping=True,
    )

    model = DummyModel()
    model.p.grad = torch.randn(3, 3) * 10.0

    class DummyTrainer:
        pass

    trainer = DummyTrainer()
    callback.on_train_start(trainer, model)
    callback.on_before_optimizer_step(trainer, model, None)

    clipped_norm = float(model.p.grad.norm(2).item())
    assert clipped_norm <= 10.0

    loss_tensor = torch.tensor(0.42)
    callback.on_train_batch_end(trainer, model, loss_tensor, None, 0)
    assert len(callback.recent_losses) == 1
    assert callback.recent_losses[0] == pytest.approx(0.42, abs=1e-4)

