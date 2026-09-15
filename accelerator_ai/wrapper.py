"""
High-Level Ergonomic Wrapping API for AcceleratorAI.

Allows wrapping any PyTorch nn.Module and optimizer into a fully functional
TurboLearningEngine in a single line of code.
"""

from typing import Any, Optional
from accelerator_ai.config import EngineConfig
from accelerator_ai.models.torch_adapter import PyTorchTurbineWrapper
from accelerator_ai.engine import TurboLearningEngine


def wrap(
    model: Any,
    optimizer: Optional[Any] = None,
    loss_fn: Optional[Any] = None,
    config: Optional[EngineConfig] = None,
    target_boost_psi: float = 14.7,
    base_learning_rate: float = 0.001,
    fast_physics: bool = True,
    enable_amp: bool = False,
    amp_dtype: str = "bfloat16",
    enable_cuda_graph: bool = False,
    adaptive_turbo: bool = True,
    **kwargs: Any,
) -> TurboLearningEngine:
    """
    Wrap any PyTorch model and optimizer into an AcceleratorAI TurboLearningEngine.

    Example:
        ```python
        import torch
        import accelerator_ai

        model = MyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # 1-Liner: Wrap model and optimizer
        engine = accelerator_ai.wrap(model, optimizer)

        # In your training loop:
        for epoch in range(epochs):
            for x, y in dataloader:
                loss = engine.step(x, y)  # or engine(x, y)
        ```

    Args:
        model: torch.nn.Module or custom model
        optimizer: torch.optim.Optimizer (for PyTorch models)
        loss_fn: Optional loss function (defaults to internal cross-entropy / MSE)
        config: Optional strongly-typed EngineConfig
        target_boost_psi: Target manifold boost pressure in PSI (default: 14.7 PSI = 1.0 atm)
        base_learning_rate: Baseline learning rate
        fast_physics: Enable zero-intermediate allocation fast execution
        enable_amp: Enable hardware automatic mixed precision (BF16/FP16)
        amp_dtype: AMP precision ("bfloat16" or "float16")
        enable_cuda_graph: Enable zero-sync CUDA Graph execution
        adaptive_turbo: Dynamic cruising vs boost governor
        **kwargs: Additional parameters passed to TurboLearningEngine

    Returns:
        TurboLearningEngine instance ready for engine.step(x, y) or engine(x, y).
    """
    if isinstance(model, TurboLearningEngine):
        return model

    if hasattr(model, "parameters") and optimizer is not None:
        wrapped_model = PyTorchTurbineWrapper(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            enable_amp=enable_amp,
            amp_dtype=amp_dtype,
        )
    else:
        wrapped_model = model

    engine = TurboLearningEngine(
        model=wrapped_model,
        config=config,
        target_boost_psi=target_boost_psi,
        base_learning_rate=base_learning_rate,
        fast_physics=fast_physics,
        enable_amp=enable_amp,
        amp_dtype=amp_dtype,
        enable_cuda_graph=enable_cuda_graph,
        adaptive_turbo=adaptive_turbo,
        **kwargs,
    )
    return engine
