"""
PyTorchTurbineWrapper: Adapter bridge to run any PyTorch nn.Module
within the AcceleratorAI VIBE Turbine training pipeline.
"""

from typing import Tuple, Any, Optional
import numpy as np


class PyTorchTurbineWrapper:
    """
    Wraps standard PyTorch torch.nn.Module and torch.optim.Optimizer
    into the TurbineModel interface expected by CombustionChamber and GradientTurbine.
    """

    def __init__(self, model: Any, optimizer: Any, loss_fn: Any):
        try:
            import torch
            self.torch = torch
        except ImportError:
            raise ImportError(
                "PyTorch is not installed in the current environment. "
                "Install with 'pip install torch' to use PyTorchTurbineWrapper, "
                "or use PureNumPyMLP for pure NumPy execution."
            )

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.last_loss_tensor: Optional[Any] = None

    def forward_and_loss(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """Converts NumPy batch to torch.Tensor, runs forward pass and evaluates loss."""
        x_tensor = self.torch.from_numpy(x).float()
        if y.ndim == 1:
            y_tensor = self.torch.from_numpy(y).long()
        else:
            y_tensor = self.torch.from_numpy(y).float()

        self.optimizer.zero_grad()
        predictions = self.model(x_tensor)
        loss = self.loss_fn(predictions, y_tensor)
        self.last_loss_tensor = loss

        pred_np = predictions.detach().cpu().numpy()
        loss_val = float(loss.item())
        return pred_np, loss_val

    def backward(self) -> float:
        """Runs loss.backward() and computes the gradient L2 norm."""
        if self.last_loss_tensor is None:
            return 0.0

        self.last_loss_tensor.backward()

        total_sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_sq = p.grad.detach().data.norm(2).item() ** 2
                total_sq += param_sq

        return float(np.sqrt(total_sq))

    def clip_gradients(self, max_norm: float) -> None:
        """Wastegate gradient clipping."""
        self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)

    def apply_updates(self, learning_rate: float) -> None:
        """Drive Shaft parameter update."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate
        self.optimizer.step()
