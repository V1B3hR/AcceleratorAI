"""
PyTorchTurbineWrapper: Adapter bridge to run any PyTorch nn.Module
within the AcceleratorAI VIBE Turbine training pipeline.

Features:
- Supports curriculum-weighted cross-entropy loss from VGT manifolds.
- Registers gradient torque hooks across nn.Module layers.
- Seamless compatibility with CombustionChamber, SequentialTurboSystem, and DriveShaft.
"""

from typing import Tuple, Any, Optional, Dict
import numpy as np


class PyTorchTurbineWrapper:
    """
    Wraps standard PyTorch torch.nn.Module and torch.optim.Optimizer
    into the TurbineModel interface expected by CombustionChamber and GradientTurbine.
    """

    def __init__(
        self,
        model: Any,
        optimizer: Any,
        loss_fn: Any,
        enable_layer_hooks: bool = False,
    ):
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
        self.enable_layer_hooks = enable_layer_hooks

        self.last_loss_tensor: Optional[Any] = None
        self.last_sample_weights: Optional[np.ndarray] = None
        self.layer_gradient_torques: Dict[str, float] = {}

        if self.enable_layer_hooks:
            self._register_hooks()

    def _register_hooks(self) -> None:
        """Registers backward hooks on parameter tensors to measure layer work."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                def make_hook(p_name):
                    def hook(grad):
                        if grad is not None:
                            self.layer_gradient_torques[p_name] = float(grad.detach().norm(2).item())
                    return hook
                param.register_hook(make_hook(name))

    def forward_and_loss(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Converts NumPy batch to torch.Tensor, runs forward pass and evaluates loss
        (with optional per-sample curriculum weights from VGT PortManifold).
        """
        x_tensor = self.torch.from_numpy(x).float()
        if y.ndim == 1:
            y_tensor = self.torch.from_numpy(y).long()
        else:
            y_tensor = self.torch.from_numpy(y).float()

        self.optimizer.zero_grad()
        predictions = self.model(x_tensor)
        self.last_sample_weights = sample_weights

        # Evaluate loss (curriculum-weighted if weights provided)
        if sample_weights is not None:
            sw_tensor = self.torch.from_numpy(sample_weights).float().to(x_tensor.device)
            # Try evaluating elementwise if loss function has reduction attribute
            if hasattr(self.loss_fn, "reduction"):
                orig_reduction = self.loss_fn.reduction
                self.loss_fn.reduction = "none"
                unreduced = self.loss_fn(predictions, y_tensor)
                self.loss_fn.reduction = orig_reduction
                if unreduced.ndim > 1:
                    unreduced = unreduced.mean(dim=-1)
                loss = (unreduced * sw_tensor).sum() / (sw_tensor.sum() + 1e-8)
            else:
                # Fallback: scale scalar loss by mean curriculum weight
                base_loss = self.loss_fn(predictions, y_tensor)
                loss = base_loss * sw_tensor.mean()
        else:
            loss = self.loss_fn(predictions, y_tensor)

        self.last_loss_tensor = loss

        pred_np = predictions.detach().cpu().numpy()
        loss_val = float(loss.item())
        return pred_np, loss_val

    def backward(self) -> float:
        """Runs loss.backward() and computes total parameter gradient L2 norm."""
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
