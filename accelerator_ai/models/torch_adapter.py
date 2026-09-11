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
        loss_fn: Optional[Any] = None,
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
        x: Any,
        y: Any,
        sample_weights: Optional[Any] = None,
        return_numpy_preds: bool = False,
    ) -> Tuple[Any, float]:
        """
        Runs forward pass and evaluates loss with automatic device placement.
        Accepts both torch.Tensor and numpy.ndarray inputs with zero redundant PCIe transfers.
        """
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = self.torch.device("cpu")

        # 1. Device and type alignment
        if isinstance(x, self.torch.Tensor):
            x_tensor = x.to(device)
        elif hasattr(x, "dtype") and np.issubdtype(x.dtype, np.integer):
            x_tensor = self.torch.from_numpy(x).long().to(device)
        else:
            x_tensor = self.torch.from_numpy(x).float().to(device)

        if isinstance(y, self.torch.Tensor):
            y_tensor = y.to(device)
        elif hasattr(y, "dtype") and np.issubdtype(y.dtype, np.integer):
            y_tensor = self.torch.from_numpy(y).long().to(device)
        elif hasattr(y, "ndim") and y.ndim == 1:
            y_tensor = self.torch.from_numpy(y).long().to(device)
        else:
            y_tensor = self.torch.from_numpy(y).float().to(device)

        self.optimizer.zero_grad()
        self.last_sample_weights = sample_weights

        # 2. Forward pass execution
        # If loss_fn is None, model is expected to accept targets and calculate internal loss (e.g. NanoGPT/LLMs)
        loss = None
        if self.loss_fn is None:
            try:
                predictions = self.model(x_tensor, y_tensor)
            except TypeError:
                try:
                    predictions = self.model(x_tensor, targets=y_tensor)
                except TypeError:
                    predictions = self.model(x_tensor)
        else:
            try:
                predictions = self.model(x_tensor)
            except TypeError:
                predictions = self.model(x_tensor, y_tensor)

        # 3. Loss resolution (model-internal loss or explicit loss_fn)
        if isinstance(predictions, tuple) and len(predictions) == 2:
            logits, internal_loss = predictions
            predictions = logits
            loss = internal_loss

        if loss is None:
            if self.loss_fn is not None:
                if sample_weights is not None:
                    if isinstance(sample_weights, self.torch.Tensor):
                        sw_tensor = sample_weights.to(device)
                    else:
                        sw_tensor = self.torch.from_numpy(sample_weights).float().to(device)

                    if hasattr(self.loss_fn, "reduction"):
                        orig_reduction = self.loss_fn.reduction
                        self.loss_fn.reduction = "none"
                        unreduced = self.loss_fn(predictions, y_tensor)
                        self.loss_fn.reduction = orig_reduction
                        if unreduced.ndim > 1:
                            unreduced = unreduced.mean(dim=-1)
                        loss = (unreduced * sw_tensor).sum() / (sw_tensor.sum() + 1e-8)
                    else:
                        base_loss = self.loss_fn(predictions, y_tensor)
                        loss = base_loss * sw_tensor.mean()
                else:
                    loss = self.loss_fn(predictions, y_tensor)
            else:
                raise ValueError("No loss_fn provided and model did not compute an internal loss.")
        else:
            # Model computed internal loss, apply curriculum weighting if present
            if sample_weights is not None:
                if isinstance(sample_weights, self.torch.Tensor):
                    sw_mean = sample_weights.to(device).mean()
                else:
                    sw_mean = float(np.mean(sample_weights))
                loss = loss * sw_mean

        self.last_loss_tensor = loss

        pred_out = predictions.detach().cpu().numpy() if return_numpy_preds else predictions.detach()
        loss_val = float(loss.item())
        return pred_out, loss_val

    def backward(self) -> float:
        """Runs loss.backward() and computes total parameter gradient L2 norm with zero intermediate device syncs."""
        if self.last_loss_tensor is None:
            return 0.0

        self.last_loss_tensor.backward()

        grads = [p.grad.detach() for p in self.model.parameters() if p.grad is not None]
        if not grads:
            return 0.0

        # Vectorized on-device sum of squares (single PCIe sync instead of N-parameter syncs)
        total_sq = sum(g.data.norm(2).square() for g in grads)
        return float(self.torch.sqrt(total_sq).item())

    def clip_gradients(self, max_norm: float) -> None:
        """Wastegate hard gradient clipping."""
        self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)

    def soft_clip_gradients(self, threshold: float) -> float:
        """Pneumatic soft-clipping via smooth tanh scaling with single on-device reduction."""
        grads = [p.grad.detach() for p in self.model.parameters() if p.grad is not None]
        if not grads or threshold <= 0.0:
            return 0.0

        total_sq = sum(g.data.norm(2).square() for g in grads)
        total_norm_t = self.torch.sqrt(total_sq)
        total_norm = float(total_norm_t.item())

        if total_norm > 1e-8:
            scale = float(self.torch.tanh(threshold / total_norm_t).item())
            for g in grads:
                g.mul_(scale)
            return total_norm * scale
        return total_norm

    def apply_updates(self, learning_rate: float) -> None:
        """Drive Shaft parameter update."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate
        self.optimizer.step()
