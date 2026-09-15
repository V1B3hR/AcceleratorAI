"""
PyTorchTurbineWrapper: Adapter bridge to run any PyTorch nn.Module
within the AcceleratorAI VIBE Turbine training pipeline.

Features:
- Supports curriculum-weighted cross-entropy loss from VGT manifolds.
- Registers gradient torque hooks across nn.Module layers.
- Seamless compatibility with CombustionChamber, SequentialTurboSystem, and DriveShaft.
"""

from typing import Tuple, Any, Optional, Dict
import contextlib
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
        enable_amp: bool = False,
        amp_dtype: str = "bfloat16",
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
        self.enable_amp = enable_amp
        self.amp_dtype = amp_dtype.lower()

        self.scaler = None
        if self.enable_amp and self.amp_dtype in ("float16", "fp16") and self.torch.cuda.is_available():
            self.scaler = self.torch.amp.GradScaler('cuda')

        self.last_loss_tensor: Optional[Any] = None
        self.last_sample_weights: Optional[np.ndarray] = None
        self.layer_gradient_torques: Dict[str, float] = {}

        if self.enable_layer_hooks:
            self._register_hooks()

    def enable_capturable_optimizer(self) -> None:
        """Enables capturable=True on optimizer parameter groups for CUDA Graph execution."""
        for param_group in self.optimizer.param_groups:
            param_group["capturable"] = True


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
        zero_grad: bool = True,
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

        if zero_grad:
            try:
                self.optimizer.zero_grad(set_to_none=True)
            except TypeError:
                self.optimizer.zero_grad()
        self.last_sample_weights = sample_weights

        # Resolve AMP context
        if self.enable_amp and device.type == "cuda":
            torch_dtype = self.torch.bfloat16 if self.amp_dtype in ("bfloat16", "bf16") else self.torch.float16
            amp_ctx = self.torch.amp.autocast(device_type="cuda", dtype=torch_dtype)
        else:
            amp_ctx = contextlib.nullcontext()

        with amp_ctx:
            # 2. Forward pass execution
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

        is_capturing = self.torch.cuda.is_current_stream_capturing() if self.torch.cuda.is_available() else False
        pred_out = predictions.detach().cpu().numpy() if (return_numpy_preds and not is_capturing) else predictions.detach()
        loss_val = 0.0 if is_capturing else float(loss.item())
        return pred_out, loss_val

    def backward(self) -> float:
        """Runs loss.backward() and computes total parameter gradient L2 norm with zero intermediate device syncs."""
        if self.last_loss_tensor is None:
            return 0.0

        if self.scaler is not None:
            self.scaler.scale(self.last_loss_tensor).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            self.last_loss_tensor.backward()

        self._last_grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        if not self._last_grads:
            return 0.0

        # Fused multi-tensor GPU norm reduction
        if hasattr(self.torch, "_foreach_norm"):
            norms = self.torch._foreach_norm(self._last_grads, 2)
            self._last_grad_norm_t = self.torch.linalg.vector_norm(self.torch.stack(norms))
        else:
            total_sq = sum(g.data.norm(2).square() for g in self._last_grads)
            self._last_grad_norm_t = self.torch.sqrt(total_sq)

        is_capturing = self.torch.cuda.is_current_stream_capturing() if self.torch.cuda.is_available() else False
        self._last_grad_norm = 0.0 if is_capturing else float(self._last_grad_norm_t.item())
        return self._last_grad_norm

    def harvest_and_regulate_fused(
        self,
        threshold: float = 5.0,
        boost_ratio: float = 1.0,
        enable_soft_clipping: bool = True,
        skip_backward: bool = False,
    ) -> Tuple[float, float, bool]:
        """
        Fused on-device gradient harvesting and pneumatic wastegate regulation.
        Performs backward pass (unless skip_backward=True), norm calculation, and tanh soft-clipping directly on GPU
        in a single fused operation without host-device synchronization stalls.

        Returns:
            (raw_norm, clipped_norm, was_vented)
        """
        if not skip_backward:
            if self.last_loss_tensor is None:
                return 0.0, 0.0, False

            if self.scaler is not None:
                self.scaler.scale(self.last_loss_tensor).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                self.last_loss_tensor.backward()
        elif self.scaler is not None and getattr(self.scaler, "is_enabled", lambda: True)():
            try:
                self.scaler.unscale_(self.optimizer)
            except RuntimeError:
                pass  # Already unscaled

        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        if not grads:
            return 0.0, 0.0, False

        self._last_grads = grads
        eff_threshold = threshold * boost_ratio

        # Fused multi-tensor GPU norm reduction
        if hasattr(self.torch, "_foreach_norm"):
            norms = self.torch._foreach_norm(grads, 2)
            norm_t = self.torch.linalg.vector_norm(self.torch.stack(norms))
        else:
            total_sq = sum(g.data.norm(2).square() for g in grads)
            norm_t = self.torch.sqrt(total_sq)

        self._last_grad_norm_t = norm_t
        is_capturing = self.torch.cuda.is_current_stream_capturing() if self.torch.cuda.is_available() else False

        if enable_soft_clipping:
            scale_t = self.torch.tanh(eff_threshold / (norm_t + 1e-8))
            if hasattr(self.torch, "_foreach_mul_"):
                self.torch._foreach_mul_(grads, scale_t)
            else:
                scale_val = float(scale_t.item())
                for g in grads:
                    g.mul_(scale_val)

            clipped_norm_t = norm_t * scale_t
            if is_capturing:
                raw_norm = 0.0
                clipped_norm = 0.0
                was_vented = False
            else:
                raw_norm = float(norm_t.item())
                clipped_norm = float(clipped_norm_t.item())
                was_vented = raw_norm > eff_threshold
        else:
            if is_capturing:
                raw_norm = 0.0
                clipped_norm = 0.0
                was_vented = False
            else:
                raw_norm = float(norm_t.item())
                if raw_norm > eff_threshold:
                    self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=eff_threshold)
                    clipped_norm = eff_threshold
                    was_vented = True
                else:
                    clipped_norm = raw_norm
                    was_vented = False

        self._last_grad_norm = clipped_norm
        return raw_norm, clipped_norm, was_vented

    def clip_gradients(self, max_norm: float) -> None:
        """Wastegate hard gradient clipping."""
        self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)

    def soft_clip_gradients(self, threshold: float, precomputed_norm: Optional[float] = None) -> float:
        """Pneumatic soft-clipping via smooth tanh scaling using fused multi-tensor kernel."""
        grads = getattr(self, "_last_grads", None)
        if grads is None or len(grads) == 0:
            grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        if not grads or threshold <= 0.0:
            return 0.0

        if hasattr(self, "_last_grad_norm_t") and self._last_grad_norm_t is not None:
            norm_t = self._last_grad_norm_t
        else:
            if hasattr(self.torch, "_foreach_norm"):
                norms = self.torch._foreach_norm(grads, 2)
                norm_t = self.torch.linalg.vector_norm(self.torch.stack(norms))
            else:
                total_sq = sum(g.data.norm(2).square() for g in grads)
                norm_t = self.torch.sqrt(total_sq)

        scale_t = self.torch.tanh(threshold / (norm_t + 1e-8))
        if hasattr(self.torch, "_foreach_mul_"):
            self.torch._foreach_mul_(grads, scale_t)
        else:
            scale_val = float(scale_t.item())
            for g in grads:
                g.mul_(scale_val)

        clipped_norm_t = norm_t * scale_t
        return float(clipped_norm_t.item())

    def apply_updates(self, learning_rate: float) -> None:
        """Drive Shaft parameter update."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

    def zero_grad(self) -> None:
        """Manually clear optimizer gradients."""
        try:
            self.optimizer.zero_grad(set_to_none=True)
        except TypeError:
            self.optimizer.zero_grad()


