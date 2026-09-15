"""
PyTorch Lightning Integration for AcceleratorAI.

Provides AcceleratorAILightningCallback for automated training acceleration,
zero-sync pneumatic gradient regulation, and dynamic boost management in PyTorch Lightning.
"""

import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

try:
    from lightning.pytorch.callbacks import Callback
    _LIGHTNING_AVAILABLE = True
except ImportError:
    try:
        from pytorch_lightning.callbacks import Callback
        _LIGHTNING_AVAILABLE = True
    except ImportError:
        _LIGHTNING_AVAILABLE = False
        Callback = object  # type: ignore


class AcceleratorAILightningCallback(Callback):
    """
    Seamless PyTorch Lightning integration.

    Usage:
        import lightning as L
        from accelerator_ai.integrations import AcceleratorAILightningCallback

        trainer = L.Trainer(
            callbacks=[AcceleratorAILightningCallback(target_boost_psi=14.7)],
            max_epochs=10,
        )
        trainer.fit(model, dataloader)
    """

    def __init__(
        self,
        target_boost_psi: float = 14.7,
        enable_soft_clipping: bool = True,
        clip_threshold: float = 1.0,
        adaptive_boost: bool = True,
    ):
        if not _LIGHTNING_AVAILABLE:
            raise ImportError(
                "Neither 'lightning' nor 'pytorch_lightning' is installed. "
                "Install with 'pip install lightning' to use AcceleratorAILightningCallback."
            )
        self.target_boost_psi = target_boost_psi
        self.enable_soft_clipping = enable_soft_clipping
        self.clip_threshold = clip_threshold
        self.adaptive_boost = adaptive_boost
        self.boost_ratio: float = 1.0
        self.recent_losses = []

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        logger.info("AcceleratorAI Lightning Turbine Callback active.")
        logger.info(f"Target Boost: {self.target_boost_psi} PSI | Soft-Clipping: {self.enable_soft_clipping}")

    def on_before_optimizer_step(self, trainer: Any, pl_module: Any, optimizer: Any) -> None:
        """Applies fused on-device wastegate soft-clipping before weight updates."""
        if not self.enable_soft_clipping:
            return

        try:
            import torch
            grads = [p.grad for p in pl_module.parameters() if p.grad is not None]
            if not grads:
                return

            norms = torch._foreach_norm(grads, 2)
            total_norm = torch.linalg.vector_norm(torch.stack(list(norms)))
            thresh = self.clip_threshold * self.boost_ratio

            scale = torch.tanh(thresh / (total_norm + 1e-7))
            torch._foreach_mul_(grads, scale)
        except Exception as e:
            logger.debug(f"Lightning soft-clipping bypass: {e}")

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Captures batch loss for adaptive turbo governor."""
        if isinstance(outputs, dict) and "loss" in outputs:
            loss_val = float(outputs["loss"].item())
        elif hasattr(outputs, "item"):
            loss_val = float(outputs.item())
        else:
            return

        self.recent_losses.append(loss_val)
        if len(self.recent_losses) > 30:
            self.recent_losses.pop(0)

        # Dynamic governor
        if self.adaptive_boost and len(self.recent_losses) >= 5:
            delta = self.recent_losses[-5] - self.recent_losses[-1]
            if delta < 0.0005:
                # Stall detected
                self.boost_ratio = min(2.5, self.boost_ratio * 1.05)
            else:
                self.boost_ratio = max(1.0, self.boost_ratio * 0.99)
