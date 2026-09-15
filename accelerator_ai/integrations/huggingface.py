"""
Hugging Face Transformers Integration for AcceleratorAI.

Provides AcceleratorAICallback for automated training acceleration,
dynamic VVT modulation, and wastegate soft-clipping within transformers.Trainer.
"""

import logging
from typing import Optional, Any, Dict

logger = logging.getLogger(__name__)

try:
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    TrainerCallback = object  # type: ignore
    TrainerControl = Any      # type: ignore
    TrainerState = Any        # type: ignore
    TrainingArguments = Any   # type: ignore


class AcceleratorAICallback(TrainerCallback):
    """
    Seamless Hugging Face Trainer integration.

    Usage:
        from transformers import Trainer
        from accelerator_ai.integrations import AcceleratorAICallback

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            callbacks=[AcceleratorAICallback(target_boost_psi=14.7, enable_soft_clipping=True)],
        )
        trainer.train()
    """

    def __init__(
        self,
        target_boost_psi: float = 14.7,
        enable_soft_clipping: bool = True,
        clip_threshold: float = 1.0,
        enable_telemetry: bool = True,
        adaptive_boost: bool = True,
    ):
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is not installed. Install with 'pip install transformers' "
                "to use AcceleratorAICallback."
            )
        self.target_boost_psi = target_boost_psi
        self.enable_soft_clipping = enable_soft_clipping
        self.clip_threshold = clip_threshold
        self.enable_telemetry = enable_telemetry
        self.adaptive_boost = adaptive_boost

        self.last_val_loss: Optional[float] = None
        self.recent_losses = []
        self.boost_ratio: float = 1.0
        self.current_gear: int = 2

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logger.info("AcceleratorAI Turbine Callback engaged on Hugging Face Trainer.")
        logger.info(f"Target Boost: {self.target_boost_psi} PSI | Soft-Clipping: {self.enable_soft_clipping}")

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Modulates dynamic pressure ahead of the forward pass."""
        # Calculate dynamic boost ratio
        if self.adaptive_boost and len(self.recent_losses) >= 5:
            delta = self.recent_losses[-5] - self.recent_losses[-1]
            if delta < 0.001:
                # Stall detected -> increase boost
                self.boost_ratio = min(2.5, self.boost_ratio * 1.1)
            else:
                # Steady cruising
                self.boost_ratio = max(1.0, self.boost_ratio * 0.98)

    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """Applies pneumatic soft-clipping after backward pass before optimizer step."""
        if self.enable_soft_clipping and model is not None:
            self._apply_wastegate_soft_clipping(model)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """Captures telemetry and loss history."""
        if state.log_history:
            latest = state.log_history[-1]
            if "loss" in latest:
                loss_val = latest["loss"]
                self.recent_losses.append(loss_val)
                if len(self.recent_losses) > 50:
                    self.recent_losses.pop(0)

        if self.enable_soft_clipping and model is not None:
            self._apply_wastegate_soft_clipping(model)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        """Monitors validation loss to bust plateau stalls."""
        if metrics and "eval_loss" in metrics:
            eval_loss = metrics["eval_loss"]
            if self.last_val_loss is not None and (self.last_val_loss - eval_loss) < 0.005:
                logger.info("AcceleratorAI detected evaluation plateau. Spooling auxiliary turbine boost.")
                self.boost_ratio = min(2.5, self.boost_ratio * 1.25)
            self.last_val_loss = eval_loss

    def _apply_wastegate_soft_clipping(self, model: Any):
        """Smooth tanh soft-clipping across model parameters."""
        try:
            import torch
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not grads:
                return

            # Compute fused L2 norm on device
            norms = torch._foreach_norm(grads, 2)
            total_norm = torch.linalg.vector_norm(torch.stack(list(norms)))
            thresh = self.clip_threshold * self.boost_ratio

            # Soft clip with tanh smooth pressure relief
            scale = torch.tanh(thresh / (total_norm + 1e-7))
            torch._foreach_mul_(grads, scale)
        except Exception as e:
            logger.debug(f"Soft-clipping bypass: {e}")
