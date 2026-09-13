"""
Distributed Master-ECU Coordinator for AcceleratorAI.
Ensures cluster-wide lockstep synchronization across multi-GPU / DDP / FSDP setups.
"""

from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger("accelerator_ai.distributed")


class DistributedECUCoordinator:
    """
    Coordinates engine states across distributed workers in PyTorch DDP / FSDP.

    In distributed training, non-deterministic or divergent micro-batch sizing (VVT),
    shock injections, or wastegate clipping across different ranks causes immediate
    gradient reduction deadlocks and NCCL timeouts.

    Architecture (Master-ECU):
    - Rank 0 acts as the Master ECU, evaluating thermodynamic pressure, VVT gear,
      learning rate, and relief decisions.
    - Packs decisions into a compact 4-float tensor:
      [vvt_gear, learning_rate, wastegate_open_flag, shock_active_flag]
    - Performs an asynchronous or synchronous single-collective broadcast to all worker ranks.
    - All ranks adopt the exact same gear and execution path in lockstep.
    - In standalone mode (single GPU / CPU), operates with zero overhead.
    """

    def __init__(
        self,
        master_rank: int = 0,
        sync_interval: int = 1,
        use_cuda_stream: bool = False,
    ):
        self.master_rank = master_rank
        self.sync_interval = max(1, sync_interval)
        self.use_cuda_stream = use_cuda_stream
        self._is_distributed: bool = False
        self._rank: int = 0
        self._world_size: int = 1
        self._last_synced_state: Tuple[int, float, bool, bool] = (1, 0.015, False, False)
        self._cached_buffer: Optional[Any] = None
        self._cuda_stream: Optional[Any] = None
        self._check_distributed_status()

    def _check_distributed_status(self) -> None:
        """Inspects if PyTorch distributed communication group is active."""
        try:
            import torch
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                self._is_distributed = True
                self._rank = dist.get_rank()
                self._world_size = dist.get_world_size()
                if self.use_cuda_stream and torch.cuda.is_available():
                    self._cuda_stream = torch.cuda.Stream()
                logger.info(
                    "DistributedECUCoordinator initialized on rank %d/%d (Master=%d, SyncInterval=%d)",
                    self._rank,
                    self._world_size,
                    self.master_rank,
                    self.sync_interval,
                )
            else:
                self._is_distributed = False
                self._rank = 0
                self._world_size = 1
        except Exception:
            self._is_distributed = False
            self._rank = 0
            self._world_size = 1

    @property
    def is_distributed(self) -> bool:
        """Returns True if training inside a multi-node / multi-GPU distributed group."""
        return self._is_distributed

    @property
    def is_master(self) -> bool:
        """Returns True if the current process is the Master ECU (Rank 0)."""
        return (not self._is_distributed) or (self._rank == self.master_rank)

    @property
    def rank(self) -> int:
        """Current process rank index."""
        return self._rank

    @property
    def world_size(self) -> int:
        """Total number of participating distributed processes."""
        return self._world_size

    def broadcast_engine_state(
        self,
        vvt_gear: int,
        learning_rate: float,
        wastegate_open: bool,
        shock_fired: bool,
        device: Optional[Any] = None,
        step: Optional[int] = None,
    ) -> Tuple[int, float, bool, bool]:
        """
        Synchronizes engine dynamics across all distributed ranks.
        Rank 0 broadcasts its decision tensor; all worker ranks receive and adopt it.

        Args:
            vvt_gear: Desired VVT Gear index (1, 2, 3).
            learning_rate: Current learning rate calculated by Braided ECU.
            wastegate_open: True if wastegate relief was tripped.
            shock_fired: True if entropy shock nozzle fired.
            device: Optional torch.device for communication buffer.
            step: Optional training step index for event-driven / interval-based sync.

        Returns:
            Tuple of (synchronized_gear, synchronized_lr, synchronized_wastegate, synchronized_shock).
        """
        if not self._is_distributed:
            self._last_synced_state = (vvt_gear, learning_rate, wastegate_open, shock_fired)
            return self._last_synced_state

        # Throttled sync: If step is provided and not on interval, return last cached state
        if step is not None and self.sync_interval > 1:
            if step % self.sync_interval != 0 and not (wastegate_open or shock_fired):
                return self._last_synced_state

        try:
            import torch
            import torch.distributed as dist

            if device is None:
                if torch.cuda.is_available():
                    device = torch.device(f"cuda:{torch.cuda.current_device()}")
                else:
                    device = torch.device("cpu")

            # Persistent pre-allocated device buffer (zero allocation overhead per step)
            if self._cached_buffer is None or self._cached_buffer.device != device:
                self._cached_buffer = torch.zeros(4, dtype=torch.float32, device=device)

            state_tensor = self._cached_buffer

            if self.is_master:
                state_tensor[0] = float(vvt_gear)
                state_tensor[1] = float(learning_rate)
                state_tensor[2] = 1.0 if wastegate_open else 0.0
                state_tensor[3] = 1.0 if shock_fired else 0.0

            # Master-to-workers collective broadcast (optionally on isolated CUDA stream)
            if self._cuda_stream is not None:
                with torch.cuda.stream(self._cuda_stream):
                    dist.broadcast(state_tensor, src=self.master_rank)
                torch.cuda.current_stream().wait_stream(self._cuda_stream)
            else:
                dist.broadcast(state_tensor, src=self.master_rank)

            sync_gear = int(round(state_tensor[0].item()))
            sync_lr = float(state_tensor[1].item())
            sync_wastegate = bool(state_tensor[2].item() > 0.5)
            sync_shock = bool(state_tensor[3].item() > 0.5)

            self._last_synced_state = (sync_gear, sync_lr, sync_wastegate, sync_shock)
            return self._last_synced_state
        except Exception as e:
            logger.warning("DistributedECUCoordinator broadcast failed: %s. Falling back to local state.", e)
            self._last_synced_state = (vvt_gear, learning_rate, wastegate_open, shock_fired)
            return self._last_synced_state

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns distributed coordinator telemetry."""
        return {
            "is_distributed": self._is_distributed,
            "rank": self._rank,
            "world_size": self._world_size,
            "is_master": self.is_master,
        }
